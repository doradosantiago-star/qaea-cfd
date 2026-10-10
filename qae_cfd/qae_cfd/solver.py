"""
Solver híbrido FVM + QAE para las ecuaciones de Euler cuasi-1D.

Discretización por volúmenes finitos con reconstrucción MUSCL
(segundo orden TVD). El paso de tiempo se escribe en forma integral y la
media temporal del lado derecho sobre el paso, calculada en M nodos con
un predictor lineal, la estima el circuito de estimación de amplitud
(``qae_grover.GroverMeanQAE``) o la suma determinista
(``baseline.DeterministicMean``).

Las ecuaciones de Euler cuasi-1D en forma conservativa:

.. math::

    \\frac{\\partial \\mathbf{Q}}{\\partial t}
    + \\frac{\\partial (A \\mathbf{F})}{\\partial x} = \\mathbf{S}

donde:

- ``Q = [ρA, ρuA, ρEA]``: variables conservativas.
- ``F = [ρu, ρu² + p, (ρE + p)u]``: flujos de Euler estándar.
- ``S = [0, p·dA/dx, 0]``: término fuente geométrico.

La marcha temporal es explícita, con el paso fijado por la condición
CFL. Con M >= 2 nodos el esquema es de segundo orden en el tiempo (error
local O(Δt³); orden 2.0 medido sobre v' = -v²).

Ref:
    - LeVeque, R. J. (2002). "Finite Volume Methods for
      Hyperbolic Problems". Cambridge University Press.
    - Hirsch, C. (2007). "Numerical Computation of Internal
      and External Flows", 2ª ed.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .riemann import RoeSolver, HLLCSolver, RiemannSolver
from .nozzle import NozzleFlow
from .qae_grover import GroverMeanQAE
from .utils import conservative_to_primitive, minmod

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Resultado del solver
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SolverResult:
    """Resultado de la simulación FVM + QAE."""
    x: np.ndarray          # Posiciones de centros de celda
    rho: np.ndarray        # Densidad final
    u: np.ndarray          # Velocidad final
    p: np.ndarray          # Presión final
    T: np.ndarray          # Temperatura final
    M: np.ndarray          # Número de Mach final
    n_steps: int           # Pasos temporales ejecutados
    elapsed_seconds: float # Tiempo de cómputo


# ═══════════════════════════════════════════════════════════════════════════
#  Solver híbrido
# ═══════════════════════════════════════════════════════════════════════════

class HybridCFDSolver:
    """
    Solver híbrido: Volúmenes Finitos (Riemann) + QAE.

    Algoritmo por paso temporal:
        1. Extraer variables primitivas ``(ρ, u, p, a)``
        2. Reconstrucción MUSCL + minmod (orden 2) o Godunov (orden 1)
        3. Flujos numéricos en interfaces via Riemann solver (Roe/HLLC)
        4. Balance de flujos con factor de área: ``H = A·F̂``
        5. Término fuente geométrico: ``S = [0, p·dA/dx, 0]``
        6. Media temporal del lado derecho sobre el paso: M evaluaciones
           del integrando en los nodos τ_j = jΔt/(M-1) del predictor
           lineal, y la media de cada celda y componente la devuelve
           ``mean_estimator`` (circuito QAE o suma determinista)
        7. Actualización temporal: ``Q += dt × media``
        8. Condiciones de frontera

    Parameters
    ----------
    nozzle : NozzleFlow
        Geometría y solución exacta de la tobera.
    riemann_solver : str
        ``"roe"`` o ``"hllc"`` (default: ``"roe"``).
    n_points : int
        Puntos de malla (default: 31).
    courant : float
        Número de Courant (default: 0.4).
    p_exit : float
        Presión de salida adimensional (default: 0.678).
    order : int
        Orden de precisión espacial: 1 o 2 (default: 2).
    """

    # Mapeo de nombres a clases de Riemann solver
    _RIEMANN_SOLVERS: dict[str, type[RiemannSolver]] = {
        "roe": RoeSolver,
        "hllc": HLLCSolver,
    }

    def __init__(
        self,
        nozzle: NozzleFlow,
        riemann_solver: str = "roe",
        n_points: int = 31,
        courant: float = 0.4,
        p_exit: float = 0.678,
        order: int = 2,
        time_integration: str = "qae_integral",
        n_integral_nodes: int = 4,
        mean_estimator: Optional[GroverMeanQAE] = None,
        mapeo: str = "vector",
    ):
        self.nozzle = nozzle
        self.N = n_points
        self.courant = courant
        self.p_exit = p_exit
        self.gamma = nozzle.gamma
        self.order = order

        # Forma integral del paso (Kacewicz/Gaitan):
        #   U(t+Δt) = U(t) + ∫₀^Δt f[U(τ)] dτ ≈ U(t) + Δt · media_τ(f[ℓ(τ)]),
        # con ℓ(τ) = U(t) + τ·f[U(t)] el predictor lineal (segundo orden
        # global). La MEDIA temporal por celda y componente la devuelve
        # ``mean_estimator``: el circuito QAE (GroverMeanQAE, que ejecuta y
        # mide el circuito en cada estimación) o la suma determinista
        # (DeterministicMean, línea base clásica). El modo histórico
        # ``euler_qae`` (cuantizador valor a valor) se retiró del paquete:
        # ya no forma parte del paquete.
        if time_integration != "qae_integral":
            raise ValueError(
                f"time_integration inválido: {time_integration!r}; "
                "el único modo es 'qae_integral'"
            )
        self.time_integration = time_integration
        self.n_integral_nodes = n_integral_nodes
        # Mapeo afín de las M muestras a [0, 1] antes del circuito:
        #   "vector": cotas = mínimo y máximo de las M muestras de la celda
        #             (la media mapeada cae en 0.5 para integrandos casi
        #             lineales; no ejercita la resolución del estimador);
        #   "campo":  cotas = mínimo y máximo de las muestras de todas las
        #             celdas interiores, por componente y paso (cotas
        #             conocidas por el computador clásico; la media mapeada
        #             se reparte en [0, 1] y el error de estimación queda
        #             multiplicado por el rango del campo).
        if mapeo not in ("vector", "campo"):
            raise ValueError(f"mapeo inválido: {mapeo!r}")
        self.mapeo = mapeo
        self.mean_estimator = mean_estimator or GroverMeanQAE()

        # Inicializar Riemann solver
        solver_key = riemann_solver.lower()
        if solver_key not in self._RIEMANN_SOLVERS:
            raise ValueError(
                f"Solver desconocido: '{riemann_solver}'. "
                f"Opciones: {list(self._RIEMANN_SOLVERS.keys())}"
            )
        self.flux_solver = self._RIEMANN_SOLVERS[solver_key](
            gamma=self.gamma
        )
        self.riemann_name = solver_key.upper()

    # ──────────────────────────────────────────────────────────────────
    #  Reconstrucción MUSCL
    # ──────────────────────────────────────────────────────────────────

    def _muscl_reconstruct(
        self, U: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstrucción MUSCL con limitador minmod (TVD).

        Para orden 1 (Godunov), retorna estados de celda directamente.
        Para orden 2, reconstruye estados izquierdo/derecho en las
        interfaces usando pendientes limitadas.

        Parameters
        ----------
        U : ndarray, shape (N, 3)
            Variables conservativas por unidad de área.

        Returns
        -------
        U_L, U_R : ndarray, shape (N-1, 3)
            Estados izquierdo y derecho en cada interfaz.
        """
        if self.order == 1:
            return U[:-1].copy(), U[1:].copy()

        # Diferencias
        delta_minus = np.zeros_like(U)
        delta_plus = np.zeros_like(U)
        delta_minus[1:] = U[1:] - U[:-1]
        delta_plus[:-1] = U[1:] - U[:-1]

        # Pendientes limitadas
        slope = minmod(delta_minus, delta_plus)

        # Reconstrucción en interfaces
        U_L = U[:-1] + 0.5 * slope[:-1]
        U_R = U[1:] - 0.5 * slope[1:]

        # Positividad de densidad
        U_L[..., 0] = np.maximum(U_L[..., 0], 1e-6)
        U_R[..., 0] = np.maximum(U_R[..., 0], 1e-6)

        return U_L, U_R

    # ──────────────────────────────────────────────────────────────────
    #  Término fuente
    # ──────────────────────────────────────────────────────────────────

    def _source_term(
        self,
        p: np.ndarray,
        A_interfaces: np.ndarray,
        dx: float,
    ) -> np.ndarray:
        """
        Término fuente geométrico: ``S = [0, p·(A_{i+½} − A_{i−½})/dx, 0]``.

        Aproxima la integral del gradiente de presión ponderado por
        el cambio de área sobre cada celda.
        """
        N = len(p)
        S = np.zeros((N, 3))
        S[1:-1, 1] = p[1:-1] * (
            A_interfaces[1:] - A_interfaces[:-1]
        ) / dx
        return S

    # ──────────────────────────────────────────────────────────────────
    #  Condiciones de frontera
    # ──────────────────────────────────────────────────────────────────

    def _apply_bcs(self, Q: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Condiciones de frontera para tobera cuasi-1D.

        Entrada (x=0): Subsónica, condiciones de estancamiento
            ``(ρ₀=1, T₀=1)``, momento extrapolado.
        Salida (x=L): Subsónica, presión fija ``p_exit``,
            densidad y momento extrapolados.
        """
        γ = self.gamma

        # Entrada: estancamiento
        Q[0, 0] = A[0]  # ρ₀ = 1 → ρA = A
        Q[0, 1] = 2.0 * Q[1, 1] - Q[2, 1]  # Extrapolación lineal
        u_in = Q[0, 1] / Q[0, 0]
        Q[0, 2] = Q[0, 0] * (1.0 / (γ - 1.0) + 0.5 * u_in**2)

        # Salida: presión fija
        Q[-1, 0] = 2.0 * Q[-2, 0] - Q[-3, 0]
        Q[-1, 1] = 2.0 * Q[-2, 1] - Q[-3, 1]
        Q[-1, 2] = (
            (self.p_exit * A[-1]) / (γ - 1.0)
            + 0.5 * Q[-1, 1] ** 2 / Q[-1, 0]
        )

        return Q

    # ──────────────────────────────────────────────────────────────────
    #  RHS semidiscreto  f[U] = −∂(A F̂)/∂x + S
    # ──────────────────────────────────────────────────────────────────

    def _compute_rhs(
        self,
        Q: np.ndarray,
        A: np.ndarray,
        A_if: np.ndarray,
        dx: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evalúa el lado derecho semidiscreto (el "oráculo" f[U]):
        MUSCL + Riemann + balance de flujos + fuente geométrica.

        Returns
        -------
        rhs_total : ndarray (N, 3)
        u, a : ndarray (N,)
            Velocidad y velocidad del sonido (para la condición CFL).
        """
        U = Q / A[:, None]
        U[:, 0] = np.maximum(U[:, 0], 1e-6)
        rho, u, p, a, _ = conservative_to_primitive(U, self.gamma)
        U_L, U_R = self._muscl_reconstruct(U)
        F_hat = self.flux_solver.compute_flux(U_L, U_R)
        ledger = getattr(self.mean_estimator, "ledger", None)
        if ledger is not None:
            ledger.charge_flux(len(U_L))
        H = A_if[:, None] * F_hat
        rhs_flux = np.zeros((self.N, 3))
        rhs_flux[1:-1] = -(H[1:] - H[:-1]) / dx
        source = self._source_term(p, A_if, dx)
        return rhs_flux + source, u, a

    # ──────────────────────────────────────────────────────────────────
    #  Solver principal
    # ──────────────────────────────────────────────────────────────────

    def solve(self, n_steps: int = 100) -> SolverResult:
        """
        Ejecuta el time-marching híbrido FVM + QAE.

        Parameters
        ----------
        n_steps : int
            Número de pasos temporales (default: 100).

        Returns
        -------
        SolverResult
            Contiene ``x``, ``rho``, ``u``, ``p``, ``T``, ``M``,
            ``n_steps``, ``elapsed_seconds``.
        """
        γ = self.gamma
        N = self.N
        L = self.nozzle.length
        dx = L / (N - 1)

        # ── Malla ──
        x = np.linspace(0.0, L, N)
        A = self.nozzle.area(x)
        x_if = 0.5 * (x[:-1] + x[1:])  # Interfaces
        A_if = self.nozzle.area(x_if)

        # ── Condición inicial ──
        logger.info("Generando condición inicial exacta (SI-5D)...")
        rho_init, u_init, T_init = self.nozzle.initial_condition(
            x, self.p_exit
        )

        # Variables conservativas Q = [ρA, ρuA, ρEA]
        Q = np.zeros((N, 3))
        Q[:, 0] = rho_init * A
        Q[:, 1] = rho_init * u_init * A
        E_init = T_init / (γ - 1.0) + 0.5 * u_init**2
        Q[:, 2] = rho_init * E_init * A

        label = f"FVM-{self.riemann_name} + QAE (orden {self.order})"
        logger.info("Iniciando %s, %d pasos...", label, n_steps)
        t0 = time.time()

        for step in range(n_steps):
            # 1. RHS semidiscreto en t (una llamada al oráculo f[U])
            rhs_total, u, a = self._compute_rhs(Q, A, A_if, dx)

            # 2. Paso temporal (CFL)
            dt = self.courant * dx / np.max(np.abs(u) + a)

            if True:  # forma integral, único modo del paquete
                # ── Forma integral (Kacewicz/Gaitan) ──
                # U(t+Δt) = U(t) + ∫₀^Δt f[U(τ)] dτ
                #         ≈ U(t) + Δt · media_τ( f[ℓ(τ)] )
                # con ℓ(τ) = U(t) + τ·f[U(t)] el predictor lineal local
                # (Taylor de primer orden de la trayectoria; el esquema
                # resultante es de segundo orden global). El nodo τ_0 = 0
                # repite la evaluación f[U(t)] del predictor: M + 1
                # evaluaciones del integrando por paso.
                # La MEDIA temporal por celda/componente la devuelve
                # mean_estimator: una sola estimación agrega las M
                # evaluaciones del integrando.
                Mn = self.n_integral_nodes
                taus = np.linspace(0.0, 1.0, Mn) * dt
                G = np.empty((Mn, N, 3))
                for k, tau in enumerate(taus):
                    Q_pred = self._apply_bcs(Q + tau * rhs_total, A)
                    G[k] = self._compute_rhs(Q_pred, A, A_if, dx)[0]
                ledger = getattr(self.mean_estimator, "ledger", None)
                for comp in range(3):
                    bounds = None
                    if self.mapeo == "campo":
                        campo = G[:, 1:N - 1, comp]
                        bounds = (float(campo.min()), float(campo.max()))
                        if ledger is not None:
                            ledger.charge_bounds(2 * campo.size)
                    for i in range(1, N - 1):
                        integral_mean = (
                            self.mean_estimator.estimate_mean_general(
                                G[:, i, comp], bounds=bounds
                            )
                        )
                        Q[i, comp] += dt * integral_mean

            # 3. Condiciones de frontera
            Q = self._apply_bcs(Q, A)

            if step % max(1, n_steps // 10) == 0:
                M_local = u / np.maximum(a, 1e-10)
                logger.info(
                    "  Paso %d/%d: Max Mach = %.3f",
                    step, n_steps, np.max(M_local),
                )

        elapsed = time.time() - t0
        logger.info("Completado en %.2f s", elapsed)

        # ── Resultados finales ──
        U_final = Q / A[:, None]
        rho_f = np.maximum(U_final[:, 0], 1e-10)
        u_f = U_final[:, 1] / rho_f
        p_f = (γ - 1.0) * (U_final[:, 2] - 0.5 * rho_f * u_f**2)
        p_f = np.maximum(p_f, 1e-10)
        T_f = p_f / rho_f
        M_f = u_f / np.sqrt(γ * np.maximum(T_f, 1e-10))

        return SolverResult(
            x=x,
            rho=rho_f,
            u=u_f,
            p=p_f,
            T=T_f,
            M=M_f,
            n_steps=n_steps,
            elapsed_seconds=elapsed,
        )

    def __repr__(self) -> str:
        return (
            f"HybridCFDSolver(\n"
            f"  riemann = {self.riemann_name},\n"
            f"  N = {self.N},\n"
            f"  CFL = {self.courant},\n"
            f"  order = {self.order},\n"
            f"  mean_estimator = {self.mean_estimator}\n"
            f")"
        )
