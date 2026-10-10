"""
Solvers de Riemann aproximados para las ecuaciones de Euler 1D.

Implementa dos métodos ampliamente usados en CFD:

- **Roe** (1981): Linealización del problema de Riemann usando
  promedios ponderados por √ρ.  Incluye la corrección de entropía
  de Harten-Hyman para evitar expansiones de rarefacción no-físicas.

- **HLLC** (Toro, 1994): Solver de tres ondas que captura
  discontinuidades de contacto.

Ref:
    - Roe, P. L. (1981). J. Comput. Phys., 43, 357-372.
    - Toro, E. F. (1994). Shock Waves, 4, 25-34.
    - LeVeque, R. J. (2002). "Finite Volume Methods for Hyperbolic
      Problems". Cambridge University Press.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .utils import conservative_to_primitive, euler_flux


# ═══════════════════════════════════════════════════════════════════════════
#  Clase base abstracta
# ═══════════════════════════════════════════════════════════════════════════

class RiemannSolver(ABC):
    """Interfaz abstracta para solvers de Riemann aproximados."""

    def __init__(self, gamma: float = 1.4):
        self.gamma = gamma

    @abstractmethod
    def compute_flux(
        self, U_L: np.ndarray, U_R: np.ndarray
    ) -> np.ndarray:
        """
        Calcula el flujo numérico en las interfaces.

        Parameters
        ----------
        U_L : ndarray, shape (N, 3)
            Estados conservativos a la izquierda de cada interfaz.
        U_R : ndarray, shape (N, 3)
            Estados conservativos a la derecha de cada interfaz.

        Returns
        -------
        ndarray, shape (N, 3)
            Flujo numérico ``F̂`` en cada interfaz.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  Solver de Roe
# ═══════════════════════════════════════════════════════════════════════════

class RoeSolver(RiemannSolver):
    """
    Solver de Riemann aproximado de Roe (1981).

    El flujo de Roe se construye como:

    .. math::

        \\hat{F}_{i+1/2} = \\frac{1}{2}(F_L + F_R)
        - \\frac{1}{2} \\sum_{k=1}^{3} |\\tilde{\\lambda}_k|
          \\alpha_k \\mathbf{r}_k

    donde ``λ̃_k`` son los eigenvalores de la matriz de Roe,
    ``α_k`` las intensidades de onda, y ``r_k`` los eigenvectores
    derechos del sistema de Euler linealizado.

    La corrección de entropía de Harten-Hyman reemplaza
    ``|λ_k|`` por una función suave cerca de puntos sónicos.

    Parameters
    ----------
    gamma : float
        Razón de calores específicos (default: 1.4).
    """

    def compute_flux(
        self, U_L: np.ndarray, U_R: np.ndarray
    ) -> np.ndarray:
        gamma = self.gamma

        # Primitivas a cada lado de la interfaz
        rho_L, u_L, p_L, a_L, H_L = conservative_to_primitive(U_L, gamma)
        rho_R, u_R, p_R, a_R, H_R = conservative_to_primitive(U_R, gamma)

        # ── Promedios de Roe (ponderados por √ρ) ──
        sqrt_L = np.sqrt(rho_L)
        sqrt_R = np.sqrt(rho_R)
        w = sqrt_L + sqrt_R

        u_roe = (sqrt_L * u_L + sqrt_R * u_R) / w
        H_roe = (sqrt_L * H_L + sqrt_R * H_R) / w
        a_sq = np.maximum(
            (gamma - 1.0) * (H_roe - 0.5 * u_roe**2), 1e-10
        )
        a_roe = np.sqrt(a_sq)

        # ── Eigenvalores ──
        lam1 = u_roe - a_roe
        lam2 = u_roe
        lam3 = u_roe + a_roe

        # ── Corrección de entropía (Harten-Hyman) ──
        eps1 = np.maximum(0.0, np.maximum(
            lam1 - (u_L - a_L), (u_R - a_R) - lam1
        ))
        eps3 = np.maximum(0.0, np.maximum(
            lam3 - (u_L + a_L), (u_R + a_R) - lam3
        ))

        abs_lam1 = np.where(
            np.abs(lam1) >= eps1,
            np.abs(lam1),
            (lam1**2 + eps1**2) / (2.0 * eps1 + 1e-30),
        )
        abs_lam2 = np.abs(lam2)
        abs_lam3 = np.where(
            np.abs(lam3) >= eps3,
            np.abs(lam3),
            (lam3**2 + eps3**2) / (2.0 * eps3 + 1e-30),
        )

        # ── Intensidades de onda (wave strengths α_k) ──
        dp = p_R - p_L
        du = u_R - u_L
        drho = rho_R - rho_L

        alpha1 = (dp - rho_L * a_roe * du) / (2.0 * a_sq)  # Nota: usa rho_roe
        alpha2 = drho - dp / a_sq
        alpha3 = (dp + rho_L * a_roe * du) / (2.0 * a_sq)

        # Corrección: usar promedios de Roe para las wave strengths
        rho_roe = sqrt_L * sqrt_R
        alpha1 = (dp - rho_roe * a_roe * du) / (2.0 * a_sq)
        alpha3 = (dp + rho_roe * a_roe * du) / (2.0 * a_sq)

        # ── Término de disipación: Σ |λ_k| α_k r_k ──
        diss = np.zeros_like(U_L)

        # Onda 1 (acústica izquierda): r₁ = [1, ũ−ã, H̃−ũã]
        diss[..., 0] += abs_lam1 * alpha1
        diss[..., 1] += abs_lam1 * alpha1 * (u_roe - a_roe)
        diss[..., 2] += abs_lam1 * alpha1 * (H_roe - u_roe * a_roe)

        # Onda 2 (contacto/entropía): r₂ = [1, ũ, ½ũ²]
        diss[..., 0] += abs_lam2 * alpha2
        diss[..., 1] += abs_lam2 * alpha2 * u_roe
        diss[..., 2] += abs_lam2 * alpha2 * 0.5 * u_roe**2

        # Onda 3 (acústica derecha): r₃ = [1, ũ+ã, H̃+ũã]
        diss[..., 0] += abs_lam3 * alpha3
        diss[..., 1] += abs_lam3 * alpha3 * (u_roe + a_roe)
        diss[..., 2] += abs_lam3 * alpha3 * (H_roe + u_roe * a_roe)

        # ── Flujo de Roe final ──
        F_L = euler_flux(U_L, gamma)
        F_R = euler_flux(U_R, gamma)

        return 0.5 * (F_L + F_R) - 0.5 * diss


# ═══════════════════════════════════════════════════════════════════════════
#  Solver HLLC
# ═══════════════════════════════════════════════════════════════════════════

class HLLCSolver(RiemannSolver):
    """
    Solver de Riemann HLLC (Toro, 1994).

    Resuelve el problema de Riemann con tres ondas: dos ondas
    exteriores ``S_L``, ``S_R`` (estimación de Davis) y una onda
    de contacto ``S_*``.

    El flujo numérico se selecciona según la región:

    .. math::

        \\hat{F} = \\begin{cases}
            F_L                     & S_L \\geq 0 \\\\
            F_L + S_L(U^*_L - U_L) & S_L < 0 \\leq S_* \\\\
            F_R + S_R(U^*_R - U_R) & S_* < 0 \\leq S_R \\\\
            F_R                     & S_R < 0
        \\end{cases}

    Parameters
    ----------
    gamma : float
        Razón de calores específicos (default: 1.4).
    """

    def compute_flux(
        self, U_L: np.ndarray, U_R: np.ndarray
    ) -> np.ndarray:
        gamma = self.gamma

        # Primitivas
        rho_L, u_L, p_L, a_L, _ = conservative_to_primitive(U_L, gamma)
        rho_R, u_R, p_R, a_R, _ = conservative_to_primitive(U_R, gamma)

        # ── Velocidades de onda (Davis, 1988) ──
        S_L = np.minimum(u_L - a_L, u_R - a_R)
        S_R = np.maximum(u_L + a_L, u_R + a_R)

        # ── Velocidad de contacto S_* ──
        num = (
            p_R - p_L
            + rho_L * u_L * (S_L - u_L)
            - rho_R * u_R * (S_R - u_R)
        )
        den = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
        S_star = num / (den + 1e-30)

        # ── Flujos físicos ──
        F_L = euler_flux(U_L, gamma)
        F_R = euler_flux(U_R, gamma)

        # ── Estados intermedios U*_K (Toro, Ec. 10.33) ──
        E_L = U_L[..., 2] / np.maximum(rho_L, 1e-10)
        E_R = U_R[..., 2] / np.maximum(rho_R, 1e-10)

        coeff_L = rho_L * (S_L - u_L) / (S_L - S_star + 1e-30)
        coeff_R = rho_R * (S_R - u_R) / (S_R - S_star + 1e-30)

        U_star_L = np.empty_like(U_L)
        U_star_L[..., 0] = coeff_L
        U_star_L[..., 1] = coeff_L * S_star
        U_star_L[..., 2] = coeff_L * (
            E_L
            + (S_star - u_L)
            * (S_star + p_L / (rho_L * (S_L - u_L) + 1e-30))
        )

        U_star_R = np.empty_like(U_R)
        U_star_R[..., 0] = coeff_R
        U_star_R[..., 1] = coeff_R * S_star
        U_star_R[..., 2] = coeff_R * (
            E_R
            + (S_star - u_R)
            * (S_star + p_R / (rho_R * (S_R - u_R) + 1e-30))
        )

        # ── Flujos intermedios (Toro, Ec. 10.26) ──
        F_star_L = F_L + S_L[..., None] * (U_star_L - U_L)
        F_star_R = F_R + S_R[..., None] * (U_star_R - U_R)

        # ── Selección del flujo ──
        return np.where(
            (S_L >= 0)[..., None],
            F_L,
            np.where(
                (S_star >= 0)[..., None],
                F_star_L,
                np.where(
                    (S_R >= 0)[..., None],
                    F_star_R,
                    F_R,
                ),
            ),
        )
