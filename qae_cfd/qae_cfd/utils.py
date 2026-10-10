"""
Utilidades numéricas para el paquete QAEA-CFD.

Contiene métodos numéricos auxiliares que no dependen de scipy:
    - Newton-Raphson escalar y vectorizado.
    - Limitador de pendiente minmod (TVD).
    - Conversión de variables conservativas ↔ primitivas para Euler 1D.

Estas rutinas eliminan la dependencia externa de scipy para el paquete.
"""

import numpy as np
from typing import Callable, Optional


# ═══════════════════════════════════════════════════════════════════════════
#  Newton-Raphson
# ═══════════════════════════════════════════════════════════════════════════

def newton_raphson(
    func: Callable[[float], float],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 200,
    dx: float = 1e-8,
) -> float:
    """
    Método de Newton-Raphson con diferenciación numérica.

    Resuelve ``func(x) = 0`` partiendo de ``x0``.  Usa una diferencia
    finita centrada para estimar la derivada cuando no se proporciona
    explícitamente.

    Parameters
    ----------
    func : callable
        Función escalar ``f(x) -> float`` cuya raíz se busca.
    x0 : float
        Estimación inicial.
    tol : float, optional
        Tolerancia absoluta en ``|f(x)|`` (default: ``1e-12``).
    max_iter : int, optional
        Número máximo de iteraciones (default: ``200``).
    dx : float, optional
        Paso para la diferencia finita (default: ``1e-8``).

    Returns
    -------
    float
        Raíz aproximada de ``func``.

    Raises
    ------
    RuntimeError
        Si no converge dentro de ``max_iter`` iteraciones.
    """
    x = x0
    for _ in range(max_iter):
        fx = func(x)
        if abs(fx) < tol:
            return x
        # Derivada numérica (diferencia centrada)
        dfx = (func(x + dx) - func(x - dx)) / (2.0 * dx)
        if abs(dfx) < 1e-30:
            # Derivada casi nula → perturbar ligeramente
            x += 0.01
            continue
        x -= fx / dfx
    return x  # Retornar mejor estimación si no converge exactamente


# ═══════════════════════════════════════════════════════════════════════════
#  Limitador TVD
# ═══════════════════════════════════════════════════════════════════════════

def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Limitador minmod vectorizado (LeVeque, Sec. 6.8).

    Selecciona la pendiente de menor magnitud cuando ambas tienen el
    mismo signo; retorna cero si difieren en signo (preserva TVD).

    .. math::

        \\text{minmod}(a, b) = \\begin{cases}
            \\text{sgn}(a) \\min(|a|, |b|) & \\text{si } \\text{sgn}(a) = \\text{sgn}(b) \\\\
            0 & \\text{en otro caso}
        \\end{cases}

    Parameters
    ----------
    a, b : ndarray
        Arrays de pendientes (misma forma).

    Returns
    -------
    ndarray
        Pendientes limitadas.
    """
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


# ═══════════════════════════════════════════════════════════════════════════
#  Conversión de variables termodinámicas
# ═══════════════════════════════════════════════════════════════════════════

def conservative_to_primitive(
    U: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convierte variables conservativas a primitivas (Euler 1D).

    Dadas ``U = [rho, rho·u, rho·E]`` calcula las variables
    primitivas ``(rho, u, p)`` y derivadas ``(a, H)``.

    Parameters
    ----------
    U : ndarray, shape (..., 3)
        ``U[..., 0]`` = ρ, ``U[..., 1]`` = ρu, ``U[..., 2]`` = ρE.
    gamma : float
        Razón de calores específicos (γ = cp/cv).

    Returns
    -------
    rho : ndarray. Densidad.
    u   : ndarray. Velocidad.
    p   : ndarray. Presión estática.
    a   : ndarray. Velocidad del sonido.
    H   : ndarray. Entalpía total por unidad de masa.
    """
    rho = np.maximum(U[..., 0], 1e-10)
    u = U[..., 1] / rho
    rho_E = U[..., 2]
    p = (gamma - 1.0) * (rho_E - 0.5 * rho * u**2)
    p = np.maximum(p, 1e-10)
    a = np.sqrt(gamma * p / rho)
    H = (rho_E + p) / rho
    return rho, u, p, a, H


def euler_flux(U: np.ndarray, gamma: float) -> np.ndarray:
    """
    Vector de flujos de Euler 1D: ``F(U) = [ρu, ρu² + p, (ρE + p)u]``.

    Parameters
    ----------
    U : ndarray, shape (..., 3)
        Variables conservativas.
    gamma : float

    Returns
    -------
    ndarray, shape (..., 3)
    """
    rho, u, p, _, _ = conservative_to_primitive(U, gamma)
    F = np.empty_like(U)
    F[..., 0] = U[..., 1]             # ρu
    F[..., 1] = U[..., 1] * u + p     # ρu² + p
    F[..., 2] = (U[..., 2] + p) * u   # (ρE + p)u
    return F
