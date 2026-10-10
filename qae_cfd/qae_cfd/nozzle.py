"""
Geometría de tobera convergente-divergente y solución exacta del flujo cuasi-1D.

Implementa las relaciones isentrópicas área-Mach (Anderson, 2003) y las
condiciones de salto de Rankine-Hugoniot para una onda de choque normal.

El perfil de Anderson es:

.. math::

    A(x) = 1 + 2.2\\,(x - 1.5)^2, \\qquad x \\in [0, 3]

con garganta (``A* = 1``) en ``x = 1.5``.

La relación isentrópica área-Mach:

.. math::

    \\left(\\frac{A}{A^*}\\right)^2
    = \\frac{1}{M^2} \\left[
        \\frac{2}{\\gamma+1}
        \\left(1 + \\frac{\\gamma-1}{2} M^2\\right)
    \\right]^{(\\gamma+1)/(\\gamma-1)}

se resuelve con Newton-Raphson (sin scipy).

Ref:
    - Anderson, J. D. (2003). "Modern Compressible Flow", 2ª ed.
    - Pletcher, Tannehill & Anderson (2013). Cap. 14.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import newton_raphson


@dataclass
class ShockJump:
    """Resultado de las relaciones de salto de choque normal."""
    M_post: float     # Mach post-choque
    p_ratio: float    # p₂/p₁
    rho_ratio: float  # ρ₂/ρ₁
    T_ratio: float    # T₂/T₁
    p0_loss: float    # Factor de pérdida de presión de estancamiento


class NozzleFlow:
    """
    Flujo cuasi-1D en tobera convergente-divergente de Anderson.

    Resuelve analíticamente el perfil isentrópico (subsónico y
    supersónico) y aplica las relaciones de Rankine-Hugoniot para
    modelar una onda de choque normal en la sección divergente.

    Parameters
    ----------
    gamma : float
        Razón de calores específicos (default: 1.4 para aire).
    x_shock : float
        Posición axial de la onda de choque normal (default: 2.1).
    length : float
        Longitud total de la tobera (default: 3.0).
    """

    def __init__(
        self,
        gamma: float = 1.4,
        x_shock: float = 2.1,
        length: float = 3.0,
    ):
        self.gamma = gamma
        self.x_shock = x_shock
        self.length = length

    # ──────────────────────────────────────────────────────────────────
    #  Geometría
    # ──────────────────────────────────────────────────────────────────

    def area(self, x: np.ndarray | float) -> np.ndarray | float:
        """Área de la sección transversal: ``A(x) = 1 + 2.2(x − 1.5)²``."""
        return 1.0 + 2.2 * (x - 1.5) ** 2

    def dAdx(self, x: np.ndarray | float) -> np.ndarray | float:
        """Derivada del área: ``dA/dx = 4.4(x − 1.5)``."""
        return 4.4 * (x - 1.5)

    # ──────────────────────────────────────────────────────────────────
    #  Relaciones isentrópicas
    # ──────────────────────────────────────────────────────────────────

    def _area_mach_residual(self, M: float, ratio: float) -> float:
        """
        Residuo de la relación área-Mach isentrópica.

        ``f(M) = (1/M²) [2/(γ+1) (1 + (γ-1)/2 M²)]^{(γ+1)/(γ-1)} - (A/A*)²``

        Se busca ``f(M) = 0``.
        """
        gamma = self.gamma
        if M <= 0:
            return 1e5
        term = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * M**2)
        exponent = (gamma + 1.0) / (gamma - 1.0)
        return (1.0 / M**2) * term**exponent - ratio**2

    def solve_mach(
        self,
        area_ratio: float,
        subsonic: bool = True,
    ) -> float:
        """
        Resuelve la relación área-Mach para un dado ``A/A*``.

        Parameters
        ----------
        area_ratio : float
            Razón de áreas ``A / A*``.
        subsonic : bool
            Si ``True``, busca la raíz subsónica (``M < 1``);
            si ``False``, la supersónica (``M > 1``).

        Returns
        -------
        float
            Número de Mach.
        """
        guess = 0.2 if subsonic else 2.5
        M = newton_raphson(
            lambda M: self._area_mach_residual(M, area_ratio),
            x0=guess,
        )
        return max(M, 1e-6)

    # ──────────────────────────────────────────────────────────────────
    #  Relaciones de choque normal (Rankine-Hugoniot)
    # ──────────────────────────────────────────────────────────────────

    def shock_jump(self, M1: float) -> ShockJump:
        """
        Relaciones de salto de Rankine-Hugoniot para choque normal.

        Dados ``M₁ > 1`` (Mach pre-choque), calcula:
            - ``M₂``: Mach post-choque
            - ``p₂/p₁``, ``ρ₂/ρ₁``, ``T₂/T₁``
            - Factor de pérdida de presión de estancamiento ``p₀₂/p₀₁``

        Ref: Anderson (2003), Ec. 3.57-3.61.
        """
        γ = self.gamma
        M1_sq = M1**2

        # Razones de salto
        rho_ratio = ((γ + 1.0) * M1_sq) / (2.0 + (γ - 1.0) * M1_sq)
        p_ratio = 1.0 + (2.0 * γ / (γ + 1.0)) * (M1_sq - 1.0)
        T_ratio = p_ratio / rho_ratio

        # Mach post-choque
        M2_sq = (1.0 + 0.5 * (γ - 1.0) * M1_sq) / (
            γ * M1_sq - 0.5 * (γ - 1.0)
        )
        M_post = np.sqrt(max(M2_sq, 1e-10))

        # Pérdida de presión de estancamiento
        p0_loss = (
            rho_ratio ** (γ / (γ - 1.0))
            * p_ratio ** (-1.0 / (γ - 1.0))
        )

        return ShockJump(
            M_post=M_post,
            p_ratio=p_ratio,
            rho_ratio=rho_ratio,
            T_ratio=T_ratio,
            p0_loss=p0_loss,
        )

    # ──────────────────────────────────────────────────────────────────
    #  Solución exacta con choque
    # ──────────────────────────────────────────────────────────────────

    def exact_solution(
        self, x_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solución exacta del flujo isentrópico + choque normal.

        Procedimiento:
            1. Zona convergente (x < 1.5): Raíz subsónica.
            2. Zona divergente pre-choque (1.5 ≤ x < x_shock): Raíz supersónica.
            3. Choque normal en ``x_shock``: Aplicar Rankine-Hugoniot.
            4. Zona divergente post-choque (x ≥ x_shock): Raíz subsónica
               con presión de estancamiento reducida.

        Parameters
        ----------
        x_arr : ndarray
            Posiciones axiales.

        Returns
        -------
        M : ndarray. Número de Mach.
        rho : ndarray. Densidad adimensional.
        T : ndarray. Temperatura adimensional.
        """
        γ = self.gamma
        x_shock = self.x_shock

        M_out = np.zeros_like(x_arr)
        rho_out = np.zeros_like(x_arr)
        T_out = np.zeros_like(x_arr)

        P0, T0, A_star = 1.0, 1.0, 1.0
        shock_done = False

        for i, x in enumerate(x_arr):
            A = self.area(x)
            ratio = A / A_star

            if x < 1.5:
                M = self.solve_mach(ratio, subsonic=True)
            elif x < x_shock:
                M = self.solve_mach(ratio, subsonic=False)
            elif not shock_done:
                # Pre-choque: Mach supersónico
                M_pre = self.solve_mach(ratio, subsonic=False)
                sj = self.shock_jump(M_pre)
                P0 *= sj.p0_loss
                A_star = 1.0 / sj.p0_loss
                shock_done = True
                M = sj.M_post
            else:
                # Post-choque: subsónico con A* modificado
                M = self.solve_mach(A / A_star, subsonic=True)

            T_static = T0 / (1.0 + 0.5 * (γ - 1.0) * M**2)
            rho_static = (
                P0 * (T_static / T0) ** (γ / (γ - 1.0))
            ) / T_static

            M_out[i] = M
            rho_out[i] = rho_static
            T_out[i] = T_static

        return M_out, rho_out, T_out

    # ──────────────────────────────────────────────────────────────────
    #  Condición inicial para FVM
    # ──────────────────────────────────────────────────────────────────

    def initial_condition(
        self,
        x_arr: np.ndarray,
        p_exit: float = 0.678,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Condición inicial basada en la solución exacta (SI-5D del paper).

        Parameters
        ----------
        x_arr : ndarray
            Posiciones de la malla.
        p_exit : float
            Presión de salida adimensional.

        Returns
        -------
        rho : ndarray. Densidad.
        u : ndarray. Velocidad.
        T : ndarray. Temperatura.
        """
        γ = self.gamma
        x_shock = self.x_shock

        rho_out = np.zeros_like(x_arr)
        u_out = np.zeros_like(x_arr)
        T_out = np.zeros_like(x_arr)

        P0, T0 = 1.0, 1.0
        A_throat = 1.0
        shock_passed = False

        for i, x in enumerate(x_arr):
            A = self.area(x)
            ratio = A / A_throat

            if x < 1.5:
                M = self.solve_mach(ratio, subsonic=True)
                p_stag = P0
            elif x < x_shock:
                M = self.solve_mach(ratio, subsonic=False)
                p_stag = P0
            else:
                if not shock_passed:
                    M_pre = self.solve_mach(ratio, subsonic=False)
                    sj = self.shock_jump(M_pre)
                    shock_passed = True

                M = self.solve_mach(ratio, subsonic=True)
                isentropic_factor = (
                    1.0 + 0.5 * (γ - 1.0) * M**2
                ) ** (-γ / (γ - 1.0))
                p_stag = p_exit / isentropic_factor

            T = T0 / (1.0 + 0.5 * (γ - 1.0) * M**2)
            rho = (p_stag / P0) * T ** (1.0 / (γ - 1.0))
            u = M * np.sqrt(T)

            rho_out[i] = rho
            u_out[i] = u
            T_out[i] = T

        return rho_out, u_out, T_out

    def __repr__(self) -> str:
        return (
            f"NozzleFlow(gamma={self.gamma}, "
            f"x_shock={self.x_shock}, length={self.length})"
        )
