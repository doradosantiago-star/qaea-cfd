#!/usr/bin/env python3
"""Verificación de los solvers de Riemann (Roe y HLLC) del paquete qae_cfd
sobre el tubo de choque de Sod (Sod, 1978).

Este script genera la Figura 2.6 de la tesis (fig_riemann_comparison.png y su
variante _es) y reporta los errores L1 contra la solución exacta. Usa las
MISMAS clases ``RoeSolver`` y ``HLLCSolver`` de ``qae_cfd/riemann.py`` que
emplea el solucionador de la tobera, de modo que la verificación certifica el
código realmente usado en la tesis, no una reimplementación.

Esquema: volúmenes finitos de primer orden (Godunov) sobre malla uniforme,
área constante y sin término fuente, que aísla el comportamiento del flujo
numérico. La solución exacta se calcula con el solucionador de Riemann exacto
clásico (iteración de presión, Toro 2009, cap. 4).

Ejecución:  python3 scripts/verificacion_sod.py   (desde qae_cfd/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from qae_cfd.riemann import RoeSolver, HLLCSolver  # noqa: E402

GAMMA = 1.4
# Estados iniciales canónicos de Sod: (rho, u, p)
IZQ = (1.0, 0.0, 1.0)
DER = (0.125, 0.0, 0.1)
T_FIN = 0.2
N_CELDAS = 200
CFL = 0.4
SALIDA = RAIZ.parent / "Imagenes"


# ── Solución exacta del problema de Riemann (Toro 2009, cap. 4) ──────────────
def _f_K(p: float, rho_K: float, p_K: float) -> tuple[float, float]:
    """Función de flujo a través de una onda (choque o rarefacción) y su derivada."""
    a_K = np.sqrt(GAMMA * p_K / rho_K)
    if p > p_K:  # choque
        A = 2.0 / ((GAMMA + 1.0) * rho_K)
        B = (GAMMA - 1.0) / (GAMMA + 1.0) * p_K
        f = (p - p_K) * np.sqrt(A / (p + B))
        df = np.sqrt(A / (B + p)) * (1.0 - (p - p_K) / (2.0 * (B + p)))
    else:  # rarefacción
        f = 2.0 * a_K / (GAMMA - 1.0) * ((p / p_K) ** ((GAMMA - 1.0) / (2.0 * GAMMA)) - 1.0)
        df = 1.0 / (rho_K * a_K) * (p / p_K) ** (-(GAMMA + 1.0) / (2.0 * GAMMA))
    return f, df


def solucion_exacta_sod(x: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perfiles exactos (rho, u, p) del problema de Sod en el tiempo t."""
    rho_L, u_L, p_L = IZQ
    rho_R, u_R, p_R = DER
    a_L = np.sqrt(GAMMA * p_L / rho_L)
    a_R = np.sqrt(GAMMA * p_R / rho_R)

    # Presión de la región estrella por Newton-Raphson
    p_star = 0.5 * (p_L + p_R)
    for _ in range(60):
        f_L, df_L = _f_K(p_star, rho_L, p_L)
        f_R, df_R = _f_K(p_star, rho_R, p_R)
        g = f_L + f_R + (u_R - u_L)
        p_new = p_star - g / (df_L + df_R)
        if abs(p_new - p_star) / p_star < 1e-12:
            p_star = p_new
            break
        p_star = max(p_new, 1e-8)
    f_L, _ = _f_K(p_star, rho_L, p_L)
    f_R, _ = _f_K(p_star, rho_R, p_R)
    u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R - f_L)

    # Densidades estrella (izquierda: rarefacción para Sod; derecha: choque)
    rho_star_L = rho_L * (p_star / p_L) ** (1.0 / GAMMA)
    razon = p_star / p_R
    gm = (GAMMA - 1.0) / (GAMMA + 1.0)
    rho_star_R = rho_R * (razon + gm) / (gm * razon + 1.0)

    # Velocidades de las ondas
    a_star_L = a_L * (p_star / p_L) ** ((GAMMA - 1.0) / (2.0 * GAMMA))
    cabeza = u_L - a_L                       # cabeza de la rarefacción
    cola = u_star - a_star_L                 # cola de la rarefacción
    S_choque = u_R + a_R * np.sqrt((GAMMA + 1.0) / (2.0 * GAMMA) * razon
                                   + (GAMMA - 1.0) / (2.0 * GAMMA))

    rho = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)
    xi = x / t
    for i, s in enumerate(xi):
        if s < cabeza:
            rho[i], u[i], p[i] = rho_L, u_L, p_L
        elif s < cola:  # abanico de rarefacción
            u[i] = 2.0 / (GAMMA + 1.0) * (a_L + 0.5 * (GAMMA - 1.0) * u_L + s)
            a = a_L - 0.5 * (GAMMA - 1.0) * (u[i] - u_L)
            rho[i] = rho_L * (a / a_L) ** (2.0 / (GAMMA - 1.0))
            p[i] = p_L * (a / a_L) ** (2.0 * GAMMA / (GAMMA - 1.0))
        elif s < u_star:
            rho[i], u[i], p[i] = rho_star_L, u_star, p_star
        elif s < S_choque:
            rho[i], u[i], p[i] = rho_star_R, u_star, p_star
        else:
            rho[i], u[i], p[i] = rho_R, u_R, p_R
    return rho, u, p


# ── FVM de primer orden con los flujos del paquete ───────────────────────────
def primitivas_a_conservativas(rho, u, p):
    E = p / (GAMMA - 1.0) + 0.5 * rho * u**2
    return np.stack([rho, rho * u, E], axis=-1)


def correr_sod(solver) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Marcha el tubo de choque hasta T_FIN con el flujo dado (orden 1)."""
    dx = 1.0 / N_CELDAS
    x = (np.arange(N_CELDAS) + 0.5) * dx
    rho = np.where(x < 0.5, IZQ[0], DER[0])
    u = np.where(x < 0.5, IZQ[1], DER[1])
    p = np.where(x < 0.5, IZQ[2], DER[2])
    U = primitivas_a_conservativas(rho, u, p)

    t = 0.0
    while t < T_FIN:
        rho = U[:, 0]
        u = U[:, 1] / rho
        p = (GAMMA - 1.0) * (U[:, 2] - 0.5 * rho * u**2)
        a = np.sqrt(GAMMA * p / rho)
        dt = min(CFL * dx / np.max(np.abs(u) + a), T_FIN - t)

        F = solver.compute_flux(U[:-1], U[1:])          # flujo en interfaces interiores
        U[1:-1] -= dt / dx * (F[1:] - F[:-1])           # celdas de borde: transmisivas
        t += dt

    rho = U[:, 0]
    u = U[:, 1] / rho
    p = (GAMMA - 1.0) * (U[:, 2] - 0.5 * rho * u**2)
    return x, rho, u, p


def main() -> None:
    x, rho_roe, u_roe, p_roe = correr_sod(RoeSolver(GAMMA))
    _, rho_hllc, u_hllc, p_hllc = correr_sod(HLLCSolver(GAMMA))
    rho_ex, u_ex, p_ex = solucion_exacta_sod(x - 0.5, T_FIN)

    def err(num, ex):
        return np.mean(np.abs(num - ex)) / np.mean(np.abs(ex)) * 100.0

    errores = {
        "roe": (err(rho_roe, rho_ex), err(u_roe, u_ex), err(p_roe, p_ex)),
        "hllc": (err(rho_hllc, rho_ex), err(u_hllc, u_ex), err(p_hllc, p_ex)),
    }
    for nombre, (e_r, e_u, e_p) in errores.items():
        print(f"{nombre.upper():5s}  L1 rho = {e_r:.2f}%   L1 u = {e_u:.2f}%   L1 p = {e_p:.2f}%")

    textos = {
        "": dict(rho=r"Density $\rho$", u="Velocity $u$", p="Pressure $p$",
                 exact="Exact (Riemann)", roe="Roe (qae_cfd)", hllc="HLLC (qae_cfd)",
                 xl="$x$"),
        "_es": dict(rho=r"Densidad $\rho$", u="Velocidad $V$", p=r"Presión $P$",  # notación del cap. 2
                    exact="Exacta (Riemann)", roe="Roe (qae_cfd)", hllc="HLLC (qae_cfd)",
                    xl="$x$"),
    }
    plt.rcParams.update({"font.family": "serif", "font.size": 12,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    for suf, t_ in textos.items():
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
        for ax, ex, roe, hllc, ylab, letra in (
            (axes[0], rho_ex, rho_roe, rho_hllc, t_["rho"], "(a)"),
            (axes[1], u_ex, u_roe, u_hllc, t_["u"], "(b)"),
            (axes[2], p_ex, p_roe, p_hllc, t_["p"], "(c)"),
        ):
            ax.plot(x, ex, "k-", lw=1.8, label=t_["exact"])
            ax.plot(x, roe, "--", color="#4477aa", lw=1.4, label=t_["roe"])
            ax.plot(x, hllc, "-.", color="#cc6600", lw=1.4, label=t_["hllc"])
            ax.set(xlabel=t_["xl"], ylabel=ylab, title=f"{letra} {ylab}")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(SALIDA / f"fig_riemann_comparison{suf}.png", dpi=250)
        plt.close(fig)
    print(f"Figuras escritas en {SALIDA}/fig_riemann_comparison[_es].png")


if __name__ == "__main__":
    main()
