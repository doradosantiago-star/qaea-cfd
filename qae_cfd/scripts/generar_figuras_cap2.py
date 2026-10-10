#!/usr/bin/env python3
"""Regenera en español las cuatro figuras conceptuales del capítulo 2.

Las versiones anteriores (sin sufijo _es) estaban en inglés, dibujaban la
geometría con y = A(x)/2 en metros (la pared de 73 grados que el capítulo 3
descarta) y una de ellas afirmaba en su pie que "la presión ambiente empuja la
pared hacia adentro", cosa que el modelo no contiene. Aquí todo es
adimensional y cada rótulo corresponde a lo que el texto dice.

Uso (desde qae_cfd/):
    python3 scripts/generar_figuras_cap2.py

Salidas en ../Imagenes/: fig_wall_force_cases_es.png, fig_full_nozzle_cv_es.png,
fig_nozzle_regimes_es.png, fig_riemann_waves_es.png
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ.parent / "Imagenes"

AZUL, ROJO, GRIS, VERDE, TINTA, NARANJA, MORADO = (
    "#1F3A93", "#C8102E", "#6B7280", "#1B7F4B", "#1F2937", "#D2691E", "#7B5EA7")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "mathtext.fontset": "dejavusans", "axes.edgecolor": "#9CA3AF",
                     "savefig.dpi": 220, "savefig.bbox": "tight"})

GAMMA = 1.4
area = lambda x: 1.0 + 2.2 * (x - 1.5) ** 2


def guardar(fig, nombre):
    fig.savefig(OUT / nombre, facecolor="white")
    plt.close(fig)
    print("escrito", nombre)


def mach_de_area(razon, supersonico):
    f = lambda M: (1 / M) * ((2 / (GAMMA + 1)) * (1 + (GAMMA - 1) / 2 * M ** 2)) ** \
        ((GAMMA + 1) / (2 * (GAMMA - 1))) - razon
    return brentq(f, 1.0, 8.0) if supersonico else brentq(f, 1e-6, 1.0)


# ------------------------------------------------ fuerza de pared: tres casos
def volumen(ax, y_izq, y_der, color, sombra):
    """Rebanada de conducto entre x=0 y x=1 con semialturas y_izq, y_der."""
    pol = Polygon([(0, -y_izq), (1, -y_der), (1, y_der), (0, y_izq)],
                  closed=True, facecolor=sombra, edgecolor="none")
    ax.add_patch(pol)
    ax.plot([0, 1], [y_izq, y_der], color=TINTA, lw=2.2)
    ax.plot([0, 1], [-y_izq, -y_der], color=TINTA, lw=2.2)
    ax.plot([0, 0], [-y_izq, y_izq], color=GRIS, lw=1, ls="--")
    ax.plot([1, 1], [-y_der, y_der], color=GRIS, lw=1, ls="--")
    ax.annotate("", (1.25, 0), (-0.2, 0), arrowprops=dict(arrowstyle="-|>", color=TINTA, lw=1.2))
    ax.text(1.27, 0.0, "$x$", va="center", fontsize=15)


def flecha(ax, origen, destino, color, lw=2.2):
    ax.add_patch(FancyArrowPatch(origen, destino, arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=lw))


fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
# (a) volumen de control diferencial
ax = axes[0]
volumen(ax, 0.75, 0.9, TINTA, "#EEF2FB")
xm, ym = 0.5, 0.5 * (0.75 + 0.9)
pend = 0.9 - 0.75
n = np.array([-pend, 1.0]); n /= np.linalg.norm(n)
flecha(ax, (xm, ym), (xm + 0.32 * n[0], ym + 0.32 * n[1]), ROJO)
ax.text(xm + 0.36 * n[0] + 0.02, ym + 0.36 * n[1], r"$d\mathbf{S}$", color=ROJO, fontsize=16)
flecha(ax, (xm, ym), (xm - 0.32 * n[0], ym - 0.32 * n[1]), AZUL)
ax.text(0.02, 0.32, r"$d\mathbf{F}_{\mathrm{pared}} = -P\,d\mathbf{S}$", color=AZUL, fontsize=15)
flecha(ax, (0.02, -0.05), (0.2, -0.05), VERDE, lw=1.6)
ax.text(0.02, -0.24, r"$PA|_x$", color=VERDE, fontsize=14)
flecha(ax, (0.98, -0.05), (0.8, -0.05), VERDE, lw=1.6)
ax.text(0.66, -0.24, r"$PA|_{x+dx}$", color=VERDE, fontsize=14)
ax.text(0.0, -1.14, "$A(x)$", fontsize=14); ax.text(0.76, -1.14, "$A(x+dx)$", fontsize=14)
ax.set_title("(a) Volumen de control diferencial", fontsize=15)
# (b) contraccion
ax = axes[1]
volumen(ax, 0.95, 0.65, TINTA, "#FDF3E7")
xm, ym = 0.5, 0.8; pend = 0.65 - 0.95
n = np.array([-pend, 1.0]); n /= np.linalg.norm(n)
flecha(ax, (xm, ym), (xm - 0.3 * n[0], ym - 0.3 * n[1]), AZUL)
flecha(ax, (xm, -ym), (xm - 0.3 * n[0], -ym + 0.3 * n[1]), AZUL)
flecha(ax, (0.5, 0.95), (0.2, 0.95), ROJO)
ax.text(0.08, 1.06, r"$dF_x = P\,\dfrac{dA}{dx}\,dx < 0$", color=ROJO, fontsize=14)
ax.text(0.26, 0.26, r"$d\mathbf{F}_{\mathrm{pared}}$", color=AZUL, fontsize=15)
ax.set_title("(b) Contracción: $dA/dx < 0$", fontsize=15)
ax.text(0.5, -1.22, "La pared empuja el fluido aguas arriba.", ha="center", fontsize=13, style="italic", color=TINTA)
# (c) expansion
ax = axes[2]
volumen(ax, 0.65, 0.95, TINTA, "#E7F3EC")
xm, ym = 0.5, 0.8; pend = 0.95 - 0.65
n = np.array([-pend, 1.0]); n /= np.linalg.norm(n)
flecha(ax, (xm, ym), (xm - 0.3 * n[0], ym - 0.3 * n[1]), AZUL)
flecha(ax, (xm, -ym), (xm - 0.3 * n[0], -ym + 0.3 * n[1]), AZUL)
flecha(ax, (0.5, 0.95), (0.8, 0.95), ROJO)
ax.text(0.36, 1.06, r"$dF_x = P\,\dfrac{dA}{dx}\,dx > 0$", color=ROJO, fontsize=14)
ax.text(0.26, 0.26, r"$d\mathbf{F}_{\mathrm{pared}}$", color=AZUL, fontsize=15)
ax.set_title("(c) Expansión: $dA/dx > 0$", fontsize=15)
ax.text(0.5, -1.22, "La pared empuja el fluido aguas abajo.", ha="center", fontsize=13, style="italic", color=TINTA)
for ax in axes:
    ax.set(xlim=(-0.25, 1.35), ylim=(-1.35, 1.35)); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
guardar(fig, "fig_wall_force_cases_es.png")


# ------------------------------------------------ tobera completa con el volumen de control
x = np.linspace(0, 3, 400)
h = area(x) / 2                     # semialtura adimensional: A(x) con A* = 1
fig, ax = plt.subplots(figsize=(9, 6.9))
ax.fill_between(x, -h, h, color="#EEF2FB")
ax.plot(x, h, color=TINTA, lw=2.4, label="Pared de la tobera, $A(x) = 1 + 2.2\\,(x-1.5)^2$")
ax.plot(x, -h, color=TINTA, lw=2.4)
ax.plot(x, h * 0.985, color=NARANJA, lw=1.8, ls="--", label="Volumen de control $\\Omega$")
ax.plot(x, -h * 0.985, color=NARANJA, lw=1.8, ls="--")
ax.plot([0, 0], [-h[0], h[0]], color=NARANJA, lw=1.8, ls="--")
ax.plot([3, 3], [-h[-1], h[-1]], color=NARANJA, lw=1.8, ls="--")
primera = True
for xf in (0.45, 1.05, 2.0, 2.6):
    hf = area(xf) / 2; pend = 2.2 * (xf - 1.5)
    n = np.array([-pend, 1.0]); n /= np.linalg.norm(n)
    for sg in (1, -1):
        p0 = (xf, sg * hf); p1 = (xf - 0.28 * n[0], sg * (hf - 0.28 * n[1]))
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, color=MORADO, lw=2,
                                     label="Fuerza de presión de pared $d\\mathbf{F}_{\\mathrm{pared}} = -P\\,d\\mathbf{S}$" if primera else None))
        primera = False
ax.axvline(1.5, color=VERDE, ls=":", lw=1.4)
ax.text(1.5, -3.0, "garganta, $A^* = 1$", ha="center", color=VERDE, fontsize=11, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.annotate("", (2.2, 0), (0.8, 0), arrowprops=dict(arrowstyle="-|>", color=TINTA, lw=2))
ax.text(1.5, 0.12, "flujo", ha="center", fontsize=12, style="italic")
ax.text(-0.15, 3.08, "cara de entrada, $A(0) = 5.95$", color=AZUL, fontsize=11, ha="left")
ax.text(3.15, 3.08, "cara de salida, $A(3) = 5.95$", color=AZUL, fontsize=11, ha="right")
ax.set(xlabel="coordenada axial $x$ (adimensional)", ylabel="semialtura $A(x)/2$ (adimensional)",
       xlim=(-0.2, 3.2), ylim=(-3.3, 3.75))
ax.legend(loc="upper center", frameon=False, fontsize=11, ncol=1, bbox_to_anchor=(0.5, -0.13))
ax.grid(alpha=0.15)
fig.tight_layout()
guardar(fig, "fig_full_nozzle_cv_es.png")


# ------------------------------------------------ geometria y perfil de Mach
xs = 2.1
M1 = mach_de_area(area(xs), True)
p0_ratio = (((GAMMA + 1) * M1 ** 2) / ((GAMMA - 1) * M1 ** 2 + 2)) ** (GAMMA / (GAMMA - 1)) * \
           (1 + 2 * GAMMA / (GAMMA + 1) * (M1 ** 2 - 1)) ** (-1 / (GAMMA - 1))
A2 = 1.0 / p0_ratio
M_choque = np.empty_like(x); M_isen = np.empty_like(x)
for i, xi in enumerate(x):
    M_isen[i] = mach_de_area(area(xi), xi >= 1.5)
    if xi < xs:
        M_choque[i] = M_isen[i]
    else:
        M_choque[i] = mach_de_area(area(xi) / A2, False)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 1.25]})
a1.fill_between(x, -h, h, color="#EEF2FB")
a1.plot(x, h, color=AZUL, lw=2.4); a1.plot(x, -h, color=AZUL, lw=2.4)
a1.plot([1.5, 1.5], [-3.2, 2.3], color=GRIS, ls="--", lw=1.2)
a1.axvline(xs, color=ROJO, lw=2.4, alpha=0.8)
a1.text(0.55, 2.2, "convergente\nsubsónico", color=AZUL, ha="center", fontsize=11)
a1.text(1.5, 2.45, "garganta, $M = 1$", color=GRIS, ha="center", va="bottom", fontsize=11)
a1.text(1.9, 2.2, "divergente\nsupersónico", color=AZUL, ha="center", fontsize=11)
a1.text(2.6, 2.2, "divergente\nsubsónico", color=AZUL, ha="center", fontsize=11)
a1.text(xs + 0.04, -2.3, "choque\nnormal", color=ROJO, fontsize=11)
a1.set(ylabel="$A(x)/2$ (adimensional)", xlim=(0, 3), ylim=(-3.2, 3.6),
       title="(a) Tobera de Anderson, $A(x) = 1 + 2.2\\,(x - 1.5)^2$")
a1.grid(alpha=0.2)
a2.plot(x, M_isen, color=AZUL, ls="--", lw=2, label="expansión isentrópica completa, sin choque")
a2.plot(x[x < xs], M_choque[x < xs], color=ROJO, lw=2.4, label=f"con choque normal en $x_s = {xs}$  ($p_e/p_0 = 0.678$)")
a2.plot(x[x >= xs], M_choque[x >= xs], color=ROJO, lw=2.4)
a2.plot([xs, xs], [M_choque[x < xs][-1], M_choque[x >= xs][0]], color=ROJO, lw=2.4)
a2.axhline(1.0, color=GRIS, ls="--", lw=1.2)
a2.text(0.1, 1.06, "$M = 1$", color=GRIS, fontsize=11)
a2.set(xlabel="coordenada axial $x$ (adimensional)", ylabel="número de Mach $M$", xlim=(0, 3), ylim=(0, 2.6),
       title="(b) Número de Mach sobre el eje")
a2.legend(frameon=False, loc="upper left", fontsize=11)
a2.grid(alpha=0.2)
fig.tight_layout()
guardar(fig, "fig_nozzle_regimes_es.png")


# ------------------------------------------------ ondas del problema de Riemann
fig, ax = plt.subplots(figsize=(10, 5))
ax.annotate("", (1.05, 0), (-1.05, 0), arrowprops=dict(arrowstyle="-|>", color=TINTA, lw=1.5))
ax.annotate("", (0, 1.0), (0, 0), arrowprops=dict(arrowstyle="-|>", color=TINTA, lw=1.5))
ax.text(1.02, -0.07, "$x$", fontsize=14); ax.text(0.03, 0.97, "$t$", fontsize=14)
ax.plot([0, -0.62], [0, 0.85], color=AZUL, lw=2.6)
ax.plot([0, 0.12], [0, 0.85], color=VERDE, lw=2.2, ls="--")
ax.plot([0, 0.75], [0, 0.85], color=ROJO, lw=2.6)
ax.text(-0.68, 0.9, "$S_L$", color=AZUL, fontsize=14)
ax.text(0.13, 0.9, "$S_*$", color=VERDE, fontsize=14)
ax.text(0.77, 0.9, "$S_R$", color=ROJO, fontsize=14)
caja = dict(boxstyle="round,pad=0.35", facecolor="#FFFBE6", edgecolor=TINTA)
ax.text(-0.85, 0.45, r"$\mathbf{U}_L$", fontsize=15, ha="center", bbox=caja)
ax.text(-0.2, 0.6, r"$\mathbf{U}_L^{*}$", fontsize=14, ha="center", bbox=dict(caja, facecolor="#E3EEFA"))
ax.text(0.33, 0.6, r"$\mathbf{U}_R^{*}$", fontsize=14, ha="center", bbox=caja)
ax.text(0.85, 0.45, r"$\mathbf{U}_R$", fontsize=15, ha="center", bbox=caja)
ax.text(-0.58, 0.22, "onda izquierda\n(choque o rarefacción)", color=AZUL, ha="center", fontsize=11, style="italic")
ax.text(0.34, 0.76, "discontinuidad\nde contacto", color=VERDE, ha="center", fontsize=11, style="italic")
ax.text(0.72, 0.33, "onda derecha\n(choque o rarefacción)", color=ROJO, ha="center", fontsize=11, style="italic")
ax.text(-0.75, -0.1, r"estado izquierdo $\mathbf{U}_L$", color=GRIS, fontsize=11.5, ha="center")
ax.text(0.7, -0.1, r"estado derecho $\mathbf{U}_R$", color=GRIS, fontsize=11.5, ha="center")
ax.set(xlim=(-1.1, 1.1), ylim=(-0.18, 1.05)); ax.axis("off")
fig.tight_layout()
guardar(fig, "fig_riemann_waves_es.png")
