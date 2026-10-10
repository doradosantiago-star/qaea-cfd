#!/usr/bin/env python3
"""Dibuja la discretización espacial del método de volúmenes finitos
(Capítulo 2): celdas contiguas, estados en los centros, flujos en las
interfaces y ancho de celda.

Uso (desde qae_cfd/):
    python3 scripts/generar_figura_discretizacion.py

Salida: ../Imagenes/fig_fvm_discretization_es.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parents[2] / "Imagenes" / "fig_fvm_discretization_es.png"

AZUL, ROJO, GRIS = "#1F3A93", "#C0392B", "#7F8C8D"
plt.rcParams.update({"mathtext.fontset": "cm", "font.size": 15})
fig, ax = plt.subplots(figsize=(12, 3.8))
ax.set_xlim(-0.9, 5.9)
ax.set_ylim(-1.45, 1.55)
ax.axis("off")

ALTO, BASE = 1.0, -0.5          # altura y base de las celdas
# Celdas completas i-1, i, i+1 (contiguas) y celdas cortadas en los extremos
celdas = [(1, r"i-1"), (2, r"i"), (3, r"i+1")]
for x0, _ in celdas:
    ax.add_patch(Rectangle((x0, BASE), 1, ALTO, fc="#DCE6F7" if x0 == 2 else "#F2F5FB",
                           ec=AZUL, lw=2))
for x0, x1 in [(0.35, 1), (4, 4.65)]:
    ax.add_patch(Rectangle((x0, BASE), x1 - x0, ALTO, fc="#F7F7F7", ec=GRIS, lw=1.2, ls="--"))
ax.text(0.05, 0.0, r"$\cdots$", ha="center", va="center", color=GRIS, fontsize=20)
ax.text(4.95, 0.0, r"$\cdots$", ha="center", va="center", color=GRIS, fontsize=20)

# Estados en los centros y posiciones de los centros
for x0, idx in celdas:
    xc = x0 + 0.5
    ax.plot(xc, 0.12, "o", color="black", ms=7)
    ax.text(xc, -0.2, rf"$\mathbf{{U}}_{{{idx}}}$", ha="center", va="center", color=AZUL, fontsize=17)
    ax.plot([xc, xc], [BASE - 0.05, BASE - 0.15], color="black", lw=1.2)
    ax.text(xc, BASE - 0.32, rf"$x_{{{idx}}}$", ha="center", va="center", fontsize=15)

# Interfaces de la celda i y flujos numéricos que las cruzan
for xf, sub in [(2, r"i-\frac{1}{2}"), (3, r"i+\frac{1}{2}")]:
    ax.plot([xf, xf], [BASE, BASE + ALTO], color=ROJO, lw=3.2)
    ax.text(xf, BASE - 0.32, rf"$x_{{{sub}}}$", ha="center", va="center", color=ROJO, fontsize=15)
    ax.add_patch(FancyArrowPatch((xf - 0.32, 0.82), (xf + 0.32, 0.82), arrowstyle="-|>",
                                 mutation_scale=18, color=ROJO, lw=2.2))
    ax.text(xf, 1.2, rf"$\widehat{{\mathbf{{F}}}}_{{{sub}}}$", ha="center", va="center",
            color=ROJO, fontsize=17)

# Ancho de la celda i
ax.add_patch(FancyArrowPatch((2, -1.1), (3, -1.1), arrowstyle="<|-|>", mutation_scale=14,
                             color="black", lw=1.4))
ax.text(2.5, -1.32, r"$\Delta x$", ha="center", va="center", fontsize=16)

fig.savefig(OUT, dpi=220, bbox_inches="tight")
print("escrito", OUT)
