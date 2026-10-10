#!/usr/bin/env python3
"""Dibuja la figura del punto de entrada del QAEA en el ciclo de volúmenes
finitos (Capítulo 4), versión en español.

Uso (desde qae_cfd/):
    python3 scripts/generar_figura_punto_entrada.py

Salida: ../Imagenes/fig_qae_entry_point_es.pdf (y .png de respaldo).
El lienzo es de 10.5 x 4.9 pulgadas con letra de 13 pt, de modo que a ancho
completo de página (17.6 cm) la letra impresa queda en unos 8.5 pt.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "Imagenes" / "fig_qae_entry_point_es"

AZUL, VERDE, NARANJA = "#2E4057", "#1B5E20", "#CC6600"
plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 12.5, "mathtext.fontset": "stix"})
fig, ax = plt.subplots(figsize=(10.5, 4.9))
ax.set_xlim(0, 15)
ax.set_ylim(-0.75, 6.75)
ax.axis("off")


def caja(x, y, w, h, texto, borde, fondo, negrita=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                ec=borde, fc=fondo, lw=2))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center",
            fontweight="bold" if negrita else "normal", linespacing=1.5)


def flecha(x0, y0, x1, y1, color):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=13))


# Fila superior: ciclo clásico
ax.text(8.2, 6.25, "Ciclo temporal clásico (FVM)", ha="center", fontsize=15,
        fontweight="bold", color=AZUL)
arriba = ["Estado\n$\\mathbf{Q}^n$", "Reconstrucción\nMUSCL + minmod",
          "Flujos de Riemann\n$\\widehat{\\mathbf{F}}$ (Roe / HLLC)",
          "Lado derecho\n$\\mathbf{f}(\\mathbf{Q})$", "Estado\n$\\mathbf{Q}^{n+1}$"]
xs = [0.2, 3.12, 6.04, 8.96, 11.88]
for x, t in zip(xs, arriba):
    caja(x, 4.7, 2.65, 1.2, t, AZUL, "#EEF1F6")
for a, b in zip(xs[:-1], xs[1:]):
    flecha(a + 2.65, 5.3, b, 5.3, AZUL)

# Caja central: media temporal
caja(9.6, 2.7, 3.4, 1.6,
     "Media temporal\n$\\int_0^{\\Delta t} f\\,d\\tau \\approx \\Delta t\\cdot\\overline{f}$\n(entrada del QAEA)",
     NARANJA, "#FDD9B5", negrita=True)
flecha(10.25, 4.7, 10.7, 4.3, NARANJA)
flecha(12.6, 4.3, 13.15, 4.7, NARANJA)

# Fila inferior: núcleo cuántico
ax.text(7.5, -0.45, "Estimador de amplitud (PennyLane, lightning.qubit)", ha="center",
        fontsize=15, fontweight="bold", color=VERDE)
abajo = ["Operador $A$: carga\nlas $M$ muestras\n$f_j = f(\\ell(\\tau_j))$",
         "Operador de Grover\n$Q = -A S_0 A^{\\dagger} S_\\chi$\n(rotación $2\\theta_a$)",
         "Estimación de fase:\n$m$ cúbits, $N = 2^m$\nllamadas al oráculo",
         "QFT inversa +\nmedición $\\rightarrow k$",
         "Media estimada\n$\\tilde{f}$, de $\\sin^2(\\pi k/2^m)$"]
xs = [0.2, 3.12, 6.04, 8.96, 11.88]
for x, t in zip(xs, abajo):
    caja(x, 0.2, 2.65, 1.6, t, VERDE, "#E8F5E9")
for a, b in zip(xs[:-1], xs[1:]):
    flecha(a + 2.65, 1.0, b, 1.0, VERDE)
flecha(10.4, 2.7, 1.55, 1.85, NARANJA)
flecha(13.2, 1.8, 12.3, 2.7, NARANJA)

fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
print("escrito", OUT.with_suffix(".pdf"))
