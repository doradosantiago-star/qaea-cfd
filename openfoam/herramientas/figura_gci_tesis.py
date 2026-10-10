#!/usr/bin/env python3
"""Figura de independencia de malla (GCI) para el trabajo de grado, en español
y con la notación del documento (M_1 en lugar de Ma_1).

Los números son los de la tabla 3.5 del documento, producidos por
herramientas/gci.py sobre las corridas de OpenFOAM (promedio de las 21
escrituras entre t = 0.05 y 0.15 s, medida continua del choque por cruce de
presión). Este guion solo los dibuja: no recalcula nada.

Salida: figuras/fig_independencia_malla_es.pdf
Ejecución: python3 herramientas/figura_gci_tesis.py   (desde openfoam/)
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "figuras"

DX = np.array([0.04, 0.02, 0.01])            # 75x20, 150x40, 300x80
MALLAS = ["$75\\times20$", "$150\\times40$", "$300\\times80$"]
DATOS = {
    # nombre, media por malla, desviación, p, extrapolado, GCI fina (%), analítico
    "xs": dict(etq="Posición del choque $x_s$ [m]", f=[2.0835, 2.0976, 2.1027],
               s=[0.0150, 0.0066, 0.0028], p=1.48, ext=2.1055, gci=0.17, ana=2.1098),
    "M1": dict(etq="Número de Mach previo al choque $M_1$", f=[1.9340, 2.0126, 2.0394],
               s=[0.0284, 0.0115, 0.0132], p=1.55, ext=2.0533, gci=0.85, ana=2.0880),
}

plt.rcParams.update({"font.family": "serif", "font.size": 12, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, (clave, d), panel in zip(axes, DATOS.items(), "ab"):
    f, s = np.array(d["f"]), np.array(d["s"])
    # curva f(dx) = ext + C dx^p, con C ajustado a la malla fina
    C = (f[2] - d["ext"]) / DX[2] ** d["p"]
    xx = np.linspace(0, 0.045, 200)
    ax.plot(xx, d["ext"] + C * xx ** d["p"], color="C0", alpha=0.5, lw=1.5,
            label=r"$f(\Delta x) = f_{\mathrm{ext}} + C\,\Delta x^{p}$")
    ax.errorbar(DX, f, yerr=s, fmt="o", color="C0", ms=7, capsize=4, lw=1.5,
                label="Media temporal de cada malla, $\\pm 1$ desviación")
    for x, y, m in zip(DX, f, MALLAS):
        ax.annotate(m, (x, y), xytext=(7, -4), textcoords="offset points", fontsize=9, color="gray")
    ax.plot(0, d["ext"], "o", mfc="white", color="C0", ms=7, label="Extrapolado a $\\Delta x \\to 0$")
    banda = d["gci"] / 100 * f[2]
    ax.errorbar([DX[2] - 0.0025], [f[2]], yerr=[banda], fmt="none", ecolor="k", capsize=5, lw=1.5,
                label="Banda del índice de convergencia de la malla fina")
    ax.axhline(d["ana"], color="k", ls="--", lw=1.2, label="Solución analítica cuasi-unidimensional")
    ax.set(xlabel="Tamaño de celda $\\Delta x$ [m]", ylabel=d["etq"],
           title=f"({panel}) {d['etq'].split(' [')[0].split(' $')[0]}, orden aparente $p = {d['p']}$",
           xlim=(-0.003, 0.047))
manejas, etiquetas = axes[0].get_legend_handles_labels()
fig.legend(manejas, etiquetas, loc="lower center", ncol=3, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(OUT / "fig_independencia_malla_es.pdf")
print("guardado", OUT / "fig_independencia_malla_es.pdf")
