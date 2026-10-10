#!/usr/bin/env python3
"""Figura de validacion del caso cdNozzle2D para el Capitulo 3.

(a) Mach sobre el eje: solucion exacta cuasi-1D con choque normal frente
    a las mallas 150x40 y 300x80 (p_e = 67 875 Pa).
(b) Posicion del choque en los siete regimenes de contrapresion: prediccion
    exacta frente al CFD (promedio de las ultimas seis escrituras).

Uso (desde openfoam/):
    python3 herramientas/figura_capitulo3.py <carpeta_de_salida>
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# regimen.py ejecuta su interfaz de linea de comandos al importarse; se
# ejecuta solo el bloque de funciones, anterior a esa interfaz.
fuente = (Path(__file__).resolve().parent / "regimen.py").read_text()
ns = {}
exec(fuente.split("if sys.argv[1] == 'prediccion'")[0], ns)
area, mach_de_area, pe_para_choque_en = ns["area"], ns["mach_de_area"], ns["pe_para_choque_en"]

RAIZ = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "figuras"
P0 = 101325.0


def choque_teorico(pe):
    return brentq(lambda xs: pe_para_choque_en(xs)[0] - pe / P0, 1.5001, 2.999, xtol=1e-10)


def perfil_teorico(pe, x):
    xs = choque_teorico(pe)
    p0r = pe_para_choque_en(xs)[4]
    M = np.empty_like(x)
    for i, xi in enumerate(x):
        if xi < 1.5:
            M[i] = mach_de_area(area(xi), False)
        elif xi < xs:
            M[i] = mach_de_area(area(xi), True)
        else:
            M[i] = mach_de_area(area(xi) * p0r, False)
    return M, xs


def leer(caso):
    filas = list(csv.DictReader(open(RAIZ / "resultados" / "perfiles" / f"{caso}_perfil_axial.csv")))
    return np.array([float(f["x"]) for f in filas]), np.array([float(f["Ma"]) for f in filas])


plt.rcParams.update({"font.family": "serif", "font.size": 12, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))

# (a) perfil axial
xt = np.linspace(0.0, 3.0, 1201)
Mt, xs = perfil_teorico(67875.0, xt)
a1.plot(xt, Mt, "k-", lw=2, label="Solución exacta cuasi-1D")
for caso, estilo, etiqueta in [("geo20p", "C0--", r"CFD $150\times40$"),
                               ("m300", "C3-", r"CFD $300\times80$")]:
    x, M = leer(caso)
    a1.plot(x, M, estilo, lw=1.4, label=etiqueta)
a1.axvline(xs, color="gray", ls=":", lw=1)
a1.text(xs + 0.03, 0.15, f"$x_s = {xs:.3f}$ m", fontsize=10, color="gray")
a1.set(xlabel="$x$ [m]", ylabel="Número de Mach sobre el eje",
       title=r"(a) Perfil axial, $p_e = 67\,875$ Pa")
a1.legend(fontsize=10, loc="upper left")

# (b) mapa de regimenes
filas = [f for f in csv.DictReader(open(RAIZ / "resultados" / "tabla_casos.csv"))
         if f["escala"] == "20" and f["nx"] == "150" and f["endTime"] == "0.15"]
pe = np.array([float(f["pe_Pa"]) for f in filas])
x_cfd = np.array([float(f["x_choque_m"]) for f in filas])
sd = np.array([float(f["desv_m"]) for f in filas])
x_th = np.array([choque_teorico(p) for p in pe])
orden = np.argsort(pe)
lim = [1.7, 2.58]
a2.plot(lim, lim, "k-", lw=1, label="CFD = exacta")
a2.errorbar(x_th[orden], x_cfd[orden], yerr=sd[orden], fmt="o", color="C0",
            ms=5, capsize=3, label=r"CFD $150\times40$")
for xt_, xc_, p in zip(x_th, x_cfd, pe):
    a2.annotate(f"{p/1000:.0f} kPa", (xt_, xc_), textcoords="offset points",
                xytext=(6, -10), fontsize=9)
a2.set(xlim=lim, ylim=lim, xlabel="Posición exacta del choque [m]",
       ylabel="Posición del choque en el CFD [m]", title="(b) Siete contrapresiones")
a2.set_aspect("equal")
a2.legend(fontsize=10, loc="upper left")

fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_openfoam_validacion_es.pdf")
print("escrito", OUT / "fig_openfoam_validacion_es.pdf", f"(x_s exacto = {xs:.4f})")
