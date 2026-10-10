#!/usr/bin/env python3
"""Regenera las figuras del capítulo de Resultados, en español.

Uso (desde qae_cfd/):
    python3 scripts/generar_figuras_resultados.py

Salidas en ../Imagenes/:
  fig_oracle_calls_scaling_es.pdf : escalado teórico de la cota de Brassard et al.
  fig_error_vs_calls_es.pdf       : error medido del estimador y fracción de
                                    violaciones de la cota, por disparos
  fig_nozzle_validation_es.pdf    : la tobera con el solucionador híbrido
  fig_nozzle_comparacion_es.pdf   : la misma tobera con la línea base clásica
  fig_nozzle_dispersion_es.pdf    : dispersión entre semillas celda a celda
  fig_presupuesto_costo_es.pdf    : compuertas por ejecución y error contra costo

Datos:
  datos/barrido_estimador_circuito.json : error del estimador midiendo el circuito,
      m = 3..10 y 1, 5 y 25 disparos (scripts/barrido_estimador_circuito.py).
  datos/validacion_qa.json : la tobera (400 pasos, 31 celdas) con la media estimada
      por el circuito medido, m = 5, cinco semillas de un disparo y una de 25
      disparos, más la línea base clásica (scripts/generar_datos_validacion.py).
  datos/compuertas_vs_M.json : compuertas por ejecución para m = 5 y M = 1..16,
      contadas sobre la cinta; se genera aquí si no existe.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
OUT = RAIZ.parent / "Imagenes"
DATOS = RAIZ / "datos"
BARRIDO = json.loads((DATOS / "barrido_estimador_circuito.json").read_text())
VALIDACION = json.loads((DATOS / "validacion_qa.json").read_text())

plt.rcParams.update({"font.family": "serif", "font.size": 12, "figure.dpi": 150,
                     "savefig.dpi": 300, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3})

BHMT = "Cota de Brassard, Høyer, Mosca y Tapp"
XN = r"Llamadas al oráculo por ejecución, $N = 2^m$"
P_FALLO = 1 - 8 / np.pi**2
DISPAROS = [int(s) for s in BARRIDO["disparos"]]
ESTILO = {1: ("C3", "o", "1 disparo (una medición)"),
          5: ("C1", "s", "5 disparos (mediana)"),
          25: ("C0", "^", "25 disparos (mediana)")}

# ═══ 1. Escalado teórico ═══
N = np.logspace(0.5, 4, 200)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.loglog(N, np.pi / N + np.pi**2 / N**2, "k--", lw=2,
          label=BHMT + r" ($a = 1/2$): $\varepsilon \leq \pi/N + \pi^2/N^2$")
ax.loglog(N, 1 / np.sqrt(N), color="gray", ls=":", lw=2,
          label=r"Monte Carlo: $\varepsilon \sim 1/\sqrt{N}$")
for m in [3, 5, 8, 10]:
    ax.axvline(2**m, color="C0", alpha=0.25)
    ax.text(2**m, 1.6, f"$m={m}$", ha="center", fontsize=10, color="C0")
ax.set(xlabel=XN, ylabel=r"Error de estimación $\varepsilon$",
       title="Escalado teórico del error con el número de aplicaciones de $Q$")
ax.legend(fontsize=10)
fig.savefig(OUT / "fig_oracle_calls_scaling_es.pdf"); plt.close(fig)

# ═══ 2. Error medido del estimador y violaciones de la cota ═══
ms = sorted(int(m) for m in BARRIDO["por_m"])
Ns = np.array([2**m for m in ms])
por_m = BARRIDO["por_m"]
cota_media = np.array([por_m[str(m)]["cota_media"] for m in ms])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for s in DISPAROS:
    color, marca, etiq = ESTILO[s]
    errs = [por_m[str(m)]["por_disparos"][str(s)]["err_medio"] for m in ms]
    viol = [por_m[str(m)]["por_disparos"][str(s)]["fraccion_violaciones"] for m in ms]
    ax1.loglog(Ns, errs, color=color, marker=marca, ms=6, label=etiq)
    ax2.semilogx(Ns, viol, color=color, marker=marca, ms=6, label=etiq)
ax1.loglog(Ns, cota_media, "k--", label=BHMT + " (media sobre los 200 vectores)")
ax1.set(xlabel=XN, ylabel="Error absoluto medio de la media estimada",
        title=f"(a) Error medido, {BARRIDO['instancias']} vectores de $M = {BARRIDO['nodos']}$ por punto")
ax2.axhline(P_FALLO, color="k", ls="--",
            label=r"Máximo que admite el teorema: $1 - 8/\pi^2 \approx 0.19$")
ax2.set(xlabel=XN, ylabel="Fracción de estimaciones fuera de la cota",
        title="(b) Estimaciones fuera de la cota, medidas", ylim=(-0.01, 0.25))
manejas, etiquetas = ax1.get_legend_handles_labels()
h2, e2 = ax2.get_legend_handles_labels()
fig.legend(manejas + [h2[-1]], etiquetas + [e2[-1]], loc="lower center", ncol=3,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(OUT / "fig_error_vs_calls_es.pdf"); plt.close(fig)

# ═══ 3 y 4. La tobera: validación del híbrido y comparación con la línea base ═══
clas = VALIDACION["clasico_exacto"]
semillas = sorted(k for k in VALIDACION if k.startswith("circuito_s1_"))
hyb = VALIDACION[semillas[0]]              # perfiles: la primera semilla de un disparo
hyb_s25 = VALIDACION.get("circuito_s25_semilla1")
x = np.array(hyb["x"])
VARIABLES = [("M", "$M$"), ("rho", r"$\rho$"), ("p", "$p$")]
EXACTA = "Solución exacta cuasi-1D"
HIBRIDO = f"Solucionador híbrido (QAEA, $m = {hyb['m']}$, 1 disparo, semilla {hyb['seed']})"
BASE = "Línea base clásica (suma determinista)"


def error_medio(d, campo):
    """Error relativo medio sobre las 31 celdas, |q - q_exacta|/|q_exacta| x 100."""
    q, e = np.array(d[campo]), np.array(d[f"{campo}_exact"])
    return float(np.mean(np.abs(q - e) / np.abs(e)) * 100)


def figura_tobera(ruta, con_base):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    campos = [("M", axes[0][0], "Número de Mach", "(a) Número de Mach"),
              ("rho", axes[0][1], r"Densidad $\rho/\rho_0$", "(b) Densidad"),
              ("p", axes[1][0], r"Presión $p/p_0$", "(c) Presión")]
    for campo, ax, ylab, titulo in campos:
        ax.plot(x, hyb[f"{campo}_exact"], "k-", lw=2, label=EXACTA)
        if con_base:
            ax.plot(x, clas[campo], "-", color="C0", lw=1.6, label=BASE)
            ax.plot(x, hyb[campo], "s", ms=5, mfc="none", mew=1.3, color="C1",
                    label=HIBRIDO)
        else:
            ax.plot(x, hyb[campo], "o", ms=4, alpha=0.85, color="C1", label=HIBRIDO)
        ax.axvline(2.1, color="r", ls="--", alpha=0.5, label="Choque ($x = 2.1$)")
        ax.set(xlabel="$x$", ylabel=ylab, title=titulo)
    # (d): error relativo medio por variable. Cada semilla de un disparo es un
    # punto pequeño; el cuadrado es su media y la barra su desviación típica.
    ax = axes[1][1]
    xpos = np.arange(len(VARIABLES))
    errs_semillas = np.array([[error_medio(VALIDACION[k], c) for c, _ in VARIABLES]
                              for k in semillas])
    for j, fila in enumerate(errs_semillas):
        ax.plot(xpos + 0.08, fila, "x", color="C1", ms=6, alpha=0.6,
                label=f"Híbrido, 1 disparo, cada semilla ({len(semillas)})" if j == 0 else None)
    ax.errorbar(xpos + 0.08, errs_semillas.mean(axis=0), yerr=errs_semillas.std(axis=0, ddof=1),
                fmt="s", ms=8, capsize=5, lw=1.5, color="C1",
                label="Híbrido, 1 disparo: media y desviación entre semillas")
    if hyb_s25 is not None:
        ax.plot(xpos + 0.22, [error_medio(hyb_s25, c) for c, _ in VARIABLES], "^",
                ms=8, color="C0", mfc="none", mew=1.5, label="Híbrido, mediana de 25 disparos")
    if con_base:
        ax.plot(xpos - 0.1, [error_medio(clas, c) for c, _ in VARIABLES], "o", ms=8,
                color="C0", label=BASE)
    ax.set_xticks(xpos)
    ax.set_xticklabels([n for _, n in VARIABLES], fontsize=12)
    ax.set(title="(d) Error relativo medio [%]", ylim=(0, 3.0))
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis="x")
    manejas, etiquetas = axes[0][0].get_legend_handles_labels()
    fig.legend(manejas, etiquetas, loc="lower center", ncol=2 if con_base else 3,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07 if con_base else 0.045, 1, 1))
    fig.savefig(OUT / ruta)
    plt.close(fig)


figura_tobera("fig_nozzle_validation_es.pdf", con_base=False)
figura_tobera("fig_nozzle_comparacion_es.pdf", con_base=True)

# ═══ 5. Dispersión entre semillas celda a celda ═══
# Tres distancias por celda, en el número de Mach: la de la línea base a la
# solución exacta (error de discretización), la de cada semilla a la línea base
# (efecto del estimador) y el rango entre semillas (ruido de la medición).
M_ex = np.array(clas["M_exact"]); M_cl = np.array(clas["M"])
M_sem = np.array([VALIDACION[k]["M"] for k in semillas])
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.semilogy(x, np.abs(M_cl - M_ex), "k-", lw=2, label="Línea base frente a solución exacta (discretización)")
for j, k in enumerate(semillas):
    ax.semilogy(x, np.abs(M_sem[j] - M_cl) + 1e-12, "-", color="C1", alpha=0.5, lw=1,
                label="Cada semilla de 1 disparo frente a la línea base" if j == 0 else None)
ax.semilogy(x, M_sem.max(axis=0) - M_sem.min(axis=0) + 1e-12, "o-", color="C3", ms=4, lw=1.2,
            label=f"Rango entre las {len(semillas)} semillas")
if hyb_s25 is not None:
    ax.semilogy(x, np.abs(np.array(hyb_s25["M"]) - M_cl) + 1e-12, "^-", color="C0", ms=4, lw=1.2,
                mfc="none", label="Mediana de 25 disparos frente a la línea base")
ax.axvline(2.1, color="r", ls="--", alpha=0.5)
ax.set(xlabel="$x$", ylabel="Diferencia en el número de Mach", ylim=(1e-8, 1))
manejas, etiquetas = ax.get_legend_handles_labels()
fig.legend(manejas, etiquetas, loc="lower center", ncol=2, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(OUT / "fig_nozzle_dispersion_es.pdf"); plt.close(fig)

# ═══ 6. Presupuesto de costo ═══
# (a) compuertas por ejecución del circuito, contadas sobre la cinta: frente a m
#     con M = 4 (del barrido) y frente a M con m = 5 (calculadas aquí una vez).
# (b) error medido frente al costo total en aplicaciones de Q (disparos × 2^m).
ruta_M = DATOS / "compuertas_vs_M.json"
if not ruta_M.exists():
    from qae_cfd.qae_grover import GroverMeanQAE
    q = GroverMeanQAE(m_precision=5, shots=1, seed=0)
    ruta_M.write_text(json.dumps({str(M): q.gate_count(M) for M in (1, 2, 4, 8, 16)}, indent=1))
VS_M = json.loads(ruta_M.read_text())
comp_m = {m: por_m[str(m)]["compuertas_por_ejecucion"] for m in ms}
fig, (ax1, axM, ax2) = plt.subplots(1, 3, figsize=(15, 4.8))
ax1.loglog(Ns, [comp_m[m]["logical_qft_expanded"] for m in ms], "ko-", ms=5,
           label="Compuertas lógicas")
ax1.loglog(Ns, [comp_m[m]["two_qubit"] for m in ms], "ks--", ms=5, mfc="none",
           label="Compuertas de dos cúbits tras descomponer")
ax1.set(xlabel=XN, ylabel="Compuertas por ejecución del circuito",
        title="(a) Compuertas frente a $m$, con $M = 4$")
ax1.legend(fontsize=9)
Ms = sorted(int(k) for k in VS_M)
axM.loglog(Ms, [VS_M[str(M)]["logical_qft_expanded"] for M in Ms], "ko-", ms=5,
           label="Compuertas lógicas")
axM.loglog(Ms, [VS_M[str(M)]["two_qubit"] for M in Ms], "ks--", ms=5, mfc="none",
           label="Compuertas de dos cúbits tras descomponer")
axM.set(xlabel="Muestras por estimación, $M$", ylabel="Compuertas por ejecución del circuito",
        title="(b) Compuertas frente a $M$, con $m = 5$", xticks=Ms, xticklabels=[str(M) for M in Ms])
axM.minorticks_off()
axM.legend(fontsize=9)
for s in DISPAROS:
    color, marca, etiq = ESTILO[s]
    errs = [por_m[str(m)]["por_disparos"][str(s)]["err_medio"] for m in ms]
    ax2.loglog(Ns * s, errs, color=color, marker=marca, ms=6, label=etiq)
ax2.loglog(Ns, cota_media, "k--", label=BHMT + " (una ejecución)")
ax2.set(xlabel=r"Llamadas al oráculo por estimación, disparos $\times\, 2^m$",
        ylabel="Error absoluto medio de la media estimada",
        title="(c) Error medido frente al costo total")
# La leyenda del panel (c) va debajo de la figura: dentro de los ejes tapa
# el arranque de la serie de 25 disparos o el final de las de 1 y 5.
manejas, etiquetas = ax2.get_legend_handles_labels()
fig.legend(manejas, etiquetas, loc="lower center", ncol=4, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(OUT / "fig_presupuesto_costo_es.pdf"); plt.close(fig)

print("figuras regeneradas (es):", sorted(p.name for p in OUT.glob("fig_*_es.pdf")))
