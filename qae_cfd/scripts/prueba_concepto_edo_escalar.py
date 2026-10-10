#!/usr/bin/env python3
"""Prueba de concepto del estimador dentro de una marcha temporal escalar (Capítulo 5).

Resuelve la ecuación diferencial escalar dv/dt = -v², v(0) = 1, cuya solución
exacta es v(t) = 1/(1 + t), con el mismo paso de tiempo que el solucionador
híbrido usa en la tobera: predictor lineal ℓ(τ) = v + τ f(v), M = 4 nodos
τ_j = j Δt/(M − 1), y la media de f(ℓ(τ_j)) sobre los nodos. Esa media se
calcula de dos formas: con la suma determinista (línea base) y con el circuito
de estimación de amplitud, ejecutado y medido (``GroverMeanQAE``), con el
reescalado afín de cada paso a [0, 1].

El experimento separa las fuentes de error del paso de tiempo:
- error de trayectoria: suma determinista contra la solución exacta, que solo
  contiene el predictor lineal y la cuadratura de M nodos;
- error de estimación: circuito contra la suma determinista, que solo contiene
  la estimación de amplitud, amplificada por el rango f_max − f_min de cada paso;
- error total: circuito contra la solución exacta.
Las tres se miden en t = T para cada paso Δt, cada m y cada semilla de la
medición, con 1 disparo y con la mediana de 5. Un m creciente con Δt fijo
debe llevar el error total al piso del error de trayectoria; un Δt decreciente
con m fijo debe mostrar el orden del predictor y el punto en que la estimación
pasa a dominar.

Mapeo afín: por defecto las cotas son el mínimo y el máximo de las cuatro
muestras de cada paso (mapeo por vector), que deja la media mapeada en 0.5.
Con --cotas se usan las cotas fijas conocidas del integrando, f ∈ [-1, 0]
para v ∈ [0, 1] (mapeo con cotas fijas): la media mapeada recorre [0, 1] a lo
largo de la marcha y la resolución 2^-m del estimador sí interviene, al
precio de un rango fijo de 1 que multiplica el error de estimación.

Ejecución (desde qae_cfd/):
    python3 scripts/prueba_concepto_edo_escalar.py --dt 0.2      # un bloque por Δt
    python3 scripts/prueba_concepto_edo_escalar.py --merge       # une los bloques
    python3 scripts/prueba_concepto_edo_escalar.py --figura      # dibuja la figura
    python3 scripts/prueba_concepto_edo_escalar.py --cotas --dt 0.2 0.05 --merge --figura
Cada Δt escribe datos/edo_escalar/dt{Δt}.json; --merge produce
datos/edo_escalar.json; --figura produce Imagenes/fig_edo_escalar_es.pdf.
Con --cotas los archivos son datos/edo_escalar_cotas/, datos/edo_escalar_cotas.json
y Imagenes/fig_edo_escalar_cotas_es.pdf, con m hasta 10.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
logging.disable(logging.CRITICAL)

from qae_cfd.cost import CostLedger  # noqa: E402
from qae_cfd.qae_grover import GroverMeanQAE  # noqa: E402

CARPETA = RAIZ / "datos" / "edo_escalar"
SALIDA = RAIZ / "datos" / "edo_escalar.json"
FIGURA = RAIZ.parent / "Imagenes" / "fig_edo_escalar_es.pdf"
COTAS_FIJAS = (-1.0, 0.0)       # f(v) = -v² con v ∈ [0, 1]
COTAS = False                    # se activa con --cotas (ver main)
T_FINAL = 2.0
PASOS_DT = (0.5, 0.4, 0.2, 0.1, 0.05, 0.025)
NODOS = 4
MS = (3, 4, 5, 6, 7, 8)
DISPAROS = (1, 5)
SEMILLAS = (0, 1, 2, 3, 4)


def f(v: float) -> float:
    return -v * v


def exacta(t: float) -> float:
    return 1.0 / (1.0 + t)


def marchar(dt: float, media, nodos: int = NODOS) -> tuple[float, float]:
    """Marcha desde v(0) = 1 hasta T_FINAL con el paso del solucionador híbrido.

    ``media`` recibe el vector de ``nodos`` muestras f(ℓ(τ_j)) y devuelve su
    media. Retorna v(T) y el promedio sobre los pasos del rango f_max − f_min.
    """
    n_pasos = int(round(T_FINAL / dt))
    taus = np.linspace(0.0, 1.0, nodos) * dt
    v, rangos = 1.0, []
    for _ in range(n_pasos):
        fv = f(v)
        muestras = np.array([f(v + tau * fv) for tau in taus])
        rangos.append(float(muestras.max() - muestras.min()))
        v += dt * media(muestras)
    return float(v), float(np.mean(rangos))


def correr_dt(dt: float) -> dict:
    v_ex = exacta(T_FINAL)
    v_det, rango = marchar(dt, lambda s: float(np.mean(s)))
    res = {"dt": dt, "pasos": int(round(T_FINAL / dt)), "nodos": NODOS,
           "v_exacta": v_ex, "v_determinista": v_det,
           "err_trayectoria": abs(v_det - v_ex), "rango_medio": rango,
           "por_m": {}}
    res["cotas"] = list(COTAS_FIJAS) if COTAS else None
    ms = MS + (9, 10) if COTAS else MS
    for m in ms:
        res["por_m"][str(m)] = {}
        for shots in DISPAROS:
            finales, t0 = [], time.time()
            for semilla in SEMILLAS:
                ledger = CostLedger()
                q = GroverMeanQAE(m_precision=m, shots=shots,
                                  seed=100000 * m + 1000 * shots + semilla,
                                  ledger=ledger)
                if COTAS:
                    media_q = lambda s, q=q: q.estimate_mean_general(s, bounds=COTAS_FIJAS)
                else:
                    media_q = q.estimate_mean_general
                v_q, _ = marchar(dt, media_q)
                finales.append(v_q)
            finales = np.array(finales)
            e_est = np.abs(finales - v_det)
            e_tot = np.abs(finales - v_ex)
            res["por_m"][str(m)][str(shots)] = {
                "v_final": finales.tolist(),
                "err_estimacion": e_est.tolist(),
                "err_estimacion_medio": float(e_est.mean()),
                "err_total": e_tot.tolist(),
                "err_total_medio": float(e_tot.mean()),
                "err_total_desv": float(e_tot.std(ddof=1)),
                "costo_por_corrida": ledger.to_dict(),  # registro de la última semilla; igual en todas
                "segundos": time.time() - t0,
            }
            print(f"dt={dt} m={m} disparos={shots}: trayectoria "
                  f"{res['err_trayectoria']:.2e} estimacion {e_est.mean():.2e} "
                  f"total {e_tot.mean():.2e} ({time.time() - t0:.0f} s)", flush=True)
    return res


def trayectoria_determinista() -> dict:
    """Error de trayectoria (suma determinista contra la exacta) para cada Δt
    y para M = 2, 4, 8 y 16 nodos, con el orden observado entre pasos
    consecutivos. No ejecuta circuitos."""
    v_ex = exacta(T_FINAL)
    salida = {}
    for nodos in (2, 4, 8, 16):
        errores = []
        for dt in PASOS_DT:
            v, _ = marchar(dt, lambda s: float(np.mean(s)), nodos)
            errores.append(abs(v - v_ex))
        ordenes = [float(np.log(errores[i] / errores[i + 1])
                         / np.log(PASOS_DT[i] / PASOS_DT[i + 1]))
                   for i in range(len(PASOS_DT) - 1)]
        salida[str(nodos)] = {"dt": list(PASOS_DT), "err_trayectoria": errores,
                              "orden_observado": ordenes}
    return salida


def merge() -> dict:
    bloques = {}
    for dt in PASOS_DT:
        archivo = CARPETA / f"dt{dt}.json"
        if archivo.exists():
            bloques[str(dt)] = json.loads(archivo.read_text())
    salida = {"ecuacion": "dv/dt = -v^2, v(0) = 1, exacta 1/(1+t)",
              "T": T_FINAL, "nodos": NODOS, "ms": list(MS),
              "disparos": list(DISPAROS), "semillas": list(SEMILLAS),
              "trayectoria_por_M": trayectoria_determinista(),
              "por_dt": bloques}
    SALIDA.write_text(json.dumps(salida, indent=1))
    print("escrito", SALIDA, "con", len(bloques), "bloques")
    return salida


def figura() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datos = json.loads(SALIDA.read_text())
    por_dt = datos["por_dt"]
    dts = sorted(float(k) for k in por_dt)
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "savefig.dpi": 300, "savefig.bbox": "tight",
                         "axes.grid": True, "grid.alpha": 0.3})
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    colores = {dt: f"C{i}" for i, dt in enumerate(dts)}
    # (a) error total, de estimación y de trayectoria frente a N = 2^m, un
    # disparo, para cuatro pasos Δt <= 0.2 (Omar, 8-10-2026: 0.5 son dos
    # puntos en el intervalo y no se lee). Las etiquetas nombran el tipo de
    # error, no la pareja comparada, para que "circuito frente a la suma"
    # no se lea como "el circuito es mejor que la suma".
    for dt in (0.2, 0.1, 0.05, 0.025):
        d = por_dt[str(dt)]
        ms = sorted(int(k) for k in d["por_m"])
        N = [2 ** m for m in ms]
        tot = [d["por_m"][str(m)]["1"]["err_total_medio"] for m in ms]
        est = [d["por_m"][str(m)]["1"]["err_estimacion_medio"] for m in ms]
        a.plot(N, tot, "o-", color=colores[dt], lw=1.8,
               label=f"$\\Delta t = {dt}$: error total")
        a.plot(N, est, "^:", color=colores[dt], mfc="white", lw=1.5,
               label=f"$\\Delta t = {dt}$: error de estimación")
        a.axhline(d["err_trayectoria"], color=colores[dt], ls="--", lw=1.2,
                  label=f"$\\Delta t = {dt}$: error de trayectoria")
    a.set(xscale="log", yscale="log", xlabel="$N = 2^{m}$",
          ylabel="Error en $v(T)$", title="(a) Error frente a $N$, un disparo")
    a.set_xticks([2 ** m for m in MS]); a.set_xticklabels([str(2 ** m) for m in MS])
    # (b) error frente a Δt, por m, un disparo, con la línea determinista
    tray = [por_dt[str(dt)]["err_trayectoria"] for dt in dts]
    b.plot(dts, tray, "k--", lw=1.5, label="Error de trayectoria (suma determinista)")
    for i, m in enumerate(MS):
        tot = [por_dt[str(dt)]["por_m"][str(m)]["1"]["err_total_medio"] for dt in dts]
        b.plot(dts, tot, "o-", color=f"C{i}", label=f"$m = {m}$")
    tot5 = [por_dt[str(dt)]["por_m"]["5"]["5"]["err_total_medio"] for dt in dts]
    b.plot(dts, tot5, "s-", color="C2", mfc="white", label="$m = 5$, mediana de 5 disparos")
    pend = np.polyfit(np.log(dts), np.log(tray), 1)[0]
    b.set(xscale="log", yscale="log", xlabel="$\\Delta t$", ylabel="Error en $v(T)$",
          title=f"(b) Error frente a $\\Delta t$; pendiente determinista {pend:.2f}")
    b.legend(fontsize=8, loc="lower right")
    # La leyenda del panel (a) va debajo de la figura, una columna por Δt,
    # para no tapar las curvas de Δt = 0.05 ni las líneas de trayectoria.
    manejas, etiquetas = a.get_legend_handles_labels()
    fig.legend(manejas, etiquetas, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIGURA)
    print("guardado", FIGURA, "pendiente determinista", pend)


def figura_cotas() -> None:
    """Figura del mapeo con cotas fijas: error total, de estimación y de
    trayectoria frente a N para los Δt corridos, con 1 disparo y mediana de 5."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datos = json.loads(SALIDA.read_text())
    por_dt = datos["por_dt"]
    dts = sorted(float(k) for k in por_dt)
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "savefig.dpi": 300, "savefig.bbox": "tight",
                         "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(1, len(dts), figsize=(6 * len(dts), 5), squeeze=False)
    for ax, dt, panel in zip(axes[0], dts, "abcd"):
        d = por_dt[str(dt)]
        ms = sorted(int(k) for k in d["por_m"])
        N = [2 ** m for m in ms]
        for shots, estilo, etq in (("1", "-", "un disparo"), ("5", "--", "mediana de 5 disparos")):
            tot = [d["por_m"][str(m)][shots]["err_total_medio"] for m in ms]
            est = [d["por_m"][str(m)][shots]["err_estimacion_medio"] for m in ms]
            ax.plot(N, tot, "o" + estilo, color="C0", label=f"circuito frente a la exacta, {etq}")
            ax.plot(N, est, "^" + estilo, color="C1", mfc="white", label=f"circuito frente a la suma, {etq}")
        ax.axhline(d["err_trayectoria"], color="k", ls=":", lw=1.5, label="suma frente a la exacta")
        ax.plot(N, [2 / 2 ** m for m in ms], color="gray", lw=1, label=r"$2 \cdot 2^{-m}$, referencia $1/N$")
        ax.set(xscale="log", yscale="log", xlabel="$N = 2^{m}$", ylabel="Error en $v(T)$",
               title=f"({panel}) $\\Delta t = {dt}$, cotas fijas $[-1, 0]$")
        ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N])
        ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURA)
    print("guardado", FIGURA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dt", type=float, nargs="*", default=[])
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--figura", action="store_true")
    ap.add_argument("--cotas", action="store_true",
                    help="mapeo con cotas fijas [-1, 0] en lugar del mínimo y máximo de cada vector")
    args = ap.parse_args()
    global COTAS, CARPETA, SALIDA, FIGURA
    if args.cotas:
        COTAS = True
        CARPETA = RAIZ / "datos" / "edo_escalar_cotas"
        SALIDA = RAIZ / "datos" / "edo_escalar_cotas.json"
        FIGURA = RAIZ.parent / "Imagenes" / "fig_edo_escalar_cotas_es.pdf"
    CARPETA.mkdir(parents=True, exist_ok=True)
    for dt in args.dt:
        res = correr_dt(dt)
        (CARPETA / f"dt{dt}.json").write_text(json.dumps(res, indent=1))
    if args.merge:
        merge()
    if args.figura:
        figura_cotas() if COTAS else figura()


if __name__ == "__main__":
    main()
