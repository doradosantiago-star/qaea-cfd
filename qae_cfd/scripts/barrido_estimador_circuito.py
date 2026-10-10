#!/usr/bin/env python3
"""Barrido del estimador de media ejecutando y midiendo el circuito (Capítulo 5).

Para cada m en {3, ..., 10} y cada número de disparos en {1, 5, 25}, estima la
media de 200 vectores aleatorios de M = 4 valores uniformes en [0, 1] con el
circuito completo (operador A, operador de Grover Q, estimación de fase y
medición del registro) sobre lightning.qubit, y compara cada estimación con la
media aritmética del vector, que aquí solo sirve de referencia para medir el
error. Con 1 disparo la estimación es la lectura de una medición; con 5 y 25,
la mediana de las lecturas.

Guarda por (m, disparos): los 200 errores individuales, su media, el error
cuadrático medio, la cota de Brassard, Høyer, Mosca y Tapp
2π√(a(1−a))/2^m + π²/4^m evaluada en la media verdadera de cada vector, y la
fracción de estimaciones que la violan. La cota es probabilística: una medición
la cumple con probabilidad al menos 8/π² ≈ 0.81, así que la fracción de
violaciones medida debe quedar por debajo de 1 − 8/π² ≈ 0.19 y no en cero.

Semillas: los vectores salen de default_rng(42) (los mismos para todo m y
todo número de disparos); la medición usa la semilla 1000·m + disparos.

Ejecución (desde qae_cfd/):
    python3 scripts/barrido_estimador_circuito.py --m 3 4 5 6 7 8   # un bloque
    python3 scripts/barrido_estimador_circuito.py --m 9 --disparos 1 5
    python3 scripts/barrido_estimador_circuito.py --m 10 --disparos 1
    python3 scripts/barrido_estimador_circuito.py --merge           # une los bloques
Cada m escribe datos/barrido/m{m}.json; --merge produce
datos/barrido_estimador_circuito.json. Las tres series llegan hasta m = 10;
con varios disparos el simulador tarda mucho más por estimación en m grandes.
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

CARPETA = RAIZ / "datos" / "barrido"
SALIDA = RAIZ / "datos" / "barrido_estimador_circuito.json"
INSTANCIAS = 200
NODOS = 4
DISPAROS = (1, 5, 25)
SEMILLA_VECTORES = 42


def cota_bhmt(a: np.ndarray, m: int) -> np.ndarray:
    return 2 * np.pi * np.sqrt(a * (1 - a)) / 2**m + np.pi**2 / 4**m


def correr_m(m: int, disparos=DISPAROS) -> dict:
    rng = np.random.default_rng(SEMILLA_VECTORES)
    muestras = rng.uniform(0, 1, (INSTANCIAS, NODOS))
    medias = muestras.mean(axis=1)          # referencia para medir el error
    cotas = cota_bhmt(medias, m)
    resultado = {"m": m, "N": 2**m, "instancias": INSTANCIAS, "nodos": NODOS,
                 "cota_media": float(cotas.mean()), "por_disparos": {}}
    for shots in disparos:
        ledger = CostLedger()
        q = GroverMeanQAE(m_precision=m, shots=shots, seed=1000 * m + shots,
                          ledger=ledger)
        t0 = time.time()
        estimaciones = np.array([q.estimate_mean(g) for g in muestras])
        dt = time.time() - t0
        errores = np.abs(estimaciones - medias)
        resultado["por_disparos"][str(shots)] = {
            "errores": errores.tolist(),
            "err_medio": float(errores.mean()),
            "err_rms": float(np.sqrt(np.mean(errores**2))),
            "err_max": float(errores.max()),
            "violaciones": int((errores > cotas).sum()),
            "fraccion_violaciones": float((errores > cotas).mean()),
            "segundos_por_estimacion": dt / INSTANCIAS,
            "costo": ledger.to_dict(),
            "device": q._device_name.get(NODOS),
        }
        d = resultado["por_disparos"][str(shots)]
        print(f"[m={m} disparos={shots}] err_medio={d['err_medio']:.2e}  "
              f"viol={d['fraccion_violaciones']:.3f}  "
              f"({d['segundos_por_estimacion']:.3f} s/estimación)", flush=True)
    resultado["compuertas_por_ejecucion"] = q.gate_count(NODOS)
    return resultado


def merge() -> None:
    por_m = {}
    for f in sorted(CARPETA.glob("m*.json"), key=lambda p: int(p.stem[1:])):
        d = json.loads(f.read_text())
        por_m[str(d["m"])] = d
    datos = {"instancias": INSTANCIAS, "nodos": NODOS, "disparos": list(DISPAROS),
             "semilla_vectores": SEMILLA_VECTORES,
             "lectura": "qml.sample del registro de precisión; 1 disparo = una "
                        "medición, >1 = mediana de las lecturas",
             "por_m": por_m}
    SALIDA.write_text(json.dumps(datos, indent=1))
    print(f"{SALIDA.name}: m = {sorted(int(k) for k in por_m)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="*", default=[])
    ap.add_argument("--disparos", type=int, nargs="*", default=list(DISPAROS),
                    help="números de disparos a correr para esos m (default 1 5 25)")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    CARPETA.mkdir(parents=True, exist_ok=True)
    for m in args.m:
        salida = CARPETA / f"m{m}.json"
        if salida.exists():
            print(f"[m={m}] ya corrido; se omite.", flush=True)
            continue
        salida.write_text(json.dumps(correr_m(m, tuple(args.disparos)), indent=1))
    if args.merge:
        merge()


if __name__ == "__main__":
    main()
