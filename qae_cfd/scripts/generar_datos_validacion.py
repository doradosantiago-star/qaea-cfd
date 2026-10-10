#!/usr/bin/env python3
"""Regenera datos/validacion_qa.json: la tobera de Anderson (400 pasos,
31 celdas, CFL 0.4, p_e/p0 = 0.678) resuelta en la formulación integral.

Modos:
  clasico_exacto : media temporal por la suma determinista de los M = 4 nodos,
                   la línea base clásica del trabajo.
  circuito       : la media temporal de cada celda y componente la estima el
                   circuito de PennyLane (m = 5, M = 4), ejecutado y medido con
                   el número de disparos indicado. Con 1 disparo la estimación
                   es la lectura de una medición; con más, la mediana.

Cada corrida escribe su propio archivo en datos/validacion_qa/ con los perfiles
finales, los errores medios relativos frente a la solución exacta, el registro
de costo completo y el tiempo de simulador. --merge une todos en
datos/validacion_qa.json con las claves:
  clasico_exacto, circuito_s{disparos}_semilla{semilla} (con el m de trabajo)
  y circuito_m{m}_s{disparos}_semilla{semilla} para otros m (barrido de m
  sobre la tobera: --m 3, 4, 6, 7). Con --mapeo campo (cotas del mapeo afín
  tomadas del campo completo, por componente y paso) la clave es
  circuito_campo_m{m}_s{disparos}_semilla{semilla}.

Ejecución (desde qae_cfd/), una corrida por proceso para correrlas en paralelo:
    python3 scripts/generar_datos_validacion.py --modo clasico_exacto
    python3 scripts/generar_datos_validacion.py --modo circuito --shots 1 --seed 1
    python3 scripts/generar_datos_validacion.py --merge
Duración de una corrida con circuito y m = 5: unas 2.4 h con seis corridas en
paralelo (34 800 circuitos); con m = 3, unos 14 min.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
logging.disable(logging.CRITICAL)

from qae_cfd import NozzleFlow  # noqa: E402
from qae_cfd.baseline import DeterministicMean  # noqa: E402
from qae_cfd.cost import CostLedger  # noqa: E402
from qae_cfd.qae_grover import GroverMeanQAE  # noqa: E402
from qae_cfd.solver import HybridCFDSolver  # noqa: E402

CARPETA = RAIZ / "datos" / "validacion_qa"
SALIDA = RAIZ / "datos" / "validacion_qa.json"
N_PASOS = 400
N_CELDAS = 31
M_PRECISION = 5
NODOS = 4



def correr(modo: str, shots: int, seed: int, m: int = M_PRECISION,
           mapeo: str = "vector") -> dict:
    nozzle = NozzleFlow()
    ledger = CostLedger()
    if modo == "clasico_exacto":
        media = DeterministicMean(ledger)
    else:
        media = GroverMeanQAE(m_precision=m, shots=shots, seed=seed,
                              ledger=ledger)
    s = HybridCFDSolver(nozzle, n_points=N_CELDAS, courant=0.4,
                        p_exit=0.678, order=2,
                        time_integration="qae_integral",
                        n_integral_nodes=NODOS, mean_estimator=media,
                        mapeo=mapeo)
    r = s.solve(n_steps=N_PASOS)
    M_e, rho_e, T_e = nozzle.exact_solution(r.x)
    p_e = rho_e * T_e  # gas ideal adimensional: p = rho T
    err = lambda a, b: float(np.mean(np.abs(a - b) / np.abs(b)) * 100)
    salida = {
        "modo": modo, "shots": shots if modo == "circuito" else None,
        "seed": seed if modo == "circuito" else None,
        "m": m if modo == "circuito" else None, "mapeo": mapeo, "nodos": NODOS,
        "celdas": N_CELDAS, "pasos": N_PASOS,
        "x": r.x.tolist(),
        "M": r.M.tolist(), "rho": r.rho.tolist(),
        "p": r.p.tolist(), "T": r.T.tolist(),
        "M_exact": M_e.tolist(), "rho_exact": rho_e.tolist(),
        "p_exact": p_e.tolist(), "T_exact": T_e.tolist(),
        "err_M": err(r.M, M_e), "err_rho": err(r.rho, rho_e),
        "err_p": err(r.p, p_e),
        "Mmax": float(r.M.max()), "Mmax_exacto": float(M_e.max()),
        "t": r.elapsed_seconds,
        "costo": ledger.to_dict(),
    }
    if modo == "circuito":
        salida["compuertas_por_ejecucion"] = media.gate_count(NODOS)
        salida["device"] = media._device_name.get(NODOS)
    return salida


def clave(modo: str, shots: int, seed: int, m: int = M_PRECISION,
          mapeo: str = "vector") -> str:
    """Las corridas con el m de trabajo y mapeo por vector conservan su nombre."""
    if modo == "clasico_exacto":
        return modo
    pref = "circuito" if mapeo == "vector" else "circuito_campo"
    if m == M_PRECISION and mapeo == "vector":
        return f"circuito_s{shots}_semilla{seed}"
    return f"{pref}_m{m}_s{shots}_semilla{seed}"


def merge() -> None:
    datos = {}
    for f in sorted(CARPETA.glob("*.json")):
        datos[f.stem] = json.loads(f.read_text())
    SALIDA.write_text(json.dumps(datos))
    print(f"{SALIDA.name}: {sorted(datos)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["clasico_exacto", "circuito"])
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--m", type=int, default=M_PRECISION,
                    help="cúbits de precisión (por defecto el de trabajo, 5)")
    ap.add_argument("--mapeo", choices=["vector", "campo"], default="vector",
                    help="cotas del mapeo afín: por vector (defecto) o por campo")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    CARPETA.mkdir(parents=True, exist_ok=True)
    if args.modo:
        k = clave(args.modo, args.shots, args.seed, args.m, args.mapeo)
        archivo = CARPETA / f"{k}.json"
        if archivo.exists():
            print(f"[{k}] ya corrido; se omite.", flush=True)
        else:
            print(f"[{k}] corriendo...", flush=True)
            d = correr(args.modo, args.shots, args.seed, args.m, args.mapeo)
            archivo.write_text(json.dumps(d))
            print(f"[{k}] err_M={d['err_M']:.4f}%  err_rho={d['err_rho']:.4f}%  "
                  f"err_p={d['err_p']:.4f}%  ({d['t']:.1f} s)  {d['costo']}",
                  flush=True)
    if args.merge:
        merge()


if __name__ == "__main__":
    main()
