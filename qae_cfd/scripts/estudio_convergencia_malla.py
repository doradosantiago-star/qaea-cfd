#!/usr/bin/env python3
"""Estudio de convergencia de malla para la validación de la tesis.

Corre la tobera de Anderson con 31, 61, 121 y 241 celdas (modo integral,
media temporal exacta, Roe, CFL 0.4). Cada malla se marcha hasta su
equilibrio discreto: duplicar los pasos no cambia el error en la tercera cifra.

Guarda los errores medios relativos de M, rho y p contra la solución exacta
en datos/convergencia_malla.json (escritura incremental por malla).

Ejecución:  python3 scripts/estudio_convergencia_malla.py   (desde qae_cfd/)
Duración aproximada: un minuto (dominada por la malla de 241 celdas).
"""
from __future__ import annotations

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
from qae_cfd.solver import HybridCFDSolver  # noqa: E402

SALIDA = RAIZ / "datos" / "convergencia_malla.json"
# Horizontes verificados: cada malla se marcha hasta su equilibrio discreto
# (duplicar los pasos no cambia el error en la tercera cifra).
MALLAS_EXACTO = [(31, 800), (61, 1600), (121, 6400), (241, 12800)]



def main() -> None:
    datos = json.loads(SALIDA.read_text()) if SALIDA.exists() else {}
    nozzle = NozzleFlow()
    variantes = [(f"{c}_exacto", p, "exacto") for c, p in MALLAS_EXACTO]
    for clave, pasos, tipo in variantes:
        celdas = int(clave.split("_")[0])
        if clave in datos:
            print(f"[{clave}] ya corrido; se omite.", flush=True)
            continue
        print(f"[{clave}, {pasos} pasos] corriendo...", flush=True)
        media = DeterministicMean()
        s = HybridCFDSolver(nozzle, n_points=celdas, courant=0.4,
                            p_exit=0.678, order=2,
                            time_integration="qae_integral",
                            n_integral_nodes=4,
                            mean_estimator=media)
        r = s.solve(n_steps=pasos)
        M_e, rho_e, T_e = nozzle.exact_solution(r.x)
        p_e = rho_e * T_e
        err = lambda a, b: float(np.mean(np.abs(a - b) / np.abs(b)) * 100)
        suave = np.abs(r.x - 2.1) > 0.15   # region suave: excluye la vecindad del choque
        err_s = lambda a, b: float(np.mean(np.abs(a[suave] - b[suave]) / np.abs(b[suave])) * 100)
        ich = int(np.argmin(np.diff(r.M)))
        datos[clave] = {
            "tipo": tipo,
            "celdas": celdas, "pasos": pasos, "dx": float(r.x[1] - r.x[0]),
            "err_M": err(r.M, M_e), "err_rho": err(r.rho, rho_e),
            "err_p": err(r.p, p_e),
            "err_M_suave": err_s(r.M, M_e), "err_rho_suave": err_s(r.rho, rho_e),
            "x_choque": float(r.x[ich]),
            "t": r.elapsed_seconds,
        }
        SALIDA.write_text(json.dumps(datos, indent=1))
        d = datos[clave]
        print(f"[{clave}] err_M={d['err_M']:.2f}%  err_rho={d['err_rho']:.2f}%  "
              f"err_p={d['err_p']:.2f}%  choque en x={d['x_choque']:.2f}  ({d['t']:.0f} s)",
              flush=True)
    print("convergencia_malla.json listo.")


if __name__ == "__main__":
    main()
