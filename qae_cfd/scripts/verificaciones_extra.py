#!/usr/bin/env python3
"""Verificaciones clásicas que respaldan afirmaciones puntuales de los
Capítulos 3 y 5.

Genera datos/verificaciones_extra.json con tres bloques. Todos usan la media
temporal exacta, es decir el promedio determinista de los M = 4 nodos
temporales, que es la línea base clásica del trabajo:

  roe_31c400  : la tobera del benchmark (31 celdas, 400 pasos, CFL 0.4,
                p_e/p0 = 0.678, MUSCL orden 2) resuelta con Roe.
  hllc_31c400 : el mismo benchmark resuelto con HLLC en lugar de Roe.
  p_exit_0670 : el benchmark con Roe a p_e/p0 = 0.670 (el valor del caso
                OpenFOAM), para comparar la celda del choque con la corrida
                a 0.678.

La verificación de la cota del estimador con circuitos está en
scripts/barrido_estimador_circuito.py.

Ejecución:  python3 scripts/verificaciones_extra.py   (desde qae_cfd/)
Duración aproximada: unos segundos.
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

SALIDA = RAIZ / "datos" / "verificaciones_extra.json"



def correr_tobera(celdas, pasos, p_exit, riemann):
    nozzle = NozzleFlow()
    s = HybridCFDSolver(nozzle, n_points=celdas, courant=0.4,
                        p_exit=p_exit, order=2,
                        time_integration="qae_integral",
                        n_integral_nodes=4, mean_estimator=DeterministicMean(),
                        riemann_solver=riemann)
    r = s.solve(n_steps=pasos)
    M_e, rho_e, T_e = nozzle.exact_solution(r.x)
    p_e = rho_e * T_e
    err = lambda a, b: float(np.mean(np.abs(a - b) / np.abs(b)) * 100)
    ich = int(np.argmin(np.diff(r.M)))
    return {
        "celdas": celdas, "pasos": pasos, "p_exit": p_exit,
        "riemann": riemann,
        "err_M": err(r.M, M_e), "err_rho": err(r.rho, rho_e),
        "err_p": err(r.p, p_e),
        "x_choque": float(r.x[ich]), "dx": float(r.x[1] - r.x[0]),
        "Mmax": float(r.M.max()), "Mmax_exacto": float(M_e.max()),
        "t": r.elapsed_seconds,
    }


def main() -> None:
    datos = {}
    for clave, p_exit, riemann in [("roe_31c400", 0.678, "roe"),
                                   ("hllc_31c400", 0.678, "hllc"),
                                   ("p_exit_0670", 0.670, "roe")]:
        d = correr_tobera(31, 400, p_exit, riemann)
        if clave == "p_exit_0670":
            # exact_solution() está parametrizada para 0.678: aquí solo valen
            # la celda del choque y el pico de Mach.
            d = {k: d[k] for k in ("celdas", "pasos", "p_exit", "riemann",
                                   "x_choque", "dx", "Mmax", "t")}
            print(f"[{clave}] choque x={d['x_choque']:.3f}", flush=True)
        else:
            print(f"[{clave}] err_M={d['err_M']:.3f}%  err_rho={d['err_rho']:.3f}%  "
                  f"err_p={d['err_p']:.3f}%  choque x={d['x_choque']:.3f}", flush=True)
        datos[clave] = d
    SALIDA.write_text(json.dumps(datos, indent=1))
    print("verificaciones_extra.json listo.")


if __name__ == "__main__":
    main()
