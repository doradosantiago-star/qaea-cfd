#!/usr/bin/env python3
"""
QAEA-CFD: demostración del solucionador híbrido en la tobera de Anderson.

Resuelve las ecuaciones de Euler cuasi-1D con volúmenes finitos (MUSCL +
Riemann de Roe o HLLC) en la forma integral del paso de tiempo: la media
temporal del lado derecho de cada celda y componente la estima el circuito
de estimación de amplitud de Brassard, Høyer, Mosca y Tapp, ejecutado y
medido en PennyLane en cada estimación. No hay ruta analítica.

Uso:
    python main.py                       # 31 celdas, 100 pasos, m = 5, 1 disparo
    python main.py --pasos 400 --shots 25 # mediana de 25 disparos por estimación
    python main.py --qubits 6 --seed 7

Los experimentos del documento están en scripts/ (barrido del estimador,
validación de la tobera, figuras).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qae_cfd import GroverMeanQAE, HybridCFDSolver, NozzleFlow, CostLedger  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def demo_nozzle(m_precision: int, shots: int, seed: int, n_points: int,
                n_steps: int) -> None:
    nozzle = NozzleFlow(gamma=1.4, x_shock=2.1)
    x = np.linspace(0, nozzle.length, n_points)
    M_exact, _, _ = nozzle.exact_solution(x)

    print("═" * 60)
    print("  Tobera convergente-divergente: volúmenes finitos + QAE")
    print("═" * 60)
    for riemann in ("roe", "hllc"):
        ledger = CostLedger()
        media = GroverMeanQAE(m_precision=m_precision, shots=shots, seed=seed,
                              ledger=ledger)
        solver = HybridCFDSolver(nozzle=nozzle, riemann_solver=riemann,
                                 n_points=n_points, mean_estimator=media)
        r = solver.solve(n_steps)
        err_M = 100 * np.mean(np.abs(r.M - M_exact) / np.maximum(np.abs(M_exact), 1e-10))
        print(f"\n─── Riemann: {riemann.upper()}   estimador: {media} ───")
        print(f"  Mach máximo : {np.max(r.M):.4f} (exacto: {np.max(M_exact):.4f})")
        print(f"  Error Mach  : {err_M:.2f} % (medio frente a la solución exacta)")
        print(f"  Tiempo      : {r.elapsed_seconds:.1f} s de simulador")
        print(f"  Costo       : {ledger}")
        print(f"  Compuertas por ejecución del circuito: {media.gate_count(solver.n_integral_nodes)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="QAEA-CFD: estimación de amplitud aplicada a CFD")
    ap.add_argument("--qubits", type=int, default=5, help="cúbits de precisión m (default 5)")
    ap.add_argument("--shots", type=int, default=1, help="disparos por estimación (default 1)")
    ap.add_argument("--seed", type=int, default=1, help="semilla de la medición (default 1)")
    ap.add_argument("--celdas", type=int, default=31, help="celdas de la malla (default 31)")
    ap.add_argument("--pasos", type=int, default=100, help="pasos de tiempo (default 100)")
    args = ap.parse_args()
    demo_nozzle(args.qubits, args.shots, args.seed, args.celdas, args.pasos)
    print("\n✓ Ejecución completada.")


if __name__ == "__main__":
    main()
