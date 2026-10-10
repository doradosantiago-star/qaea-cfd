#!/usr/bin/env python3
"""Diagnóstico de la media reescalada en la corrida de la tobera (Capítulo 5).

Repite la corrida de la línea base clásica (31 celdas, 400 pasos, M = 4,
suma determinista) y anota, en cada una de las 34 800 estimaciones de media
temporal, el rango f_max − f_min de las cuatro muestras y la media reescalada
ḡ = media((f − f_min)/(f_max − f_min)). No ejecuta ningún circuito.

Lo que mide: cuánto se aleja ḡ de 0.5. Con nodos igualmente espaciados que
incluyen los extremos y un integrando casi lineal en τ, las muestras
reescaladas quedan en (0, 1/3, 2/3, 1) y ḡ = 0.5 + O(Δt). El valor 0.5 está en
la rejilla sin²(πk/2^m) del estimador para todo m ≥ 2 (k = 2^{m−2}), de modo
que la lectura más probable devuelve exactamente 0.5 siempre que
|θ(ḡ) − π/4| < π/2^{m+1}, es decir siempre que la fase quede más cerca de ese
entero que de sus vecinos. El guion cuenta en qué fracción de las estimaciones
ocurre eso para cada m, que es la fracción en que la resolución 2^{-m} no
interviene.

Salida: datos/diagnostico_media_reescalada.json.
Ejecución (desde qae_cfd/): python3 scripts/diagnostico_media_reescalada.py
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
from qae_cfd.solver import HybridCFDSolver  # noqa: E402

SALIDA = RAIZ / "datos" / "diagnostico_media_reescalada.json"
N_CELDAS, N_PASOS, NODOS = 31, 400, 4
MS = (2, 3, 4, 5, 6, 7, 8, 10)


class MediaQueAnota:
    """Suma determinista que registra ḡ y el rango de cada estimación."""

    def __init__(self):
        self.g_barra, self.rango = [], []

    def estimate_mean_general(self, f):
        f = np.asarray(f, dtype=float)
        ancho = float(f.max() - f.min())
        self.rango.append(ancho)
        if ancho > 0.0:
            self.g_barra.append(float(np.mean((f - f.min()) / ancho)))
        else:
            self.g_barra.append(0.0)
        return float(np.mean(f))


def main() -> None:
    media = MediaQueAnota()
    s = HybridCFDSolver(NozzleFlow(), n_points=N_CELDAS, courant=0.4, p_exit=0.678,
                        order=2, time_integration="qae_integral",
                        n_integral_nodes=NODOS, mean_estimator=media)
    s.solve(n_steps=N_PASOS)
    g = np.array(media.g_barra)
    rango = np.array(media.rango)
    con_rango = rango > 0
    d = np.abs(g[con_rango] - 0.5)
    theta = np.arcsin(np.sqrt(g[con_rango]))        # sin²θ = ḡ
    desvio_fase = np.abs(theta - np.pi / 4)        # distancia a la fase de ḡ = 0.5
    salida = {
        "estimaciones": int(len(g)), "con_rango_positivo": int(con_rango.sum()),
        "rango_medio": float(rango.mean()), "rango_mediana": float(np.median(rango)),
        "rango_max": float(rango.max()),
        "desvio_de_0.5": {"media": float(d.mean()), "mediana": float(np.median(d)),
                          "p90": float(np.percentile(d, 90)),
                          "p99": float(np.percentile(d, 99)), "max": float(d.max()),
                          "fraccion_exactamente_cero": float(np.mean(d < 1e-12))},
        # fracción de estimaciones cuya fase queda más cerca del entero de 0.5
        # que de sus vecinos: el entero más probable devuelve 0.5 exacto
        "fraccion_rejilla_en_0.5_por_m": {
            str(m): float(np.mean(desvio_fase < np.pi / 2 ** (m + 1))) for m in MS},
        # error de estimación por paso si la lectura cae en ese entero: Δt·rango·|ḡ−0.5|
        "error_si_lectura_en_0.5_medio": float(np.mean(rango[con_rango] * d)),
    }
    SALIDA.write_text(json.dumps(salida, indent=1))
    print(json.dumps(salida, indent=1))


if __name__ == "__main__":
    main()
