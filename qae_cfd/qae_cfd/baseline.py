"""
Línea base clásica: la media de las M muestras con la suma determinista.

Es el procedimiento que el estimador cuántico reemplaza dentro de la etapa
cinco del paso de tiempo. Lee las M muestras una vez y devuelve su media
aritmética sin error. Si recibe un registro de costo, le cobra esas M
lecturas (``classical_reads``), de modo que la columna clásica del
presupuesto del Capítulo 5 también sale del registro y no de una cuenta a
mano.
"""

from __future__ import annotations

import numpy as np

from .cost import CostLedger


class DeterministicMean:
    """Media aritmética de un vector real; misma interfaz que ``GroverMeanQAE``."""

    def __init__(self, ledger: CostLedger | None = None):
        self.ledger = ledger

    def estimate_mean_general(self, f: np.ndarray, bounds=None) -> float:
        f = np.asarray(f, dtype=float)
        if self.ledger is not None:
            self.ledger.charge_reads(len(f))
        return float(np.mean(f))

    def __repr__(self) -> str:
        return "DeterministicMean()"
