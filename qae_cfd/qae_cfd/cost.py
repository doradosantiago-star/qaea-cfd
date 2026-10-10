"""
Contabilidad de costo computacional del solucionador híbrido.

Métrica acordada con el Prof. Jaramillo (24 de marzo de 2026) y
ampliada tras la revisión del Prof. Calderón (1 de octubre de 2026): el
costo no se reporta en segundos de reloj sino en cantidades
algorítmicas contadas por el propio código, y el registro desglosa
TODO lo que una estimación de media consume, no solo la convención de
llamadas al oráculo:

- ``oracle_calls``: aplicaciones del operador de Grover Q, con la
  convención de Brassard et al. (2^m por ejecución del circuito), por
  el número de ejecuciones.
- ``circuit_executions``: ejecuciones completas del circuito (disparos).
- ``controlled_ry``: rotaciones RY controladas aplicadas. Crecen con M,
  porque cada aplicación de A o A† contiene M de ellas.
- ``logical_gates``: compuertas del circuito contadas sobre la cinta de
  PennyLane, con la transformada de Fourier inversa abierta en sus
  compuertas (679 por ejecución con M = 4 y m = 5).
- ``classical_rescale_ops``: lecturas clásicas del vector para el
  reescalado a [0, 1] (mínimo, máximo y normalización: 3M por estimación).
- ``classical_reads``: lecturas del vector que hace la suma determinista de
  la línea base clásica (M por media, ver ``baseline.DeterministicMean``).
- ``classical_bound_reads``: lecturas del campo para fijar las cotas del
  mapeo por campo (2·M·N_interior por componente y paso).
- ``flux_evals``: evaluaciones de flujo de Riemann del solucionador de
  volúmenes finitos, una por interfaz por evaluación del lado derecho.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostLedger:
    """Libro de costos compartido entre estimador y solucionador."""

    oracle_calls: int = 0
    qae_estimates: int = 0
    circuit_executions: int = 0
    controlled_ry: int = 0
    logical_gates: int = 0
    classical_rescale_ops: int = 0
    classical_reads: int = 0
    classical_bound_reads: int = 0
    flux_evals: int = 0
    snapshots: list = field(default_factory=list)

    def charge_estimate(self, m: int, M: int, shots: int,
                        logical_gates: int) -> None:
        """Carga una estimación de media con m cúbits de precisión sobre
        M valores, ejecutada ``shots`` veces."""
        self.qae_estimates += 1
        self.circuit_executions += int(shots)
        self.oracle_calls += int(shots) * 2 ** int(m)
        # A una vez, y A† y A dentro de cada una de las 2^m − 1 aplicaciones de Q
        self.controlled_ry += int(shots) * int(M) * (2 * (2 ** int(m) - 1) + 1)
        self.logical_gates += int(shots) * int(logical_gates)

    def charge_rescale(self, M: int) -> None:
        """Carga el reescalado clásico de un vector de M valores."""
        self.classical_rescale_ops += 3 * int(M)

    def charge_bounds(self, n: int) -> None:
        """Carga n lecturas del campo para fijar las cotas del mapeo."""
        self.classical_bound_reads += int(n)

    def charge_reads(self, M: int) -> None:
        """Carga las M lecturas de la suma determinista de una media."""
        self.classical_reads += int(M)

    def charge_flux(self, n: int) -> None:
        """Carga n evaluaciones de flujo de Riemann."""
        self.flux_evals += int(n)

    def snapshot(self, label: str) -> dict:
        """Guarda y retorna el estado actual con una etiqueta."""
        s = dict(label=label, **self.to_dict())
        self.snapshots.append(s)
        return s

    def to_dict(self) -> dict:
        return dict(oracle_calls=self.oracle_calls,
                    qae_estimates=self.qae_estimates,
                    circuit_executions=self.circuit_executions,
                    controlled_ry=self.controlled_ry,
                    logical_gates=self.logical_gates,
                    classical_rescale_ops=self.classical_rescale_ops,
                    classical_reads=self.classical_reads,
                    classical_bound_reads=self.classical_bound_reads,
                    flux_evals=self.flux_evals)

    def __repr__(self) -> str:
        campos = ", ".join(f"{k}={v}" for k, v in self.to_dict().items())
        return f"CostLedger({campos})"
