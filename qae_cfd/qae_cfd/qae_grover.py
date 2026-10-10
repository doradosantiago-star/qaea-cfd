"""
QAE canónico de Brassard, Høyer, Mosca y Tapp (2002): estimación de la
MEDIA de M valores g_j ∈ [0, 1] ejecutando y midiendo el circuito.

Cadena del algoritmo, tal como la ejecuta este módulo:

1. **Operador A** (codificación): prepara

   .. math::

       A|0\\rangle = \\frac{1}{\\sqrt{M}} \\sum_{j=0}^{M-1} |j\\rangle
       \\otimes \\big( \\cos\\theta_j |0\\rangle + \\sin\\theta_j |1\\rangle \\big),
       \\qquad \\theta_j = \\arcsin\\sqrt{g_j}

   La probabilidad de medir el ancilla en |1⟩ es la media a = (1/M) Σ g_j.

2. **Operador de Grover** Q = −A S₀ A† S_χ, rotación de ángulo 2θ_a en el
   plano ⟨malo, bueno⟩, con sin²θ_a = a.

3. **Estimación de fase**: m cúbits de precisión, potencias controladas
   Q^{2^j} y QFT inversa. El registro de precisión se MIDE: cada
   ejecución del circuito (un disparo) produce un entero k distribuido
   según la salida de la estimación de fase, y la estimación es
   ã = sin²(πk/2^m). Con probabilidad al menos 8/π² el resultado cumple
   |ã − a| ≤ 2π√(a(1−a))/2^m + π²/2^{2m} (Teorema 12 de Brassard et al.).

Lo que este módulo NO hace, a propósito: no calcula la media clásicamente
en ningún punto, no construye la distribución de salida de forma
analítica y no toma el máximo de probabilidades exactas. La única
fuente del entero k es la medición del registro en el simulador de
vector de estado de PennyLane (``lightning.qubit``, con ``default.qubit``
como respaldo), con un número finito de disparos.

Con ``shots > 1`` se aplica la receta de Brassard et al. para reducir la
probabilidad de fallo: se repite la medición y se toma la mediana de las
estimaciones individuales. Cada disparo es una ejecución completa del
circuito y se contabiliza como tal en el registro de costo.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class GroverMeanQAE:
    """
    Estimador QAE de la media de un vector g ∈ [0, 1]^M, M potencia de 2.

    Parameters
    ----------
    m_precision : int
        Cúbits de precisión de la estimación de fase. N = 2^m.
    shots : int
        Ejecuciones del circuito por estimación. Con 1, la estimación es
        la lectura de una sola medición. Con más, la mediana de las
        lecturas individuales.
    seed : int | None
        Semilla del generador de la medición del simulador. Fija la
        secuencia de disparos para que una corrida sea reproducible.
    ledger : CostLedger | None
        Registro de costo compartido (ver ``cost.CostLedger``).
    """

    def __init__(self, m_precision: int = 5, shots: int = 1,
                 seed: int | None = None, ledger=None):
        if m_precision < 1:
            raise ValueError("m_precision debe ser al menos 1")
        if shots < 1:
            raise ValueError("shots debe ser al menos 1")
        self.m = int(m_precision)
        self.shots = int(shots)
        self.seed = seed
        self.ledger = ledger
        self._rng = np.random.default_rng(seed)
        # Un QNode por tamaño M: el device y el circuito se construyen una
        # sola vez y los ángulos θ_j entran como parámetros.
        self._pl_cache: dict[int, object] = {}
        self._device_name: dict[int, str] = {}
        # Conteo de compuertas por ejecución, medido sobre la cinta real.
        self._gate_count: dict[int, dict] = {}
        self.last_samples: np.ndarray | None = None

    # ──────────────────────────────────────────────────────────────
    #  Métodos principales
    # ──────────────────────────────────────────────────────────────

    def estimate_mean(self, g: np.ndarray) -> float:
        """
        Estima a = media(g) para g ∈ [0, 1]^M ejecutando el circuito.

        Devuelve la estimación leída del registro de precisión: con un
        disparo, sin²(πk/2^m) para el k medido; con varios, la mediana de
        esas lecturas.
        """
        g = np.asarray(g, dtype=float)
        if np.any(g < -1e-12) or np.any(g > 1 + 1e-12):
            raise ValueError("estimate_mean requiere g ∈ [0, 1]")
        g = np.clip(g, 0.0, 1.0)
        M = len(g)
        if M & (M - 1):
            raise ValueError(f"M={M} debe ser potencia de 2")

        qnode = self._qnode(M)
        thetas = 2.0 * np.arcsin(np.sqrt(g))  # RY(2θ_j)|0⟩ = cosθ_j|0⟩ + sinθ_j|1⟩
        samples = np.atleast_2d(np.asarray(qnode(thetas)))  # (shots, m) bits
        self.last_samples = samples
        ks = self._bits_to_int(samples)
        estimates = np.sin(np.pi * ks / 2 ** self.m) ** 2

        if self.ledger is not None:
            self.ledger.charge_estimate(
                m=self.m, M=M, shots=self.shots,
                logical_gates=self.gate_count(M)["logical_qft_expanded"],
            )
        if self.shots == 1:
            return float(estimates[0])
        return float(np.median(estimates))

    def estimate_mean_general(self, f: np.ndarray,
                              bounds: tuple[float, float] | None = None) -> float:
        """
        Media de un vector real arbitrario: mapeo afín a [0, 1], estimación
        con el circuito y vuelta a la escala original.

        Sin ``bounds``, las cotas del mapeo son el mínimo y el máximo del
        propio vector (mapeo "por vector"): con un integrando casi lineal
        en los nodos, la media mapeada cae en 0.5, que está en la rejilla
        del estimador para todo m. Con ``bounds = (f_min, f_max)`` se usan
        esas cotas fijas (mapeo "por campo" o cotas conocidas): la media
        mapeada queda donde la ponga el vector y la resolución 2^-m sí
        interviene. El mapeo es clásico y se contabiliza en el registro. La
        media es lineal, así que media(f) = f_min + (f_max − f_min)·media(g).
        """
        f = np.asarray(f, dtype=float)
        M = len(f)
        if bounds is None:
            f_min, f_max = float(np.min(f)), float(np.max(f))
        else:
            f_min, f_max = float(bounds[0]), float(bounds[1])
        width = f_max - f_min
        if width > 0.0:
            g = np.clip((f - f_min) / width, 0.0, 1.0)
        else:
            g = np.zeros(M)
        if self.ledger is not None:
            self.ledger.charge_rescale(M)
        return f_min + width * self.estimate_mean(g)

    # ──────────────────────────────────────────────────────────────
    #  Circuito
    # ──────────────────────────────────────────────────────────────

    def _registers(self, M: int):
        n_idx = int(np.log2(M)) if M > 1 else 0
        idx = list(range(n_idx))
        anc = n_idx
        prec = list(range(n_idx + 1, n_idx + 1 + self.m))
        return n_idx, idx, anc, prec

    def _build_device(self, M: int):
        import pennylane as qml

        n_idx, _, _, _ = self._registers(M)
        n_wires = n_idx + 1 + self.m
        try:
            dev = qml.device("lightning.qubit", wires=n_wires, seed=self._rng)
            name = "lightning.qubit"
        except Exception:
            dev = qml.device("default.qubit", wires=n_wires, seed=self._rng)
            name = "default.qubit"
        self._device_name[M] = name
        return dev

    def _build_qnode(self, M: int):
        """
        QNode del circuito QAE completo para M valores, con los ángulos
        θ_j como parámetros. Mide el registro de precisión (``qml.sample``)
        con ``self.shots`` disparos.
        """
        import pennylane as qml

        n_idx, idx, anc, prec = self._registers(M)
        dev = self._build_device(M)

        def A_op(thetas):
            for w in idx:
                qml.Hadamard(wires=w)
            if n_idx == 0:
                qml.RY(thetas[0], wires=anc)
                return
            for j in range(M):
                ctrl_vals = [int(b) for b in format(j, f"0{n_idx}b")]
                qml.ctrl(qml.RY, control=idx, control_values=ctrl_vals)(
                    thetas[j], wires=anc
                )

        def S_chi():
            # Refleja los estados "buenos" (ancilla = |1⟩)
            qml.PauliZ(wires=anc)

        def S_0():
            # Refleja |0…0⟩ del registro de trabajo (índice + ancilla)
            wires = idx + [anc]
            for w in wires:
                qml.PauliX(wires=w)
            if n_idx == 0:
                qml.PauliZ(wires=anc)
            else:
                qml.ctrl(qml.PauliZ, control=wires[:-1])(wires=wires[-1])
            for w in wires:
                qml.PauliX(wires=w)

        def Q_op(thetas):
            # Q = −A S₀ A† S_χ  (Brassard et al. 2002, ec. 4)
            S_chi()
            qml.adjoint(A_op)(thetas)
            S_0()
            A_op(thetas)
            qml.GlobalPhase(np.pi)  # el signo −1, observable bajo control

        @qml.qnode(dev)
        def circuit(thetas):
            A_op(thetas)
            for w in prec:
                qml.Hadamard(wires=w)
            for i, w in enumerate(prec):
                reps = 2 ** (self.m - 1 - i)

                def repeated_Q():
                    for _ in range(reps):
                        Q_op(thetas)

                qml.ctrl(repeated_Q, control=w)()
            qml.adjoint(qml.QFT)(wires=prec)
            return qml.sample(wires=prec)

        return qml.set_shots(circuit, shots=self.shots)

    def _qnode(self, M: int):
        if M not in self._pl_cache:
            self._pl_cache[M] = self._build_qnode(M)
        return self._pl_cache[M]

    @staticmethod
    def _bits_to_int(samples: np.ndarray) -> np.ndarray:
        """Bits del registro de precisión (primer alambre = bit más
        significativo, el mismo orden que ``qml.probs``) a entero k."""
        m = samples.shape[1]
        pesos = 2 ** np.arange(m - 1, -1, -1)
        return samples.astype(int) @ pesos

    # ──────────────────────────────────────────────────────────────
    #  Conteo de compuertas medido sobre la cinta
    # ──────────────────────────────────────────────────────────────

    def gate_count(self, M: int) -> dict:
        """
        Compuertas de una ejecución del circuito para M valores, contadas
        sobre la cinta que PennyLane construye (no estimadas a mano).

        ``logical``: operaciones tal como las escribe el circuito (RY
        multicontroladas, Z multicontrolada, Hadamard, QFT inversa como
        un bloque, fase global bajo control). ``logical_qft_expanded``:
        lo mismo con la QFT inversa abierta en sus compuertas. ``controlled_ry``: cuántas de ellas son
        rotaciones RY controladas, que crecen con M. ``elementary``:
        compuertas de uno y dos cúbits tras descomponer, que es lo que un
        dispositivo ejecuta. ``depth_logical`` y ``depth_elementary``: la
        profundidad de cada cinta, es decir el número de capas de
        compuertas que actúan sobre cúbits distintos y pueden ejecutarse
        a la vez.
        """
        if M in self._gate_count:
            return self._gate_count[M]
        import pennylane as qml

        qnode = self._qnode(M)
        thetas = np.full(M, 0.5)
        tape = qml.workflow.construct_tape(qnode)(thetas)
        ops = tape.operations
        n_cry = sum(1 for op in ops if "RY" in op.name)

        def es_elemental(op):
            return len(op.wires) <= 2

        [tape_el], _ = qml.devices.preprocess.decompose(
            tape, stopping_condition=es_elemental,
        )
        n_el = len(tape_el.operations)
        n_2q = sum(1 for op in tape_el.operations if len(op.wires) == 2)
        # Profundidad: capas de compuertas que no comparten cúbits y pueden
        # ejecutarse a la vez, calculada sobre el grafo de la cinta.
        depth_logical = tape.graph.get_depth()
        depth_el = tape_el.graph.get_depth()
        # La QFT inversa entra en la cinta como un solo bloque; abierta en
        # Hadamard, fases controladas y SWAP son estas compuertas:
        _, _, _, prec = self._registers(M)
        n_qft = len(qml.adjoint(qml.QFT(wires=prec)).decomposition())
        self._gate_count[M] = dict(
            logical=len(ops), logical_qft_expanded=len(ops) - 1 + n_qft,
            controlled_ry=n_cry, elementary=n_el, two_qubit=n_2q,
            depth_logical=depth_logical, depth_elementary=depth_el,
            wires=len(tape.wires),
        )
        return self._gate_count[M]

    def __repr__(self) -> str:
        dev = next(iter(self._device_name.values()), "lightning.qubit")
        return (f"GroverMeanQAE(m_precision={self.m}, shots={self.shots}, "
                f"seed={self.seed}, device='{dev}')")
