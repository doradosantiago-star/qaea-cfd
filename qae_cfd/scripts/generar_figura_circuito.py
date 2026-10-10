"""
Figura del circuito QAE real de la tesis (fig_qaea_circuit_real).

Dibuja el circuito EXACTO que ejecuta ``GroverMeanQAE`` en el punto de
trabajo de la tesis: M = 4 nodos de
cuadratura, m = 5 cúbits de precisión → 8 cúbits y 2^5 − 1 = 31
aplicaciones controladas de Q.

Fidelidad garantizada por construcción, no por re-implementación:

1. Se construye el QNode real con ``GroverMeanQAE._build_qnode(4)``
   y se extrae su cinta (tape) para un ejemplo concreto
   g = (0.2, 0.5, 0.7, 0.9).
2. Las compuertas de A (Hadamards del índice + CRY multiplexadas) se
   RE-USAN tal cual salen de la cinta real (no se redibujan a mano).
3. La matriz del operador de Grover Q se extrae numéricamente del
   propio circuito real (del bloque controlado por el último cúbit de
   precisión) y las potencias Q^{2^j} se dibujan como cajas
   etiquetadas Q, Q², Q⁴, Q⁸, Q¹⁶ cuya matriz es exactamente la del
   circuito.
4. Antes de guardar la figura, se VERIFICA que el circuito de
   presentación tiene la MISMA matriz unitaria que el circuito real
   (allclose a precisión de máquina). Después se ejecuta el circuito
   real una vez, con un disparo, y se imprime el entero k medido.

Salida: Imagenes/fig_qaea_circuit_real.pdf y .png (300 dpi).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pennylane as qml

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]          # .../qae_cfd
TESIS_ROOT = HERE.parents[2]        # raíz del repositorio (las figuras caen en Imagenes/)
sys.path.insert(0, str(PKG_ROOT))

from qae_cfd.qae_grover import GroverMeanQAE  # noqa: E402

# ── Punto de trabajo de la tesis ─────────────────────────────────────
M = 4                    # nodos de cuadratura → 2 cúbits de índice
M_PREC = 5               # cúbits de precisión → 31 aplicaciones de Q
G_EXAMPLE = np.array([0.2, 0.5, 0.7, 0.9])   # muestras g_j ∈ [0, 1]

N_IDX = int(np.log2(M))
IDX = list(range(N_IDX))                       # [0, 1]
ANC = N_IDX                                    # 2
PREC = list(range(N_IDX + 1, N_IDX + 1 + M_PREC))  # [3..7]
WORK = IDX + [ANC]
N_WIRES = N_IDX + 1 + M_PREC

# Ops por aplicación de Q en la cinta real:
#   S_chi (1) + A† (2 H + M CRY) + S_0 (2·(n_idx+1) X + 1 ctrl-Z)
#   + A (2 H + M CRY) + GlobalPhase (1)
OPS_PER_Q = 1 + (N_IDX + M) + (2 * (N_IDX + 1) + 1) + (N_IDX + M) + 1


def extraer_cinta_real(thetas):
    """QNode real del paquete y su cinta para los ángulos dados."""
    est = GroverMeanQAE(m_precision=M_PREC, shots=1, seed=1)
    qnode = est._build_qnode(M)
    tape = qml.workflow.construct_tape(qnode)(thetas)
    return est, qnode, tape


def extraer_matriz_Q(tape):
    """
    Matriz de Q extraída del propio circuito real: el bloque
    controlado por el último cúbit de precisión aplica Q una sola vez.
    Con wire_order = [control, trabajo...], la matriz del bloque es
    diag(I, Q) y Q es el bloque inferior derecho.
    """
    w_last = PREC[-1]
    ops_ctrl = [
        op for op in tape.operations
        if w_last in op.wires
        and op.name != "Hadamard"
        and "QFT" not in op.name
    ]
    assert len(ops_ctrl) == OPS_PER_Q, (
        f"esperaba {OPS_PER_Q} ops controladas por el cable {w_last}, "
        f"encontré {len(ops_ctrl)}"
    )
    sub_tape = qml.tape.QuantumScript(ops_ctrl)
    m_full = qml.matrix(sub_tape, wire_order=[w_last] + WORK)
    dim = 2 ** len(WORK)
    q_mat = m_full[dim:, dim:]
    # el bloque de control en |0⟩ debe ser la identidad
    assert np.allclose(m_full[:dim, :dim], np.eye(dim), atol=1e-10)
    assert np.allclose(m_full[:dim, dim:], 0.0, atol=1e-10)
    return q_mat


class QPower(qml.operation.Operation):
    """
    Potencia de Q dibujada como caja etiquetada (Q, Q², Q⁴, ...).

    No es una re-implementación: su matriz es la potencia exacta de la
    matriz de Q extraída del circuito real. Se define como Operation
    genérica (y no como QubitUnitary) para que ``qml.ctrl`` la envuelva
    en un ``Controlled`` que conserva la etiqueta al dibujar.
    """

    num_wires = 3
    grad_method = None

    def __init__(self, base_matrix, power, wires):
        self._mat = np.linalg.matrix_power(base_matrix, power)
        self._power = power
        super().__init__(wires=wires)

    def matrix(self, wire_order=None):  # noqa: ARG002
        return self._mat

    def label(self, decimals=None, base_label=None, cache=None):
        return "$Q$" if self._power == 1 else f"$Q^{{{self._power}}}$"


def construir_circuito_presentacion(a_ops, q_mat):
    """
    Mismo circuito, con A expandida (las compuertas reales de la
    cinta) y las potencias de Q colapsadas en cajas etiquetadas.
    """
    dev = qml.device("default.qubit", wires=N_WIRES)

    @qml.qnode(dev)
    def circuito():
        for op in a_ops:
            qml.apply(op)
        for w in PREC:
            qml.Hadamard(wires=w)
        for i, w in enumerate(PREC):
            reps = 2 ** (M_PREC - 1 - i)
            qml.ctrl(QPower(q_mat, reps, wires=WORK), control=w)
        qml.adjoint(qml.QFT)(wires=PREC)
        return qml.sample(wires=PREC)

    return circuito


def main():
    thetas = 2.0 * np.arcsin(np.sqrt(G_EXAMPLE))  # como estimate_mean
    est, qnode_real, tape = extraer_cinta_real(thetas)

    # A real: Hadamards del índice + M rotaciones CRY multiplexadas
    n_a_ops = N_IDX + M
    a_ops = tape.operations[:n_a_ops]
    nombres = [op.name for op in a_ops]
    assert nombres[:N_IDX] == ["Hadamard"] * N_IDX, nombres
    assert all("RY" in n for n in nombres[N_IDX:]), nombres

    # conteo de aplicaciones de Q en la cinta real
    ops_q = [
        op for op in tape.operations[n_a_ops:]
        if op.name != "Hadamard" and "QFT" not in op.name
    ]
    n_q_apps = len(ops_q) // OPS_PER_Q
    assert len(ops_q) % OPS_PER_Q == 0
    assert n_q_apps == 2 ** M_PREC - 1, n_q_apps

    q_mat = extraer_matriz_Q(tape)
    circ_pres = construir_circuito_presentacion(a_ops, q_mat)

    # ── VERIFICACIÓN: misma matriz unitaria que el circuito real ─────
    wire_order = list(range(N_WIRES))
    u_real = qml.matrix(tape, wire_order=wire_order)
    tape_pres = qml.workflow.construct_tape(circ_pres)()
    u_pres = qml.matrix(tape_pres, wire_order=wire_order)
    assert np.allclose(u_real, u_pres, atol=1e-9), np.max(np.abs(u_real - u_pres))

    # Una ejecución real, con un disparo: el entero k sale de la medición.
    a_hat = est.estimate_mean(G_EXAMPLE)
    k_medido = int(est._bits_to_int(est.last_samples)[0])
    a_ref = float(np.mean(G_EXAMPLE))  # solo como referencia del ejemplo
    bound = (2 * np.pi * np.sqrt(a_ref * (1 - a_ref)) / 2 ** M_PREC
             + np.pi ** 2 / 4 ** M_PREC)

    print(f"cúbits: {N_WIRES}  (índice {N_IDX} + ancilla 1 + "
          f"precisión {M_PREC})")
    print(f"aplicaciones de Q en la cinta real: {n_q_apps}")
    print(f"g = {G_EXAMPLE.tolist()},  media de referencia = {a_ref:.4f}")
    print(f"k medido en un disparo = {k_medido},  media estimada = {a_hat:.4f} "
          f"(cota BHMT = {bound:.4f}, cumplida: {abs(a_hat - a_ref) <= bound})")
    print("verificación circuito presentación == circuito real (matriz): OK")

    # ── Dibujo ───────────────────────────────────────────────────────
    fig, ax = qml.draw_mpl(
        circ_pres, decimals=2, style="black_white", fontsize=13,
    )()

    etiquetas = {
        "0": r"$q_0$  index",
        "1": r"$q_1$  index",
        "2": r"$q_2$  ancilla",
        **{str(w): rf"$q_{w}$  precision" for w in PREC},
    }
    for txt in ax.texts:
        if txt.get_text() in etiquetas:
            txt.set_text(etiquetas[txt.get_text()])
            txt.set_horizontalalignment("right")

    out_dir = TESIS_ROOT / "Imagenes"
    for nombre in ("fig_qaea_circuit_real.pdf", "fig_qaea_circuit_real.png"):
        fig.savefig(out_dir / nombre, dpi=300, bbox_inches="tight")
        print(f"guardado: {out_dir / nombre}")


if __name__ == "__main__":
    main()
