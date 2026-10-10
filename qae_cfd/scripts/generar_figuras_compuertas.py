"""
Figuras a nivel de compuerta del circuito QAE (Capítulo 4).

Tres figuras, todas derivadas del circuito que ejecuta ``GroverMeanQAE``
con M = 4 y m = 5, para g = (0.2, 0.5, 0.7, 0.9):

1. fig_qae_canonico_es       : el esquema de estimación de amplitud de
                               Brassard, Høyer, Mosca y Tapp (2002), con A y
                               Q como bloques. Es la referencia de la
                               literatura.
2. fig_q_compuertas_es       : una aplicación de Q = −A S_0 A† S_χ abierta
                               a compuertas, tomada de la cinta real (las
                               operaciones controladas por el último cúbit
                               de precisión, sin su control). Se verifica
                               que su matriz coincide con la de Q extraída
                               del circuito completo.
3. fig_qft_inversa_es        : la transformada de Fourier inversa sobre los
                               m = 5 cúbits de precisión, abierta a
                               Hadamard, rotaciones de fase controladas y
                               SWAP, con la descomposición de PennyLane. Se
                               verifica que su matriz coincide con QFT†.

Salida: Imagenes/*.png y *.pdf (300 dpi).
Ejecución: python3 scripts/generar_figuras_compuertas.py   (desde qae_cfd/)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pennylane as qml

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
TESIS_ROOT = HERE.parents[2]
sys.path.insert(0, str(PKG_ROOT))

from qae_cfd.qae_grover import GroverMeanQAE  # noqa: E402

M, M_PREC = 4, 5
G_EXAMPLE = np.array([0.2, 0.5, 0.7, 0.9])
N_IDX = int(np.log2(M))
IDX = list(range(N_IDX))
ANC = N_IDX
PREC = list(range(N_IDX + 1, N_IDX + 1 + M_PREC))
WORK = IDX + [ANC]
OUT = TESIS_ROOT / "Imagenes"

ETIQ_TRABAJO = {"0": r"$q_0$  índice", "1": r"$q_1$  índice", "2": r"$q_2$  ancilla"}
ETIQ_PREC = {str(w): rf"$q_{w}$  precisión" for w in PREC}


def etiquetar(ax, etiquetas, tamano=16):
    for txt in ax.texts:
        if txt.get_text() in etiquetas:
            txt.set_text(etiquetas[txt.get_text()])
            txt.set_horizontalalignment("right")
            txt.set_fontsize(max(txt.get_fontsize(), tamano))  # legible impreso


def guardar(fig, nombre):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{nombre}.{ext}", dpi=300, bbox_inches="tight")
    print("guardado", nombre)


# ── cinta real ─────────────────────────────────────────────────────────
thetas = 2.0 * np.arcsin(np.sqrt(G_EXAMPLE))
est = GroverMeanQAE(m_precision=M_PREC)
qnode = est._build_qnode(M)
tape = qml.workflow.construct_tape(qnode)(thetas)

# operaciones de una sola aplicación de Q: las controladas por el último
# cúbit de precisión, que aplica Q^1
w_last = PREC[-1]
ops_q_ctrl = [op for op in tape.operations
              if w_last in op.wires and op.name != "Hadamard" and "QFT" not in op.name]


def sin_control_de_precision(op):
    """Quita únicamente el control del cúbit de precisión y conserva los
    controles propios de la compuerta (índice → ancilla en las RY
    multiplexadas; índice → ancilla en la Z multicontrolada de S_0)."""
    if not hasattr(op, "control_wires") or w_last not in op.control_wires:
        return op
    restantes = [(w, v) for w, v in zip(op.control_wires, op.control_values) if w != w_last]
    if not restantes:
        return op.base
    return qml.ctrl(op.base, control=[w for w, _ in restantes], control_values=[v for _, v in restantes])


def angulo_legible(op):
    """RY tiene periodo 4π: simplify escribe −θ como 4π − θ; se devuelve −θ."""
    if op.name == "C(RY)" and float(op.parameters[0]) > 2 * np.pi:
        return qml.ctrl(qml.RY(float(op.parameters[0]) - 4 * np.pi, wires=op.base.wires),
                        control=list(op.control_wires), control_values=list(op.control_values))
    return op


ops_q = [angulo_legible(qml.simplify(sin_control_de_precision(op))) for op in ops_q_ctrl]
ops_q = [op for op in ops_q if op.name != "GlobalPhase"]
print("compuertas por aplicación de Q (sin la fase global):", len(ops_q))
print("  ", [op.name for op in ops_q])

# matriz de Q desde el bloque controlado completo (como en la figura del circuito)
m_full = qml.matrix(qml.tape.QuantumScript(ops_q_ctrl), wire_order=[w_last] + WORK)
dim = 2 ** len(WORK)
q_mat = m_full[dim:, dim:]
# matriz de las compuertas sueltas (sin control): debe ser Q salvo la fase global −1
q_sueltas = qml.matrix(qml.tape.QuantumScript(ops_q), wire_order=WORK)
assert np.allclose(q_sueltas, -q_mat, atol=1e-10) or np.allclose(q_sueltas, q_mat, atol=1e-10)
print("verificación: compuertas de Q == matriz de Q (salvo fase global): OK")

# ── figura 2: Q abierta a compuertas ───────────────────────────────────
n_a = N_IDX + M
fig, ax = qml.drawer.tape_mpl(qml.tape.QuantumScript(ops_q), decimals=2, style="black_white",
                              fontsize=20, wire_order=WORK)
etiquetar(ax, ETIQ_TRABAJO)
guardar(fig, "fig_q_compuertas_es")

# ── figura 3: QFT inversa abierta a compuertas ─────────────────────────


with qml.QueuingManager.stop_recording():
    ops_qft = []
    for op in qml.adjoint(qml.QFT(wires=PREC)).decomposition():
        op = qml.simplify(op)
        if op.name == "ControlledPhaseShift":
            phi = float(op.parameters[0])
            phi = phi - 2 * np.pi if phi > np.pi else phi          # −π/2^k en lugar de 2π − π/2^k
            op = qml.ControlledPhaseShift(phi, wires=op.wires)
        ops_qft.append(op)


print("compuertas de la QFT inversa (m = 5):", len(ops_qft),
      {n: [o.name for o in ops_qft].count(n) for n in set(o.name for o in ops_qft)})
mat_dec = qml.matrix(qml.tape.QuantumScript(ops_qft), wire_order=PREC)
mat_ref = qml.matrix(qml.adjoint(qml.QFT(wires=PREC)), wire_order=PREC)
assert np.allclose(mat_dec, mat_ref, atol=1e-10)
print("verificación: descomposición == QFT†: OK")
fig, ax = qml.drawer.tape_mpl(qml.tape.QuantumScript(ops_qft), decimals=2, style="black_white",
                              fontsize=18, wire_order=PREC)
etiquetar(ax, ETIQ_PREC)
guardar(fig, "fig_qft_inversa_es")


# ── figura 1: esquema canónico de Brassard et al. ──────────────────────
class Bloque(qml.operation.Operation):
    num_wires = None
    grad_method = None

    def __init__(self, etiqueta, wires):
        self._etq = etiqueta
        super().__init__(wires=wires)

    def matrix(self, wire_order=None):  # noqa: ARG002
        return np.eye(2 ** len(self.wires))

    def label(self, decimals=None, base_label=None, cache=None):
        return self._etq


devc = qml.device("default.qubit", wires=["s"] + PREC)


@qml.qnode(devc)
def circ_canonico():
    Bloque(r"$A$", wires=["s"])
    for w in PREC:
        qml.Hadamard(wires=w)
    for i, w in enumerate(PREC):
        p = 2 ** (M_PREC - 1 - i)
        qml.ctrl(Bloque(r"$Q$" if p == 1 else rf"$Q^{{{p}}}$", wires=["s"]), control=w)
    Bloque(r"$\mathrm{QFT}^\dagger$", wires=PREC)
    return qml.probs(wires=PREC)


# La figura completa se imprime a media página de ancho: letra grande.
fig, ax = qml.draw_mpl(circ_canonico, style="black_white", fontsize=24, show_all_wires=True)()
etiquetar(ax, {"s": r"$|0\rangle^{\otimes n}$  sistema", **ETIQ_PREC}, tamano=28)
guardar(fig, "fig_qae_canonico_es")

# ── conteo para el texto ───────────────────────────────────────────────
n_q_apps = 2 ** M_PREC - 1
print(f"total: A = {n_a} compuertas; Q = {len(ops_q)} por aplicación × {n_q_apps} aplicaciones "
      f"= {len(ops_q) * n_q_apps}; QFT† = {len(ops_qft)}; Hadamard de precisión = {M_PREC}")
