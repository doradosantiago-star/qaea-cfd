"""
QAEA-CFD: Estimación de Amplitud Cuántica aplicada a CFD.

Paquete híbrido cuántico-clásico para resolver las ecuaciones de Euler
cuasi-1D (tobera convergente-divergente) mediante Volúmenes Finitos
acoplados con Estimación de Amplitud Cuántica (QAE).

Dónde entra el algoritmo cuántico y qué se obtiene de él
--------------------------------------------------------
El paso temporal se escribe en forma integral,

    U(t+Δt) = U(t) + ∫₀^Δt f[U(τ)] dτ  ≈  U(t) + Δt · media_τ(f),

y esa MEDIA temporal del lado derecho es lo que estima el circuito
cuántico (módulo ``qae_grover``): el operador A codifica las M
evaluaciones del integrando como amplitudes, el operador de Grover Q
las rota un ángulo 2θ con sin²θ = media, y la estimación de fase lee
θ. Lo que se obtiene es la media estimada ã por celda y componente,
con error O(1/2^m); acumulada paso a paso produce los campos ρ, u, p
y Mach de la tobera, que se validan contra la solución exacta de
Anderson. El costo de cada estimación (disparos, aplicaciones de Q,
rotaciones controladas, compuertas, reescalado clásico) lo cuenta el
módulo ``cost``.

Módulos
-------
qae_grover : QAE canónico de Brassard (A, Q, QPE) que estima medias
             ejecutando y midiendo el circuito en PennyLane, con un
             número finito de disparos. Este es el núcleo cuántico.
baseline   : Suma determinista de la media (línea base clásica), con
             la misma interfaz que el estimador cuántico.
riemann    : Solvers de Riemann aproximados (Roe, HLLC).
nozzle     : Geometría de tobera y solución exacta del flujo cuasi-1D.
solver     : Solver híbrido FVM + QAE para las ecuaciones de Euler.
cost       : Registro de costo (llamadas al oráculo, disparos, compuertas,
             lecturas clásicas, evaluaciones de flujo).
utils      : Utilidades numéricas (Newton-Raphson, limitadores, etc.).

Autor  : Santiago Dorado
Afil.  : Universidad del Valle
Fecha  : 2026
"""

__version__ = "2.1.0"
__author__ = "Santiago Dorado"

from .baseline import DeterministicMean
from .qae_grover import GroverMeanQAE
from .cost import CostLedger
from .riemann import RoeSolver, HLLCSolver
from .nozzle import NozzleFlow
from .solver import HybridCFDSolver

__all__ = [
    "DeterministicMean",
    "GroverMeanQAE",
    "CostLedger",
    "RoeSolver",
    "HLLCSolver",
    "NozzleFlow",
    "HybridCFDSolver",
]
