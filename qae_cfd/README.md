# QAEA-CFD

Código del trabajo de grado: **Estimación de Amplitud Cuántica (QAEA) aplicada
al flujo compresible en una tobera convergente-divergente**, resuelto con un
solver híbrido de Volúmenes Finitos + QAE y validado contra la solución exacta
de Anderson (flujo isentrópico + choque normal).

Autor: Santiago Dorado, Universidad del Valle.

## Por qué se usa el QAE, dónde, y qué se obtiene de él

El punto de entrada del algoritmo cuántico sale de escribir
el paso de tiempo en su forma integral:

```
U(t+Δt) = U(t) + ∫₀^Δt f[U(τ)] dτ  ≈  U(t) + Δt · media_τ( f[ℓ(τ)] )
```

donde `f[U]` es el lado derecho semidiscreto del FVM (flujos de Riemann +
fuente geométrica) y `ℓ(τ)` es el predictor lineal local de Kacewicz. La
integral queda reducida a una **media de M evaluaciones del integrando**, y
esa media es la cantidad que el QAE de Brassard estima:

1. **Operador A** (codificación): prepara `(1/√M) Σⱼ |j⟩ (cosθⱼ|0⟩ + sinθⱼ|1⟩)`
   con `θⱼ = arcsin(√gⱼ)`. La probabilidad del ancilla en `|1⟩` es igual a la media.
2. **Operador de Grover Q**: una rotación de ángulo `2θ_a`, con
   `sin²(θ_a) = media`.
3. **Estimación de fase**: `m` cúbits de precisión, potencias controladas de
   `Q` y QFT⁻¹. El registro de precisión se **mide**: cada ejecución del
   circuito (un disparo) devuelve un entero `k` y la estimación es
   `sin²(πk/2^m)`. Con probabilidad al menos `8/π²` cumple la cota del
   Teorema 12 de Brassard et al.; la fracción de violaciones se mide en
   `scripts/barrido_estimador_circuito.py`.

**Lo que se obtiene del circuito** es la media estimada `ã` por celda y
componente en cada paso; multiplicada por `Δt` avanza el flujo. Acumulando
pasos se obtienen los campos ρ, u, p y Mach de la tobera, con error ~1.8 %
frente a la solución exacta (malla de 31 puntos).

## Ejecución del circuito

Cada estimación **ejecuta y mide el circuito completo** (A, Q, estimación de
fase) en `lightning.qubit` (`default.qubit` como respaldo), con los ángulos
del flujo como parámetros del circuito y un número finito de disparos
(`shots`, por defecto 1; con más de uno, mediana de las lecturas). No existe
una ruta analítica: el paquete no calcula la media clásicamente ni construye
la distribución de salida; el único origen del entero `k` es la medición. La
semilla (`seed`) fija la secuencia de disparos para reproducir una corrida.

El registro de costo (`cost.CostLedger`) cuenta por estimación los disparos,
las aplicaciones de `Q` (convención `2^m` por ejecución), las rotaciones RY
controladas (que crecen con `M`), las compuertas de la cinta y las lecturas
clásicas del reescalado a `[0, 1]`.

## Estructura

| Módulo | Qué hace |
|---|---|
| `qae_cfd/qae_grover.py` | QAE canónico de Brassard que estima medias ejecutando y midiendo el circuito (A, Q, QPE). |
| `qae_cfd/cost.py` | Registro de costo: disparos, llamadas al oráculo (2^m por ejecución), rotaciones controladas, compuertas, reescalado, lecturas de la suma, flujos. |
| `qae_cfd/solver.py` | Solver híbrido: FVM (MUSCL + Riemann) + integración temporal con QAE (modo `qae_integral`, el de la tesis). |
| `qae_cfd/nozzle.py` | Geometría A(x) = 1 + 2.2(x−1.5)², relaciones área-Mach y solución exacta (Anderson). |
| `qae_cfd/riemann.py` | Solvers de Riemann aproximados (Roe y HLLC). |
| `qae_cfd/baseline.py` | Línea base clásica: la media por suma determinista, con la misma interfaz que el estimador y sus lecturas cobradas al registro. |
| `qae_cfd/utils.py` | Newton-Raphson, limitador minmod, conversión de variables. |

## Qué devuelve cada función

| Llamada | Devuelve |
|---|---|
| `GroverMeanQAE(m_precision, shots, seed, ledger).estimate_mean(g)` | un `float`: la media estimada de `g ∈ [0,1]^M` leída de la medición (mediana si `shots > 1`). |
| `GroverMeanQAE.estimate_mean_general(f)` | un `float`: la media de un vector real cualquiera, tras el mapeo afín a [0, 1] y su inversa. |
| `GroverMeanQAE.gate_count(M)` | un `dict` con `logical`, `logical_qft_expanded`, `controlled_ry`, `elementary`, `two_qubit`, `depth_logical`, `depth_elementary`, `wires`. |
| `DeterministicMean(ledger).estimate_mean_general(f)` | un `float`: la media aritmética exacta. |
| `HybridCFDSolver(...).solve(n_steps)` | un `SolverResult` con los arreglos `x, rho, u, p, T, M`, `n_steps` y `elapsed_seconds`. |
| `NozzleFlow().exact_solution(x)` | tres arreglos de NumPy: Mach, densidad y temperatura exactas en `x`. |
| `CostLedger.to_dict()` | un `dict` con los conteos acumulados. |

Los guiones de `scripts/` escriben sus resultados como JSON en `datos/`, y las figuras se regeneran leyendo esos archivos. El modo histórico `euler_qae` (cuantizador valor a valor) ya no está en el paquete.

## Uso

```bash
python main.py                          # tobera, 31 celdas, 100 pasos, m = 5, 1 disparo
python main.py --pasos 400 --shots 25   # mediana de 25 disparos por estimación
python3 scripts/barrido_estimador_circuito.py --m 3 4 5 6 7 8 9 10 --merge
python3 scripts/generar_datos_validacion.py --modo circuito --shots 1 --seed 1
python3 scripts/generar_figuras_resultados.py
```

## Alcance

Este paquete cubre exclusivamente el problema del trabajo de grado: la
tobera convergente-divergente de Anderson con el estimador de media QAEA
sobre la cuadratura temporal del FVM.
