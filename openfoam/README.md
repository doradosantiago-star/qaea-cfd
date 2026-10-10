# Caso `cdNozzle2D` en OpenFOAM 13

Tobera convergente-divergente de Anderson, `A(x) = 1 + 2.2 (x − 1.5)²` con `x` en
[0, 3] m, resuelta como canal plano bidimensional con las ecuaciones de Euler. La
referencia es la solución exacta cuasi-unidimensional con choque normal: para la
contrapresión del caso, el choque está en `x = 2.1098 m` con Mach `2.0880` justo antes.

Entorno con el que se corrió todo: OpenFOAM 13 (Foundation, paquete `20260624`),
Ubuntu 24.04 bajo WSL2.

## Contenido

| Ruta | Qué es |
|---|---|
| `cdNozzle2D/` | El caso listo para `./Allrun`. Solo entradas: `0/`, `constant/`, `system/`. La malla y los tiempos se generan al correr. |
| `herramientas/` | Guiones de diagnóstico, comparación con la teoría, variantes y figuras (ver tabla abajo). |
| `resultados/` | Tablas de la campaña de validación y perfiles axiales del instante final de cada corrida. |
| `figuras/` | Carpeta de salida de las gráficas; se entrega vacía. |

## Qué gobierna cada archivo del caso

| Archivo | Qué contiene | Se edita para |
|---|---|---|
| `0/p`, `0/T`, `0/U` | Campos iniciales y condiciones de frontera: `totalPressure` 101 325 Pa y `totalTemperature` 300 K en la entrada, `fixedValue` 67 875 Pa en la salida, `slip` en la pared, `symmetryPlane` en el eje. | Cambiar el punto de operación. La contrapresión mueve o elimina el choque. |
| `constant/thermophysicalProperties` | Aire como gas ideal caloríficamente perfecto (`hePsiThermo`, `perfectGas`, `hConst`), W = 28.96 g/mol, Cp = 1004.4 J/(kg·K), viscosidad nula. | Cambiar el fluido. |
| `constant/momentumTransport` | `laminar`: sin modelo de turbulencia. Lo exige OpenFOAM 13 aunque el caso sea no viscoso. | No se edita. |
| `system/blockMeshDict` | Semialtura de la pared `h(x) = A(x)/(2k)` con k = 20, siete puntos de control, malla de 150 × 40 × 1 hexaedros. | Refinar la malla o cambiar la esbeltez. |
| `system/controlDict` | `shockFluid`, `adjustTimeStep` con Co máximo 0.4, tiempo final 0.15 s, escritura cada 5 ms, muestreo axial (`axisSampling`), Mach (`MachNo`) y residuales. | Extender el tiempo final o la frecuencia de escritura. |
| `system/fvSchemes`, `system/fvSolution` | Euler en el tiempo, limitadores `vanLeer` en la reconstrucción, `diagonal` para la densidad y `smoothSolver` para el resto. | Cambiar limitadores o tolerancias. |
| `Allrun`, `Allclean` | `blockMesh`, comprobación de que la malla tiene exactamente 6000 celdas, `checkMesh`, `foamRun`. | No se editan. |

## Las dos decisiones de modelado

Las dos están documentadas en la cabecera del diccionario que las contiene.

1. **Escala transversal (`system/blockMeshDict`).** La ley de áreas de Anderson es
   adimensional. Leerla como semialtura en metros da paredes de 73°, el flujo se despega
   y forma un chorro: no hay solución cuasi-unidimensional contra la cual comparar. El
   caso divide solo la coordenada transversal por `k = 20`, lo que deja la pared por
   debajo de 10° y conserva los cocientes `A/A*`. Con `k = 10, 20, 40` el choque queda
   en el mismo sitio dentro de media celda (`resultados/tabla_uniformidad.csv`).
2. **Contrapresión `fixedValue` (`0/p`).** Una salida no reflectiva de tipo
   `waveTransmissive` no sirve: durante el arranque la salida es supersónica, no puede
   imponer la contrapresión y la solución converge a la expansión isentrópica sin choque
   (1445 Pa en la salida en vez de 67 875). Con la geometría esbelta, `fixedValue` da un
   choque quieto.

## Correr y diagnosticar

Copiar el caso fuera del repositorio antes de correr, porque `./Allclean` borra lo que
produce una corrida y `controlDict` arranca desde el último tiempo que encuentre.

```bash
cp -r cdNozzle2D $FOAM_RUN/cdNozzle2D && cd $FOAM_RUN/cdNozzle2D
./Allrun
```

| Pregunta | Comando | Qué devuelve |
|---|---|---|
| ¿Está quieto el choque? | `herramientas/choque.sh <caso>` | Posición del choque en cada escritura. Las últimas deben repetir el mismo valor. |
| ¿Se conserva la masa? | `herramientas/balance.sh <caso>` | Balance entrada/salida y un veredicto de estacionariedad. |
| ¿El flujo es cuasi-unidimensional? | `herramientas/transversal.sh <caso> 2.0 0.03875` | Variación del Mach a través de la sección en `x = 2.0 m`. |
| ¿Coincide con la teoría? | `python3 herramientas/regimen.py perfil <caso>` | Perfil axial contra la solución exacta con las dos ramas y el salto de Rankine-Hugoniot. |
| ¿Dónde predice la teoría el choque? | `python3 herramientas/regimen.py prediccion 101325 72000` | Posición y Mach antes del choque para esa contrapresión, sin CFD. |
| Promedio de las últimas escrituras | `python3 herramientas/resumen.py <caso> 6` | Posición media del choque y desviación. Así se reportan todas las corridas. |

Dependencias de los guiones de Python: `numpy`, `scipy`, `matplotlib`;
`figura_regimenes.py` usa además `Pillow`, y `render.py` corre con `pvbatch` de ParaView.

## La campaña de validación

Cada fila de `resultados/tabla_casos.csv` es una corrida, y sus cinco primeras columnas
son los argumentos de `herramientas/preparar.py`, que fija escala, malla, contrapresión
y tiempo final sobre una copia limpia del caso:

```bash
cp -r cdNozzle2D $FOAM_RUN/m300 && cd $FOAM_RUN/m300
python3 <ruta>/herramientas/preparar.py 20 300 80 67875 0.15
blockMesh && foamRun > log.foamRun 2>&1
```

| Corridas | Qué prueban | Tabla |
|---|---|---|
| `m075`, `geo20p`, `m300` | Mallas de 75 × 20, 150 × 40 y 300 × 80: independencia de malla por el índice de convergencia (GCI) de Celik et al. (2008), con `herramientas/gci.py`. | `tabla_casos.csv` |
| `s10`, `geo20p`, `s40` | Esbeltez k = 10, 20 y 40: el resultado no depende de k. | `tabla_uniformidad.csv` |
| `pe40` a `pe95` | Siete contrapresiones entre 40 y 95 kPa: todas dentro del 1 % de la posición exacta. | `tabla_regimenes.txt` |
| `long` | 0.30 s de tiempo final: el choque no se mueve al duplicar el horizonte. | `tabla_casos.csv` |
| `cn5co1`, `ddt2`, `cn9`, `cn5` | Esquemas temporales de segundo orden: divergen con Co 0.4; Crank-Nicolson 0.5 con Co 0.1 da la misma posición a seis veces el costo. | `tabla_esquemas.csv` |

`herramientas/gci.py` recalcula el GCI a partir del `postProcessing` de las tres
corridas de malla, así que necesita haberlas corrido. Las gráficas se obtienen con
`herramientas/graficas.py` (convergencia y mapa de regímenes),
`herramientas/figura_capitulo3.py` (perfil axial y siete contrapresiones, desde
`resultados/`) y `herramientas/figura_gci_tesis.py`; los campos de Mach se renderizan con
`herramientas/render.py` y se apilan con `herramientas/figura_regimenes.py`.

## Trampas conocidas

- `blockMesh` y `checkMesh` aceptan sin error un `blockMeshDict` cuyo número de celdas
  cambió por un reescalado mal hecho. Por eso `Allrun` aborta si no hay 6000 celdas, y
  `reescalar.py` y `preparar.py` excluyen la línea `hex ... (nx ny 1)` del reescalado.
- Los residuales no sirven como criterio de estacionariedad: la actualización explícita
  basada en densidad los reporta nulos. La estacionariedad se juzga con `choque.sh`.
- El balance de masa instantáneo sigue la vibración del choque y cambia de signo entre
  escrituras. Se reporta, pero no decide nada.
