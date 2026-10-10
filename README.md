# qaea-cfd

Código y caso CFD del trabajo de grado **Viabilidad del algoritmo de estimación de
amplitud cuántica aplicado al análisis de flujo compresible en toberas
convergente-divergentes** (Santiago Dorado Saldarriaga, Ingeniería Mecánica,
Universidad del Valle, 2026). Directores: Omar Calderón Losada y Guillermo Andrés
Jaramillo Pizarro.

El trabajo mide, en simulación, qué pasa cuando el algoritmo de estimación de amplitud
cuántica (QAEA) de Brassard, Høyer, Mosca y Tapp reemplaza la media temporal de un paso
de volúmenes finitos en la tobera de Anderson con choque normal. El repositorio tiene
dos partes independientes:

| Carpeta | Qué es | Qué necesita |
|---|---|---|
| [`qae_cfd/`](qae_cfd/) | Paquete de Python: solucionador de volúmenes finitos (MUSCL, Roe, HLLC), estimador QAEA sobre circuitos de PennyLane, registro de costo, solución exacta de la tobera, guiones y datos de todos los experimentos del documento. | Python 3.10 o superior, `numpy`, `matplotlib`, `pennylane`. |
| [`openfoam/`](openfoam/) | Caso `cdNozzle2D` de OpenFOAM 13 que resuelve la misma tobera en dos dimensiones, con los guiones de diagnóstico y las tablas de la campaña de validación (trece corridas). | OpenFOAM 13 (Foundation). Para las gráficas, `numpy`, `scipy`, `matplotlib`. |

Cada parte tiene su propio `README.md` con los detalles. Lo que sigue es lo mínimo para
arrancar.

## Parte cuántica en dos minutos

```bash
cd qae_cfd
pip install -r requirements.txt
python3 main.py                          # tobera, 31 celdas, 100 pasos, m = 5, 1 disparo
python3 scripts/verificacion_sod.py      # Roe y HLLC contra la solución exacta de Sod
```

`main.py` corre el solucionador híbrido: en cada paso de tiempo, en cada celda y
componente, el circuito de estimación de amplitud se construye, se ejecuta y se mide en
`lightning.qubit` para estimar la media temporal del lado derecho. No hay atajo
analítico: el único origen de cada estimación es la medición del registro de precisión.

Los guiones de `qae_cfd/scripts/` reproducen los experimentos del documento y escriben
sus resultados como JSON en `qae_cfd/datos/`. Las figuras se regeneran a partir de esos
JSON y caen en `Imagenes/`. La tabla siguiente dice qué guion produce qué.

| Resultado del documento | Guion | Datos |
|---|---|---|
| Barrido del estimador con circuito: error frente a llamadas al oráculo, fracción fuera de la cota (figura 5.1, tablas 5.2 y 5.3) | `scripts/barrido_estimador_circuito.py` | `datos/barrido/*.json`, `datos/barrido_estimador_circuito.json` |
| Prueba de concepto escalar v' = −v² (figura 5.2) | `scripts/prueba_concepto_edo_escalar.py` | `datos/edo_escalar.json`, `datos/edo_escalar/` |
| Corrida de la tobera con el circuito, cinco semillas, mediana de 25 disparos y barrido de m (figuras 5.3 y 5.4) | `scripts/generar_datos_validacion.py` | `datos/validacion_qa.json`, `datos/validacion_qa/` |
| Refinamiento de malla de la línea base (tabla 5.1) | `scripts/estudio_convergencia_malla.py` | `datos/convergencia_malla.json` |
| Distribución de la media mapeada en la tobera | `scripts/diagnostico_media_reescalada.py` | `datos/diagnostico_media_reescalada.json` |
| Compuertas y profundidad frente a m y M (tabla 5.2, figura 5.5) | `scripts/generar_figuras_resultados.py` | `datos/compuertas_vs_M.json` |
| Verificación cruzada con HLLC y contrapresión 0.670 | `scripts/verificaciones_extra.py` | `datos/verificaciones_extra.json` |
| Tubo de choque de Sod (figura 2.4) | `scripts/verificacion_sod.py` | se calcula al vuelo |
| Calibración del procesador `ibm_marrakesh` citada en la sección 5.4 | `scripts/calibracion_ibm.py` (requiere `qiskit-ibm-runtime` y una cuenta de IBM Quantum) | `datos/calibracion_ibm.json` |
| Figuras del circuito (4.1 y 4.2) | `scripts/figuras_circuito/*.tex` (quantikz) y `scripts/generar_figuras_compuertas.py`, que imprime la lista de compuertas de PennyLane y comprueba sus matrices | |

La corrida completa de la tobera con el circuito (34 800 estimaciones) toma unas 2.4 h
por semilla en un portátil. Los JSON entregados son los de las corridas del documento,
así que las figuras se regeneran sin volver a correr nada.

## Parte OpenFOAM en dos minutos

```bash
source /opt/openfoam13/etc/bashrc        # o el equivalente de la instalación
cp -r openfoam/cdNozzle2D $FOAM_RUN/cdNozzle2D && cd $FOAM_RUN/cdNozzle2D
./Allrun                                 # blockMesh, checkMesh, foamRun (unos 10 min)
../../herramientas/choque.sh .           # posición del choque en cada escritura
python3 ../../herramientas/regimen.py perfil .   # perfil contra la solución exacta
```

El caso es la tobera de Anderson escalada en la coordenada transversal (k = 20), con
contrapresión `fixedValue` de 67 875 Pa, solucionador `shockFluid`, Co máximo 0.4 y
0.15 s de tiempo final. Reproduce la posición exacta del choque dentro del 1 % en siete
contrapresiones. `openfoam/README.md` explica cada diccionario, las dos decisiones de
modelado que hicieron funcionar el caso, cómo generar las variantes de la campaña con
`herramientas/preparar.py` y de dónde sale cada tabla del capítulo 3.

## Licencia

MIT, ver `LICENSE`.
