"""Figuras del Capítulo 1 en español: árbol de problemas, árbol de objetivos y
diagrama de flujo de la metodología.

Reemplazan a `Problem Tree.png`, `Árbol de objetivos.drawio.png` y
`Flow chart of the methodology.png`, que estaban en inglés. El diagrama de
metodología describe lo que se hizo, sin el lazo de "ajuste iterativo de
parámetros" que la versión anterior dibujaba y que nunca ocurrió.

Uso:  python3 generar_figuras_cap1.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SALIDA = Path(__file__).resolve().parents[2] / "Imagenes"
TINTA = "#222222"
GRIS = "#F1F1F1"
plt.rcParams.update({"font.family": "serif", "font.size": 9.5})


def caja(ax, x, y, w, h, texto, ancho=34, negrita=False, relleno="white", lw=1.0, fs=9.5):
    """Dibuja una caja redondeada centrada en (x, y) con texto envuelto."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=relleno, ec=TINTA, lw=lw))
    ax.text(x, y, "\n".join(textwrap.wrap(texto, ancho)), ha="center", va="center",
            color=TINTA, fontsize=fs, fontweight="bold" if negrita else "normal",
            linespacing=1.25)


def flecha(ax, p0, p1, estilo="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=estilo, mutation_scale=12,
                                 color=TINTA, lw=1.0, shrinkA=0, shrinkB=0))


def arbol(nombre, centro, abajo, arriba, titulo_abajo, titulo_arriba, ancho_centro=40):
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    cx, cy = 5.25, 3.6
    caja(ax, cx, cy, 4.4, 1.9, centro, ancho=ancho_centro, negrita=True, relleno=GRIS, lw=1.6, fs=9)

    xs = [1.85, 5.25, 8.65]
    for x, t in zip(xs, abajo):
        caja(ax, x, 1.05, 3.2, 1.55, t, ancho=30, fs=9)
        flecha(ax, (x, 1.05 + 0.775), (x, cy - 0.95) if x == cx else (cx + (x - cx) * 0.45, cy - 0.95))
    for x, t in zip(xs, arriba):
        caja(ax, x, 6.15, 3.2, 1.55, t, ancho=30, fs=9)
        flecha(ax, (x, cy + 0.95) if x == cx else (cx + (x - cx) * 0.45, cy + 0.95), (x, 6.15 - 0.775))

    ax.text(0.25, 2.05, titulo_abajo, fontsize=9, style="italic", color=TINTA)
    ax.text(0.25, 5.15, titulo_arriba, fontsize=9, style="italic", color=TINTA)
    fig.savefig(SALIDA / nombre, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def arbol_problemas():
    arbol(
        "fig_arbol_problemas_es.png",
        centro=("¿Es viable el uso de algoritmos de estimación de amplitud cuántica, en "
                "exactitud y desempeño, para resolver el flujo compresible en una tobera "
                "convergente-divergente, comparado con el método de volúmenes finitos?"),
        abajo=[
            "No se ha caracterizado qué ocurre al reemplazar la media temporal de un paso "
            "de volúmenes finitos por una sola estimación de amplitud sobre todo el campo.",
            "No hay comparaciones controladas contra una línea base de volúmenes finitos "
            "validada, con el costo contado en llamadas al oráculo.",
            "No hay un criterio para elegir el número de cúbits de precisión frente al "
            "error que ya impone la discretización espacial.",
        ],
        arriba=[
            "Sin una medición propia, la ventaja cuadrática del algoritmo se cita pero no "
            "se verifica en un problema de mecánica de fluidos.",
            "El tiempo de reloj de un circuito simulado se toma por costo del algoritmo, "
            "y las comparaciones publicadas no son reproducibles.",
            "La decisión de cuántos cúbits usar queda sin fundamento en la exactitud del "
            "campo calculado.",
        ],
        titulo_abajo="Causas",
        titulo_arriba="Consecuencias",
    )


def arbol_objetivos():
    arbol(
        "fig_arbol_objetivos_es.png",
        centro=("Evaluar la exactitud y el desempeño del algoritmo QAEA en la solución de un "
                "problema de flujo compresible en una tobera convergente-divergente, "
                "comparándolo contra el método de volúmenes finitos tradicional."),
        abajo=[
            "Definir las ecuaciones gobernantes del flujo compresible en la tobera, "
            "con y sin onda de choque.",
            "Implementar el algoritmo QAEA en un código computacional para resolver esas "
            "ecuaciones en el problema bajo estudio.",
            "Comparar los resultados del QAEA contra la solución obtenida con el método "
            "de volúmenes finitos.",
        ],
        arriba=[
            "Una solución exacta de referencia y un solucionador de volúmenes finitos "
            "validado contra ella.",
            "Un estimador de medias temporales especificado hasta la compuerta e "
            "implementado dentro del paso de tiempo.",
            "Una comparación de exactitud y costo en llamadas al oráculo, reproducible "
            "con los guiones del paquete.",
        ],
        titulo_abajo="Objetivos específicos",
        titulo_arriba="Productos",
    )


def flujo_metodologia():
    fig, ax = plt.subplots(figsize=(7.4, 11.2))
    ax.set_xlim(0, 7.4)
    ax.set_ylim(0, 11.2)
    ax.axis("off")
    cx = 3.7

    pasos = [
        ("Objetivo general", "Evaluar exactitud y desempeño del QAEA frente al método de "
         "volúmenes finitos en la tobera convergente-divergente.", 1.1),
        ("Fase 1. Ecuaciones y referencia (Capítulo 2)",
         "Ecuaciones de Euler cuasi-unidimensionales con término fuente; solución exacta "
         "de cuatro zonas con choque normal; método de volúmenes finitos en seis etapas; "
         "solucionadores de Riemann verificados en el tubo de Sod.", 1.6),
        ("Fase 2a. Caso OpenFOAM (Capítulo 3)",
         "Caso cdNozzle2D: esbeltez de la tobera, condiciones de frontera, malla y esquema. "
         "Validación contra la solución exacta en siete contrapresiones.", 1.35),
        ("Fase 2b. Solucionador propio y QAEA (Capítulo 4)",
         "Solucionador de volúmenes finitos en Python; estimador de amplitud con PennyLane "
         "que reemplaza la media temporal; registro de costo en llamadas al oráculo; "
         "presupuesto de error que fija m = 5.", 1.6),
        ("Fase 3. Medición (Capítulo 5)",
         "Validación de la línea base clásica y su refinamiento de malla; barrido del "
         "estimador m = 3..10 contra la cota de Brassard et al.; corrida completa con "
         "m = 5; comparación a exactitud equiparada.", 1.6),
        ("Conclusiones (Capítulo 6)",
         "Respuesta a la pregunta de investigación y condiciones bajo las que vale.", 1.05),
    ]
    alturas = [p[2] for p in pasos]
    sep = 0.38
    total = sum(alturas) + sep * (len(pasos) - 1)
    y = 11.2 - (11.2 - total) / 2 - alturas[0] / 2
    centros = []
    for (titulo, cuerpo, h) in pasos:
        ax.add_patch(FancyBboxPatch((cx - 3.1, y - h / 2), 6.2, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=GRIS if titulo.startswith(("Objetivo", "Conclu")) else "white",
                                    ec=TINTA, lw=1.0))
        ax.text(cx, y + h / 2 - 0.3, titulo, ha="center", va="center", fontweight="bold", fontsize=9.5)
        ax.text(cx, y - 0.2, "\n".join(textwrap.wrap(cuerpo, 64)), ha="center", va="center",
                fontsize=9, linespacing=1.25)
        centros.append((y, h))
        y -= h + sep
    for (y0, h0), (y1, h1) in zip(centros[:-1], centros[1:]):
        flecha(ax, (cx, y0 - h0 / 2), (cx, y1 + h1 / 2))
    fig.savefig(SALIDA / "fig_flujo_metodologia_es.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    SALIDA.mkdir(exist_ok=True)
    arbol_problemas()
    arbol_objetivos()
    flujo_metodologia()
    for n in ("fig_arbol_problemas_es.png", "fig_arbol_objetivos_es.png", "fig_flujo_metodologia_es.png"):
        print("OK", SALIDA / n)
