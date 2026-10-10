#!/usr/bin/env python3
"""Apila los campos de Mach de tres contrapresiones en una sola figura para
el Capitulo 3, recortando el margen blanco de cada render de pvbatch.

Uso (desde openfoam/):
    python3 herramientas/figura_regimenes.py <carpeta_de_salida>
"""
import sys
from pathlib import Path

from PIL import Image, ImageChops

RAIZ = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "figuras"
CASOS = ["mach_regimen_pe40000.png", "mach_regimen_pe72000.png", "mach_regimen_pe95000.png"]
MARGEN, SEPARACION = 12, 30


def recortar(img):
    rgb = img.convert("RGB")
    fondo = Image.new("RGB", rgb.size, (255, 255, 255))
    caja = ImageChops.difference(rgb, fondo).getbbox()
    x0, y0, x1, y1 = caja
    return rgb.crop((max(0, x0 - MARGEN), max(0, y0 - MARGEN),
                     min(rgb.width, x1 + MARGEN), min(rgb.height, y1 + MARGEN)))


paneles = [recortar(Image.open(RAIZ / "figuras" / c)) for c in CASOS]
ancho = max(p.width for p in paneles)
alto = sum(p.height for p in paneles) + SEPARACION * (len(paneles) - 1)
lienzo = Image.new("RGB", (ancho, alto), (255, 255, 255))
y = 0
for p in paneles:
    lienzo.paste(p, ((ancho - p.width) // 2, y))
    y += p.height + SEPARACION

OUT.mkdir(parents=True, exist_ok=True)
destino = OUT / "fig_openfoam_regimenes_es.png"
lienzo.save(destino)
print("escrito", destino, lienzo.size)
