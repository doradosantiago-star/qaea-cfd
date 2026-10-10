#!/usr/bin/env python3
"""Reescala SOLO la coordenada y del blockMeshDict, dejando intacta la ley de areas.

Uso:  python3 reescalar.py <factor>     (ejecutar dentro del directorio del caso)

Cuidado: las lineas 'hex ... (150 40 1) simpleGrading (1 1 1)' tambien son
ternas entre parentesis. Si no se excluyen, el reescalado divide el numero de
celdas y la malla queda destruida sin dar ningun error.
"""
import re, sys, shutil

S = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
path = 'system/blockMeshDict'
shutil.copy(path, path + '.orig')

terna = re.compile(r'\(([\d.]+) ([\d.]+) ([\d.]+)\)')
escala = lambda m: "(%s %.6g %s)" % (m.group(1), float(m.group(2)) / S, m.group(3))

salida, tocadas = [], 0
for linea in open(path):
    if 'hex' in linea or 'simpleGrading' in linea:   # resolucion de malla: NO tocar
        salida.append(linea)
        continue
    nueva = terna.sub(escala, linea)
    if nueva != linea:
        tocadas += 1
    salida.append(nueva)

open(path, 'w').writelines(salida)
print("factor %g  Â·  %d lineas reescaladas  Â·  respaldo en %s.orig" % (S, tocadas, path))
