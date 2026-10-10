#!/usr/bin/env python3
"""Prepara una variante del caso cdNozzle2D.

Uso (dentro del directorio del caso):
    python3 preparar.py <escala_y> <nx> <ny> <p_salida> <endTime>

- escala_y : divide SOLO la coordenada y del blockMeshDict (la ley A(x) no cambia)
- nx, ny   : resolucion de malla; se reescribe la linea 'hex ... (nx ny 1)'
- p_salida : contrapresion fixedValue en Pa
- endTime  : tiempo final en s
"""
import re, sys

S      = float(sys.argv[1])
NX     = int(sys.argv[2])
NY     = int(sys.argv[3])
PE     = sys.argv[4]
ENDT   = sys.argv[5]

# ---- blockMeshDict: escala transversal + resolucion ----------------------
bmd = 'system/blockMeshDict'
terna = re.compile(r'\(([\d.]+) ([\d.]+) ([\d.]+)\)')
escala = lambda m: "(%s %.8g %s)" % (m.group(1), float(m.group(2)) / S, m.group(3))

salida = []
for linea in open(bmd):
    if 'hex' in linea or 'simpleGrading' in linea:
        linea = re.sub(r'hex \(([\d ]+)\) \([\d ]+\)',
                       lambda m: "hex (%s) (%d %d 1)" % (m.group(1), NX, NY), linea)
        salida.append(linea)
        continue
    salida.append(terna.sub(escala, linea))
open(bmd, 'w').writelines(salida)

# ---- controlDict: linea de muestreo + endTime ----------------------------
cd = 'system/controlDict'
t = open(cd).read()

# el muestreo axial debe quedar cerca del eje sea cual sea la escala
y_muestreo = 0.5 / S * 0.02          # 2 % de la semialtura de garganta
t = re.sub(r'start\s+\(0\.0 [\d.eE+-]+ 0\.05\);',
           'start   (0.0 %.6g 0.05);' % y_muestreo, t)
t = re.sub(r'end\s+\(3\.0 [\d.eE+-]+ 0\.05\);',
           'end     (3.0 %.6g 0.05);' % y_muestreo, t)
t = re.sub(r'^endTime\s+[\d.eE+-]+;', 'endTime         %s;' % ENDT, t, flags=re.M)
open(cd, 'w').write(t)

# ---- 0/p: contrapresion --------------------------------------------------
p = open('0/p').read()
p = re.sub(r'(outlet\s*\{[^}]*?value\s+uniform\s+)[\d.eE+-]+(;)',
           lambda m: m.group(1) + PE + m.group(2), p, flags=re.S)
open('0/p', 'w').write(p)

print("escala=%g  malla=%dx%d  p_salida=%s  endTime=%s  y_muestreo=%.6g"
      % (S, NX, NY, PE, ENDT, y_muestreo))
