#!/usr/bin/env python3
"""Solucion analitica completa de la tobera de Anderson CON choque normal.

A diferencia de teoria.py, este script resuelve las DOS ramas:
  - aguas arriba del choque, rama supersonica isentropica con A*
  - aguas abajo, rama subsonica con el A* corregido por la perdida de p0

y ademas predice la posicion del choque a partir de la contrapresion, que es
la comprobacion que cierra el lazo con el CFD.

Uso:
    python3 regimen.py prediccion <p0> <pe>
    python3 regimen.py perfil <caso> [tiempo]
"""
import sys, os, csv
from scipy.optimize import brentq

G = 1.4
GM = G - 1.0
GP = G + 1.0

def area(x):
    "Ley de areas de Anderson, adimensional, A* = A(1.5) = 1"
    return 1.0 + 2.2 * (x - 1.5) ** 2

def A_Astar(M):
    return (1.0 / M) * ((2.0 / GP) * (1.0 + GM / 2.0 * M * M)) ** (GP / (2.0 * GM))

def mach_de_area(ar, supersonico):
    lo, hi = (1.0 + 1e-9, 30.0) if supersonico else (1e-9, 1.0 - 1e-9)
    return brentq(lambda M: A_Astar(M) - ar, lo, hi, xtol=1e-12)

def p_sobre_p0(M):
    return (1.0 + GM / 2.0 * M * M) ** (-G / GM)

def choque_normal(M1):
    "Devuelve (M2, p02/p01) para un choque normal con Mach aguas arriba M1"
    M2 = ((1.0 + GM / 2.0 * M1 * M1) / (G * M1 * M1 - GM / 2.0)) ** 0.5
    p0r = (((GP * M1 * M1) / (GM * M1 * M1 + 2.0)) ** (G / GM) *
           (GP / (2.0 * G * M1 * M1 - GM)) ** (1.0 / GM))
    return M2, p0r

def pe_para_choque_en(xs, x_salida=3.0):
    "Contrapresion (relativa a p01) que coloca el choque normal en xs"
    M1 = mach_de_area(area(xs), True)
    M2, p0r = choque_normal(M1)
    ar_salida = area(x_salida) * p0r          # A_e / A*2
    Me = mach_de_area(ar_salida, False)
    return p0r * p_sobre_p0(Me), M1, M2, Me, p0r

# --------------------------------------------------------------- prediccion
if sys.argv[1] == 'prediccion':
    p0, pe = float(sys.argv[2]), float(sys.argv[3])
    objetivo = pe / p0
    f = lambda xs: pe_para_choque_en(xs)[0] - objetivo
    xs = brentq(f, 1.5001, 2.999, xtol=1e-10)
    _, M1, M2, Me, p0r = pe_para_choque_en(xs)
    print("p0 = %.0f Pa   pe = %.0f Pa   pe/p0 = %.4f" % (p0, pe, objetivo))
    print("-" * 52)
    print("  posicion del choque   x_s   = %.4f m" % xs)
    print("  Mach antes del choque M1    = %.4f" % M1)
    print("  Mach despues          M2    = %.4f" % M2)
    print("  perdida de remanso  p02/p01 = %.4f" % p0r)
    print("  Mach en la salida     Me    = %.4f" % Me)
    print("  A* aguas abajo              = %.4f  (era 1.0)" % (1.0 / p0r))
    sys.exit()

# ------------------------------------------------------------------ perfil
caso = sys.argv[2]
d = os.path.join(caso, 'postProcessing', 'axisSampling')
t = sys.argv[3] if len(sys.argv) > 3 else sorted(os.listdir(d), key=float)[-1]
filas = list(csv.DictReader(open(os.path.join(d, t, 'axialLine.csv'))))

xs_cfd, caida = None, 0.0
prev = None
for f in filas:
    ma = float(f['Ma'])
    if prev is not None and prev - ma > caida:
        caida, xs_cfd = prev - ma, float(f['x'])
    prev = ma

_, M1t, M2t, Met, p0r = pe_para_choque_en(xs_cfd)
print("caso %s   t = %s" % (caso, t))
print("choque detectado en el CFD: x = %.3f  (caida de Ma = %.3f)\n" % (xs_cfd, caida))
print("teoria con el choque puesto en esa misma x:")
print("   M1 = %.4f    M2 = %.4f    p02/p01 = %.4f    Me = %.4f\n" % (M1t, M2t, p0r, Met))

print("%7s %8s %10s %10s %9s   %s" % ("x[m]", "A/A*", "Ma_CFD", "Ma_teor", "error", "rama"))
for i, f in enumerate(filas):
    if i % 20 != 1:
        continue
    x, ma = float(f['x']), float(f['Ma'])
    if x < 1.5:
        mt, rama = mach_de_area(area(x), False), "subsonica"
    elif x < xs_cfd:
        mt, rama = mach_de_area(area(x), True), "supersonica"
    else:
        mt, rama = mach_de_area(area(x) * p0r, False), "tras el choque"
    print("%7.3f %8.3f %10.4f %10.4f %8.1f%%   %s"
          % (x, area(x), ma, mt, abs(ma - mt) / mt * 100, rama))
