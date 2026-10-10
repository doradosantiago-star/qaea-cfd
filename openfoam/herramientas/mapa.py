#!/usr/bin/env python3
"""Mapa de regimenes: compara la posicion del choque medida en el CFD contra
la prediccion analitica, para varias contrapresiones.

Uso:  python3 mapa.py <caso>:<pe> [<caso>:<pe> ...]
"""
import sys, os, csv, statistics as st
from scipy.optimize import brentq

G, GM, GP = 1.4, 0.4, 2.4
area = lambda x: 1.0 + 2.2 * (x - 1.5) ** 2

def A_Astar(M):
    return (1.0 / M) * ((2.0 / GP) * (1.0 + GM / 2.0 * M * M)) ** (GP / (2.0 * GM))

def mach_de_area(ar, sup):
    lo, hi = (1.0 + 1e-9, 30.0) if sup else (1e-9, 1.0 - 1e-9)
    return brentq(lambda M: A_Astar(M) - ar, lo, hi, xtol=1e-12)

p_p0 = lambda M: (1.0 + GM / 2.0 * M * M) ** (-G / GM)

def choque(M1):
    M2 = ((1.0 + GM / 2.0 * M1 * M1) / (G * M1 * M1 - GM / 2.0)) ** 0.5
    p0r = (((GP * M1 * M1) / (GM * M1 * M1 + 2.0)) ** (G / GM) *
           (GP / (2.0 * G * M1 * M1 - GM)) ** (1.0 / GM))
    return M2, p0r

def pe_rel(xs):
    M1 = mach_de_area(area(xs), True)
    _, p0r = choque(M1)
    Me = mach_de_area(area(3.0) * p0r, False)
    return p0r * p_p0(Me), M1

def predecir(p0, pe):
    xs = brentq(lambda x: pe_rel(x)[0] - pe / p0, 1.5001, 2.999, xtol=1e-10)
    return xs, pe_rel(xs)[1]

def medir(caso, n=6):
    d = os.path.join(caso, 'postProcessing', 'axisSampling')
    xs, m1 = [], []
    for t in sorted(os.listdir(d), key=float)[-n:]:
        f = os.path.join(d, t, 'axialLine.csv')
        if not os.path.exists(f):
            continue
        prev, caida, xch, mmax = None, 0.0, None, 0.0
        for r in csv.DictReader(open(f)):
            ma, x = float(r['Ma']), float(r['x'])
            mmax = max(mmax, ma)
            if prev is not None and prev - ma > caida:
                caida, xch = prev - ma, x
            prev = ma
        if xch:
            xs.append(xch); m1.append(mmax)
    sd = st.stdev(xs) if len(xs) > 1 else 0.0
    return st.mean(xs), sd, st.mean(m1)

P0 = 101325.0
print("%9s %8s %11s %18s %9s %11s %10s" %
      ("pe[Pa]", "pe/p0", "x_teorico", "x_CFD", "error", "Ma1_teor", "Ma1_CFD"))
print("-" * 84)
for arg in sys.argv[1:]:
    caso, pe = arg.rsplit(':', 1)
    pe = float(pe)
    try:
        xt, m1t = predecir(P0, pe)
        xc, sd, m1c = medir(caso)
        print("%9.0f %8.4f %11.4f %11.4f+/-%.4f %8.2f%% %11.4f %10.4f"
              % (pe, pe / P0, xt, xc, sd, abs(xc - xt) / xt * 100, m1t, m1c))
    except Exception as e:
        print("%9.0f  fallo: %s" % (pe, e))
