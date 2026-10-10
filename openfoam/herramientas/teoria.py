#!/usr/bin/env python3
"""Compara el perfil axial del CFD contra la solucion isentropica cuasi-1D.
Uso:  python3 teoria.py <caso> [tiempo]"""
import sys, os, csv
from scipy.optimize import brentq

G = 1.4
area = lambda x: 1.0 + 2.2*(x-1.5)**2                      # ley de Anderson
def A_Astar(M):
    return (1/M)*((2/(G+1))*(1+(G-1)/2*M*M))**((G+1)/(2*(G-1)))

case = sys.argv[1]
d = os.path.join(case, 'postProcessing', 'axisSampling')
t = sys.argv[2] if len(sys.argv) > 2 else sorted(os.listdir(d), key=float)[-1]
rows = list(csv.DictReader(open(os.path.join(d, t, 'axialLine.csv'))))

print(f"caso {case}   t = {t}\n")
print(f"{'x[m]':>7} {'A/A*':>7} {'Ma_CFD':>9} {'Ma_teor':>9} {'error':>8}")
for i, r in enumerate(rows):
    if i % 25 != 1:
        continue
    x, ma = float(r['x']), float(r['Ma'])
    ar = area(x)
    try:
        lo, hi = (1e-6, 1.0) if x <= 1.5 else (1.0, 25.0)
        mt = brentq(lambda M: A_Astar(M)-ar, lo, hi)
    except ValueError:
        continue
    print(f"{x:7.3f} {ar:7.3f} {ma:9.4f} {mt:9.4f} {abs(ma-mt)/mt*100:7.1f}%")
print("\nNota: la comparacion solo vale aguas arriba del choque;")
print("      despues del choque la rama isentropica ya no aplica.")
