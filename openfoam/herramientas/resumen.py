#!/usr/bin/env python3
"""Resume una corrida: promedia las ultimas escrituras para separar la
solucion estacionaria de la vibracion residual del choque.

Uso: python3 resumen.py <caso> [n_ultimas]
"""
import sys, os, csv, statistics as st

caso = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
d = os.path.join(caso, 'postProcessing', 'axisSampling')
tiempos = sorted(os.listdir(d), key=float)[-N:]

xs, m1s, pes = [], [], []
for t in tiempos:
    f = os.path.join(d, t, 'axialLine.csv')
    if not os.path.exists(f):
        continue
    filas = list(csv.DictReader(open(f)))
    prev, caida, x_ch, ma_max = None, 0.0, None, 0.0
    for r in filas:
        ma, x = float(r['Ma']), float(r['x'])
        if ma > ma_max:
            ma_max = ma
        if prev is not None and prev - ma > caida:
            caida, x_ch = prev - ma, x
        prev = ma
        pe = float(r['p'])
    if x_ch is not None:
        xs.append(x_ch)
        m1s.append(ma_max)
        pes.append(pe)

sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
print("%-10s n=%d  x_choque = %.4f +/- %.4f m   M1 = %.4f +/- %.4f   pe = %.0f +/- %.0f Pa"
      % (os.path.basename(caso.rstrip('/')), len(xs),
         st.mean(xs), sd(xs), st.mean(m1s), sd(m1s), st.mean(pes), sd(pes)))
