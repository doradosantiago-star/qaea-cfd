#!/usr/bin/env python3
"""Independencia de malla por GCI (Celik et al. 2008, ASME J. Fluids Eng. 130:078001).

Reprocesa las tres mallas ya corridas (m075, geo20p, m300) con una medida del
choque que NO depende del espaciado de muestreo: el cruce lineal de la presion
media entre la meseta antes y despues del choque.

Uso: python3 gci.py [n_ultimas] [--figura]   (21 por defecto; con 6 la x del choque da p=0.3, no es robusto)
"""
import sys, os, csv, math, statistics as st

R = os.path.expanduser('~/OpenFOAM/santiago-13/run')
MALLAS = [('m300', 300, 80), ('geo20p', 150, 40), ('m075', 75, 20)]  # fina -> gruesa
FIGURA = '--figura' in sys.argv
ARGS = [a for a in sys.argv[1:] if a != '--figura']
N = int(ARGS[0]) if ARGS else 21   # t = 0.05 -> 0.15 s: todo lo posterior al transitorio
X_EST = 1.8   # estacion supersonica suave, aguas arriba del choque

def leer(caso, t):
    f = os.path.join(R, caso, 'postProcessing', 'axisSampling', t, 'axialLine.csv')
    filas = list(csv.DictReader(open(f)))
    return [(float(r['x']), float(r['p']), float(r['Ma'])) for r in filas]

def interp(xs, ys, x0):
    for i in range(1, len(xs)):
        if xs[i - 1] <= x0 <= xs[i]:
            w = (x0 - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + w * (ys[i] - ys[i - 1])

def medidas(caso, t):
    d = leer(caso, t)
    x = [r[0] for r in d]; p = [r[1] for r in d]; ma = [r[2] for r in d]
    i1 = max(range(len(ma)), key=lambda i: ma[i])      # pico de Mach = pie del choque
    # meseta despues del choque: maximo de p en los 0.15 m siguientes al pico
    j = [k for k in range(i1, len(x)) if x[k] <= x[i1] + 0.15]
    i2 = max(j, key=lambda k: p[k])
    pm = 0.5 * (p[i1] + p[i2])
    xs = None
    for k in range(i1 + 1, i2 + 1):
        if p[k - 1] <= pm <= p[k]:
            xs = x[k - 1] + (pm - p[k - 1]) / (p[k] - p[k - 1]) * (x[k] - x[k - 1])
            break
    return {'x_choque': xs, 'Ma1': ma[i1], 'Ma_x18': interp(x, ma, X_EST),
            'p2_p1': p[i2] / p[i1]}

res = {}
for caso, nx, ny in MALLAS:
    d = os.path.join(R, caso, 'postProcessing', 'axisSampling')
    ts = sorted(os.listdir(d), key=float)[-N:]
    vals = [medidas(caso, t) for t in ts]
    res[caso] = {k: (st.mean(v[k] for v in vals), st.stdev(v[k] for v in vals)) for k in vals[0]}
    res[caso]['h'] = (3.0 / nx, 0)

print('%-8s %8s' % ('malla', 'h[m]') + ''.join('%22s' % k for k in ('x_choque', 'Ma1', 'Ma_x18', 'p2_p1')))
for caso, nx, ny in MALLAS:
    r = res[caso]
    print('%-8s %8.4f' % (caso, r['h'][0]) + ' sem_x=%.4f' % (r['x_choque'][1]/math.sqrt(N)) +
          ''.join('%13.4f +/- %.4f' % r[k] for k in ('x_choque', 'Ma1', 'Ma_x18', 'p2_p1')))

out = {}
print('\nGCI (r = 2, Fs = 1.25)')
for k in ('x_choque', 'Ma1', 'Ma_x18', 'p2_p1'):
    f1, f2, f3 = (res[c][k][0] for c, _, _ in MALLAS)
    ruido = max(res[c][k][1] for c, _, _ in MALLAS)
    e21, e32 = f2 - f1, f3 - f2
    r = 2.0
    linea = '%-9s e21=%+.4f e32=%+.4f ruido=%.4f ' % (k, e21, e32, ruido)
    if e21 * e32 <= 0:
        print(linea + '-> convergencia OSCILATORIA, GCI no aplica'); continue
    p = abs(math.log(abs(e32 / e21))) / math.log(r)
    fext = f1 + (f1 - f2) / (r ** p - 1)
    gci = 1.25 * abs((f1 - f2) / f1) / (r ** p - 1) * 100
    gci32 = 1.25 * abs((f2 - f3) / f2) / (r ** p - 1) * 100
    asin = gci32 / (r ** p * gci)
    out[k] = (p, fext, gci)
    print(linea + 'p=%.2f f_ext=%.4f GCI_fina=%.2f%% GCI_media=%.2f%% asintotico=%.3f'
          % (p, fext, gci, gci32, asin))

# ------------------------------------------------------------------ figura
if FIGURA:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                         'axes.linewidth': 0.8, 'figure.dpi': 150})
    AZUL, TINTA, GRIS = '#2a78d6', '#0b0b0b', '#52514e'
    TEORIA = {'x_choque': 2.1098, 'Ma1': 2.0880}
    PANELES = [('x_choque', 'Posición del choque $x_s$ [m]'),
               ('Ma1', 'Mach previo al choque $Ma_1$')]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (k, titulo) in zip(axs, PANELES):
        p, fext, gci = out[k]
        hs = [res[c]['h'][0] for c, _, _ in MALLAS]
        fs = [res[c][k][0] for c, _, _ in MALLAS]
        sd = [res[c][k][1] for c, _, _ in MALLAS]
        # ajuste de Richardson f(h) = f_ext + C h^p, pasa por la malla fina
        C = (fs[0] - fext) / hs[0] ** p
        hh = [i * 0.0425 / 200 for i in range(201)]
        ax.plot(hh, [fext + C * h ** p for h in hh], color=AZUL, lw=1, alpha=0.5)
        ax.errorbar(hs, fs, yerr=sd, fmt='o', ms=7, color=AZUL, ecolor=AZUL,
                    elinewidth=1.2, capsize=3, zorder=3)
        # el GCI es la banda de incertidumbre de la malla FINA (Celik 2008), no del
        # extrapolado: se dibuja como corchete junto al punto 300x80
        banda = abs(fs[0]) * gci / 100
        xb = hs[0] - 0.0028
        ax.plot([xb + 0.0006, xb, xb, xb + 0.0006],
                [fs[0] - banda, fs[0] - banda, fs[0] + banda, fs[0] + banda],
                color=TINTA, lw=1, zorder=3)
        ax.annotate('GCI ±%.2f %%' % gci, (xb, fs[0] + banda), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom',
                    color=TINTA, fontsize=9)
        ax.plot([0], [fext], 'o', ms=7, mfc='white', mec=AZUL, mew=1.5, zorder=3)
        ax.axhline(TEORIA[k], color=GRIS, lw=1, ls='--')
        ax.annotate('analítico', (0.0425, TEORIA[k]), xytext=(0, -4),
                    textcoords='offset points', ha='right', va='top',
                    color=GRIS, fontsize=9)
        ax.annotate('extrapolado (Δx → 0) = %.4f' % fext, (0, fext),
                    xytext=(0.05, 0.12), textcoords='axes fraction', ha='left',
                    va='bottom', color=TINTA, fontsize=9,
                    arrowprops=dict(arrowstyle='-', color=GRIS, lw=0.8, shrinkB=6))
        for (c, nx, ny), h, f in zip(MALLAS, hs, fs):
            ax.annotate('%d×%d' % (nx, ny), (h, f), xytext=(8, -2),
                        textcoords='offset points', ha='left', va='top',
                        color=GRIS, fontsize=9)
        ax.set_title('%s   (p = %.2f)' % (titulo, p), fontsize=11, color=TINTA)
        ax.set_xlabel(r'Tamaño de celda $\Delta x$ [m]')
        ax.set_xlim(-0.003, 0.046)
        ax.set_xticks([0, 0.01, 0.02, 0.04])
        ax.grid(True, color='#e6e5e0', lw=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    sal = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figuras', 'convergencia_gci')
    sal = os.environ.get('GCI_SALIDA', sal)
    for ext in ('pdf', 'png'):
        fig.savefig(sal + '.' + ext, bbox_inches='tight')
    print('figura:', sal + '.{pdf,png}')
