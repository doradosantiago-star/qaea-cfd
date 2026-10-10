#!/usr/bin/env python3
"""Genera las figuras de datos de la campana de verificacion en PDF y PNG.

Salida en openfoam/figuras/:
  convergencia_malla.pdf/.png
  mapa_regimenes.pdf/.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'figure.dpi': 150,
})

SAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figuras')

# ---------------------------------------------------------------- teoria
G, GM, GP = 1.4, 0.4, 2.4
area = lambda x: 1.0 + 2.2 * (x - 1.5) ** 2

def A_Astar(M):
    return (1.0 / M) * ((2.0 / GP) * (1.0 + GM / 2.0 * M * M)) ** (GP / (2.0 * GM))

def mach_de_area(ar, sup):
    lo, hi = (1.0 + 1e-9, 30.0) if sup else (1e-9, 1.0 - 1e-9)
    return brentq(lambda M: A_Astar(M) - ar, lo, hi, xtol=1e-12)

p_p0 = lambda M: (1.0 + GM / 2.0 * M * M) ** (-G / GM)

def salto(M1):
    p0r = (((GP * M1 * M1) / (GM * M1 * M1 + 2.0)) ** (G / GM) *
           (GP / (2.0 * G * M1 * M1 - GM)) ** (1.0 / GM))
    return p0r

def pe_rel_de_xs(xs):
    """Contrapresion relativa p_e/p_0 que coloca el choque en xs."""
    M1 = mach_de_area(area(xs), True)
    p0r = salto(M1)
    Me = mach_de_area(area(3.0) * p0r, False)
    return p0r * p_p0(Me)

X_TEORICO_REF = 2.1098   # para p_e = 67875 Pa

# ============================================================ FIGURA 1
# Convergencia de malla
celdas = np.array([1500, 6000, 24000])
x_ch   = np.array([2.0733, 2.0900, 2.1067])
sigma  = np.array([0.0163, 0.0089, 0.0052])
etiq   = ['75x20', '150x40', '300x80']

fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.axhline(X_TEORICO_REF, ls='--', lw=1.4, color='0.35', zorder=1)
ax.text(24000, X_TEORICO_REF + 0.0016, r'solución analítica  $x_s = 2.1098$ m',
        ha='right', va='bottom', fontsize=9.5, color='0.25')

ax.errorbar(celdas, x_ch, yerr=sigma, fmt='o-', color='#17457F',
            lw=1.6, ms=6, capsize=4, capthick=1.2, elinewidth=1.2, zorder=3,
            label='OpenFOAM 13 · promedio de las últimas 6 escrituras')

# etiquetas al costado, para no cruzarse con las barras de error
lados = [(14, -4, 'left'), (14, -4, 'left'), (-14, -4, 'right')]
for c, x, e, (dx, dy, ha) in zip(celdas, x_ch, etiq, lados):
    ax.annotate(f'{e}\n{x:.4f}', (c, x), textcoords='offset points',
                xytext=(dx, dy), ha=ha, va='center', fontsize=8.5, color='0.3')

ax.set_xscale('log')
ax.set_xlim(1000, 38000)
ax.set_ylim(2.045, 2.125)
ax.set_xlabel('Número de celdas')
ax.set_ylabel(r'Posición del choque  $x_s$  [m]')
ax.set_title('Independencia de malla', fontweight='bold', fontsize=12, pad=10)
ax.grid(True, which='both', ls=':', alpha=0.45)
ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{SAL}/convergencia_malla.{ext}', bbox_inches='tight')
plt.close(fig)
print('convergencia_malla.pdf / .png')

# ============================================================ FIGURA 2
# Mapa de regimenes: curva teorica continua + puntos CFD
pe_cfd  = np.array([40000, 60000, 67875, 72000, 80000, 88000, 95000], float)
xs_cfd  = np.array([2.5067, 2.2000, 2.0900, 2.0400, 1.9600, 1.8567, 1.7500])
sd_cfd  = np.array([0.0103, 0.0000, 0.0089, 0.0126, 0.0000, 0.0052, 0.0000])
P0 = 101325.0

xs_curva = np.linspace(1.55, 2.90, 400)
pe_curva = np.array([pe_rel_de_xs(x) for x in xs_curva])

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(pe_curva, xs_curva, '-', lw=1.8, color='0.35', zorder=2,
        label='Solución analítica (Rankine–Hugoniot)')
ax.errorbar(pe_cfd / P0, xs_cfd, yerr=np.maximum(sd_cfd, 1e-4),
            fmt='o', color='#17457F', ms=6.5, capsize=4, capthick=1.2,
            elinewidth=1.2, zorder=3, label='OpenFOAM 13')

ax.set_xlim(0.33, 0.99)
ax.set_ylim(1.6, 2.75)
ax.set_xlabel(r'Relación de contrapresión  $p_e / p_0$')
ax.set_ylabel(r'Posición del choque  $x_s$  [m]')
ax.set_title('Mapa de regímenes con choque interno',
             fontweight='bold', fontsize=12, pad=10)
ax.grid(True, ls=':', alpha=0.45)
ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

err = np.abs(xs_cfd - np.array([brentq(lambda x: pe_rel_de_xs(x) - p / P0,
                                       1.5001, 2.999) for p in pe_cfd]))
ax.text(0.355, 1.68, f'error máximo {100*np.max(err/xs_cfd):.2f} %  ·  7 regímenes',
        fontsize=9, color='0.3')

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{SAL}/mapa_regimenes.{ext}', bbox_inches='tight')
plt.close(fig)
print('mapa_regimenes.pdf / .png')
print('error maximo en el mapa: %.2f %%' % (100 * np.max(err / xs_cfd)))
