"""Calibración de un procesador de IBM Quantum para la perspectiva del cap. 5.

Consulta las propiedades del procesador en una fecha dada (IBM conserva el
historial) y resume lo que la sección 5.4 necesita: error mediano de la
compuerta de dos cúbits, error mediano de lectura, T1 y T2 medianos y la
duración de las compuertas. Escribe datos/calibracion_ibm.json.

Uso: python3 scripts/calibracion_ibm.py [ibm_marrakesh] [AAAA-MM-DD ...]
Requiere una cuenta guardada con QiskitRuntimeService.save_account.
"""
import json, math, statistics as st, sys
from datetime import datetime, timezone
from pathlib import Path
from qiskit_ibm_runtime import QiskitRuntimeService

RAIZ = Path(__file__).resolve().parent.parent
N_2Q = 4430          # compuertas de dos cúbits por disparo, M = 4, m = 5 (sección 4.4)
PROF_ELEM = 7824     # capas elementales por disparo (sección 4.4)

def mediana_segura(valores):
    v = [x for x in valores if x is not None and not math.isnan(x)]
    return st.median(v) if v else None

def resumen(props, backend):
    err2, dur2, dur1, nombres = [], [], [], set()
    for g in props.gates:
        pe = {p.name: p.value for p in g.parameters}
        if len(g.qubits) == 2 and 'gate_error' in pe and pe['gate_error'] < 0.5:
            err2.append(pe['gate_error']); nombres.add(g.gate)
            if 'gate_length' in pe: dur2.append(pe['gate_length'])
        elif len(g.qubits) == 1 and g.gate in ('sx', 'x') and 'gate_length' in pe:
            dur1.append(pe['gate_length'])
    ro, t1, t2 = [], [], []
    for q in range(backend.num_qubits):
        for lista, nombre in ((ro, 'readout_error'), (t1, 'T1'), (t2, 'T2')):
            try: lista.append(props.qubit_property(q, nombre)[0])
            except Exception: pass
    e2 = mediana_segura(err2)
    d2 = mediana_segura(dur2); d1 = mediana_segura(dur1)
    # gate_length viene en ns (unidad de las propiedades de compuerta de IBM)
    duracion_circuito_ns = PROF_ELEM * (d2 or 0.0)   # cota: toda capa como una de dos cúbits
    return {
        'fecha_calibracion': str(props.last_update_date),
        'compuertas_2q': sorted(nombres), 'n_pares_2q': len(err2),
        'error_2q_mediano': e2, 'error_2q_min': min(err2), 'error_2q_max': max(err2),
        'error_lectura_mediano': mediana_segura(ro),
        'T1_mediano_us': mediana_segura(t1) * 1e6, 'T2_mediano_us': mediana_segura(t2) * 1e6,
        'duracion_2q_ns': d2, 'duracion_1q_ns': d1,
        'fidelidad_acumulada_4430_2q': (1 - e2) ** N_2Q,
        'compuertas_2q_para_fidelidad_0.5': math.log(0.5) / math.log(1 - e2),
        'duracion_circuito_estimada_us': duracion_circuito_ns * 1e-3,
    }

def main():
    args = sys.argv[1:]
    nombre = args[0] if args and not args[0][0].isdigit() else 'ibm_marrakesh'
    fechas = [a for a in args if a[0].isdigit()] or ['2026-07-30']
    svc = QiskitRuntimeService(channel='ibm_quantum_platform')
    b = svc.backend(nombre)
    salida = {'backend': nombre, 'procesador': getattr(b, 'processor_type', None), 'n_qubits': b.num_qubits, 'por_fecha': {}}
    for f in fechas + ['hoy']:
        props = b.properties() if f == 'hoy' else b.properties(datetime=datetime.fromisoformat(f).replace(hour=12, tzinfo=timezone.utc))
        salida['por_fecha'][f] = resumen(props, b)
        r = salida['por_fecha'][f]
        print(f"{f}: calibración {r['fecha_calibracion']}; 2Q {r['compuertas_2q']} error mediano {r['error_2q_mediano']:.2e} "
              f"({r['n_pares_2q']} pares); lectura {r['error_lectura_mediano']:.2e}; T1 {r['T1_mediano_us']:.0f} us; T2 {r['T2_mediano_us']:.0f} us; "
              f"dur 2Q {r['duracion_2q_ns']:.0f} ns; fidelidad 4430 2Q = {r['fidelidad_acumulada_4430_2q']:.1e}; "
              f"2Q tolerables (F=0.5) = {r['compuertas_2q_para_fidelidad_0.5']:.0f}; duración circuito ≈ {r['duracion_circuito_estimada_us']:.0f} us")
    out = RAIZ / 'datos' / 'calibracion_ibm.json'
    out.write_text(json.dumps(salida, indent=1, default=str))
    print('escrito', out)

if __name__ == '__main__':
    main()
