"""Paso 03: promedio de los 10 modelos y archivo para el bot (segundos)."""
import os

import numpy as np

import comun as cm

pr = np.load(cm.PROBAS)
faltan = [f'm{i}' for i in range(len(cm.JUEGOS)) if f'm{i}' not in pr]
assert not faltan, f'faltan modelos del paso 02: {faltan}'
ids = pr['numero_de_cliente']
n = len(ids)

# Los 10 modelos estiman lo mismo (P(BAJA+1 o BAJA+2)) con el mismo target: se promedian probabilidades.
p = np.mean([pr[f'm{i}'] for i in range(len(cm.JUEGOS))], axis=0)
# Orden estable: los empates se resuelven siempre igual y el archivo es reproducible.
orden = np.argsort(-p, kind='stable')
elegidos = orden[:cm.N_ENVIOS]

os.makedirs(cm.SALIDA, exist_ok=True)
with open(cm.SUBMIT, 'w', encoding='ascii', newline='') as f:
    for v in ids[elegidos]:
        f.write(f'{int(v)}\n')
pred = np.zeros(n, dtype=int)
pred[elegidos] = 1
with open(cm.PREDICCION, 'w', encoding='ascii', newline='') as f:
    f.write('numero_de_cliente,Predicted\n')
    for v, q in zip(ids, pred):
        f.write(f'{int(v)},{int(q)}\n')
print(f'{cm.N_ENVIOS:,} de {n:,} clientes -> {cm.SUBMIT}')
