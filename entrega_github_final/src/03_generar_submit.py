"""Paso 03: blend de rankings y archivo para el bot (segundos)."""
import os

import numpy as np
from scipy.stats import rankdata

import comun as cm

pr = np.load(cm.PROBAS)
faltan = [f'{m}_{s}' for s in cm.SEMILLAS for m in ('lgbm', 'rf') if f'{m}_{s}' not in pr]
assert not faltan, f'faltan modelos del paso 02: {faltan}'
ids = pr['numero_de_cliente']
n = len(ids)

# Promedio rankings y no probabilidades: el LightGBM (BAJA+1 y BAJA+2) y el RF (P(BAJA+2))
# estiman cosas distintas y sus escalas no son comparables.
p_lgbm = np.mean([pr[f'lgbm_{s}'] for s in cm.SEMILLAS], axis=0)
p_rf = np.mean([pr[f'rf_{s}'] for s in cm.SEMILLAS], axis=0)
score = rankdata(p_lgbm) / n
score = (score + rankdata(p_rf) / n) / 2
# Orden estable: los empates se resuelven siempre igual y el archivo es reproducible.
orden = np.argsort(-score, kind='stable')
elegidos = orden[:cm.N_ENVIOS]

os.makedirs(cm.SALIDA, exist_ok=True)
with open(cm.SUBMIT, 'w', encoding='ascii', newline='') as f:
    for v in ids[elegidos]:
        f.write(f'{int(v)}\n')
pred = np.zeros(n, dtype=int)
pred[elegidos] = 1
with open(cm.PREDICCION, 'w', encoding='ascii', newline='') as f:
    f.write('numero_de_cliente,Predicted\n')
    for v, p in zip(ids, pred):
        f.write(f'{int(v)},{int(p)}\n')
print(f'{cm.N_ENVIOS:,} de {n:,} clientes -> {cm.SUBMIT}')
