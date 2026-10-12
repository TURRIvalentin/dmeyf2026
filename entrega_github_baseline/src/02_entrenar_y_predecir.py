"""Paso 02: entrena 10 LightGBM con 202103-202106 y predice 202108 (~80 min).

Un modelo por cada uno de los 10 mejores juegos de hiperparámetros de la bayesiana; el modelo i
usa la semilla i. Cada modelo se guarda apenas termina, así una corrida cortada retoma donde quedó.
"""
import os
from time import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

import comun as cm

df = pd.read_parquet(cm.DATASET)
# ccajas_depositos está toda en cero en 202105 (dato roto): se pasa a NA antes de armar los lags,
# así tampoco contamina el lag de 202106 ni el lag2 de 202107.
for var, meses in cm.ENTREGA['variables_en_na'].items():
    df.loc[df['foto_mes'].isin(meses), var] = np.nan
es_train = df['foto_mes'].isin(cm.MESES_TRAIN).to_numpy()
es_pred = (df['foto_mes'] == cm.MES_PRED).to_numpy()
# Los lags de 202108 usan 202107 y 202106: esos meses no tienen target, pero sus variables sí se conocen.
X = cm.matriz(df, es_train | es_pred)
mes = df.loc[es_train | es_pred, 'foto_mes'].to_numpy()
X_train, X_pred = X[mes != cm.MES_PRED], X[mes == cm.MES_PRED]
del X
# Sin undersampling. BAJA+1 como positivo duplica los casos de fuga para aprender; la ganancia
# se mide igual solo con BAJA+2.
y = np.isin(df.loc[es_train, 'clase_ternaria'].to_numpy(), cm.ENTREGA['lgbm_target_positivos']).astype(int)
print(f'train {X_train.shape[0]:,} filas x {X_train.shape[1]} columnas', flush=True)

probas = dict(np.load(cm.PROBAS)) if os.path.exists(cm.PROBAS) else {}
probas['numero_de_cliente'] = df.loc[es_pred, 'numero_de_cliente'].to_numpy()
for i, (juego, s) in enumerate(zip(cm.JUEGOS, cm.SEMILLAS)):
    if f'm{i}' in probas:
        continue
    t0 = time()
    m = LGBMClassifier(**cm.LGBM_FIJOS, **juego, random_state=s)
    m.fit(X_train, y)
    probas[f'm{i}'] = m.predict_proba(X_pred)[:, 1]
    np.savez(cm.PROBAS, **probas)
    print(f'modelo {i} {juego} semilla {s}: {time() - t0:.0f}s', flush=True)
