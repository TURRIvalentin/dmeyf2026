"""Paso 02: entrena 10 LightGBM y 10 Random Forest con 202103-202106 y predice 202108 (~25-30 min).

Cada semilla se guarda apenas termina, así una corrida cortada retoma donde quedó.
"""
import os
from time import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

import comun as cm

df = pd.read_parquet(cm.DATASET)
es_train = df['foto_mes'].isin(cm.MESES_TRAIN).to_numpy()
es_pred = (df['foto_mes'] == cm.MES_PRED).to_numpy()
# Los lags de 202108 usan 202107: ese mes no tiene target, pero sus variables sí se conocen.
X = cm.matriz(df, es_train | es_pred)
mes = df.loc[es_train | es_pred, 'foto_mes'].to_numpy()
X_train, X_pred = X[mes != cm.MES_PRED], X[mes == cm.MES_PRED]
del X
y = df.loc[es_train, 'clase_ternaria'].to_numpy()
# BAJA+1 como positivo duplica los casos de fuga para aprender; la ganancia se mide igual
# solo con BAJA+2.
y_lgbm = np.isin(y, cm.ENTREGA['lgbm_target_positivos']).astype(int)
nf = len(cm.FEATURES)

probas = dict(np.load(cm.PROBAS)) if os.path.exists(cm.PROBAS) else {}
probas['numero_de_cliente'] = df.loc[es_pred, 'numero_de_cliente'].to_numpy()
for s in cm.SEMILLAS:
    t0 = time()
    if f'lgbm_{s}' not in probas:
        m = LGBMClassifier(**cm.LGBM_PARAMS, random_state=s)
        m.fit(X_train, y_lgbm)
        probas[f'lgbm_{s}'] = m.predict_proba(X_pred)[:, 1]
        np.savez(cm.PROBAS, **probas)
    if f'rf_{s}' not in probas:
        # El RF ve solo las crudas y la clase ternaria: aporta al blend justamente por
        # mirar distinto que el LightGBM.
        m = RandomForestClassifier(**cm.RF_PARAMS, random_state=s)
        m.fit(X_train[:, :nf], y)
        col = int(np.where(m.classes_ == cm.EVENTO)[0][0])
        probas[f'rf_{s}'] = m.predict_proba(X_pred[:, :nf])[:, col]
        np.savez(cm.PROBAS, **probas)
    print(f'semilla {s}: {time() - t0:.0f}s', flush=True)
