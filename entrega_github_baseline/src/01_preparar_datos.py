"""Paso 01: del crudo al dataset de entrenamiento (~15 s).

Calcula clase_ternaria con la definición de la notebook z101 de la cátedra, verifica los
conteos por mes y reordena las filas como en el entrenamiento original.
"""
import os
import sys

import numpy as np
import pandas as pd

import comun as cm

crudo = cm.ruta_crudo()
if not os.path.exists(crudo):
    sys.exit(f'No encuentro el crudo en {crudo} (ver README).')

# float32: es la precisión con la que se entrenaron los modelos de la entrega.
df = pd.read_csv(crudo, dtype={c: 'float32' for c in cm.FEATURES})
# Algunos IDs vienen en notación científica y pandas los lee como float.
ids = df['numero_de_cliente']
assert (ids == ids.astype('int64')).all(), 'numero_de_cliente no es entero exacto'
df['numero_de_cliente'] = ids.astype('int64')
assert not df.duplicated(['numero_de_cliente', 'foto_mes']).any()

# clase_ternaria: BAJA+1 si el cliente no está en m+1, BAJA+2 si está en m+1 pero no en m+2.
meses = sorted(df['foto_mes'].unique())
mi = df['foto_mes'].map({m: i for i, m in enumerate(meses)}).tolist()
cli = df['numero_de_cliente'].tolist()
presentes = set(zip(cli, mi))
en_m1 = np.fromiter(((c, m + 1) in presentes for c, m in zip(cli, mi)), bool, len(df))
en_m2 = np.fromiter(((c, m + 2) in presentes for c, m in zip(cli, mi)), bool, len(df))
clase = np.where(~en_m1, 'BAJA+1', np.where(~en_m2, 'BAJA+2', 'CONTINUA')).astype(object)
# Sin m+1 (último mes) no hay clase; sin m+2 (anteúltimo) solo se puede afirmar BAJA+1.
mi = np.array(mi)
clase[mi == len(meses) - 1] = None
clase[(mi == len(meses) - 2) & en_m1] = None
df['clase_ternaria'] = clase

# Si el crudo no es el de la cátedra, los conteos no coinciden y conviene cortar acá.
conteo = df.groupby('foto_mes')['clase_ternaria'].value_counts(dropna=False).unstack(fill_value=0)
conteo.columns = [str(c) if pd.notna(c) else 'nan' for c in conteo.columns]
for mes, esperado in cm.ENTREGA['conteo_clase_ternaria_esperado'].items():
    for clase_, n in esperado.items():
        obtenido = int(conteo.loc[int(mes), clase_]) if clase_ in conteo.columns else 0
        assert obtenido == n, f'clase_ternaria {mes} {clase_}: {obtenido} != {n}'

# El bagging del LightGBM y el bootstrap del RF eligen filas por posición: sin este orden
# (el del dataset con el que entrené) los modelos y el archivo final cambian.
orden = np.load(os.path.join(cm.CONFIG, 'orden_filas.npz'))['orden']
assert len(orden) == len(df) and len(np.unique(orden)) == len(df), 'orden_filas no corresponde a este crudo'
df = df.iloc[orden].reset_index(drop=True)

os.makedirs(cm.TRABAJO, exist_ok=True)
df[['numero_de_cliente', 'foto_mes', 'clase_ternaria'] + cm.FEATURES].to_parquet(cm.DATASET, index=False)
print(f'{len(df):,} filas, clase_ternaria verificada -> {cm.DATASET}')
