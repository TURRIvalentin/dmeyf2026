"""Configuración y funciones compartidas por los pasos 01-04.

Todo lo que define el modelo sale de config/, no del código: parámetros fijos y los 10 juegos de
hiperparámetros de la búsqueda bayesiana, lista de variables, semillas, meses, cantidad de envíos
y hash esperado.
"""
import argparse
import json
import os

import numpy as np

RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CONFIG = os.path.join(RAIZ, 'config')
TRABAJO = os.path.join(RAIZ, 'trabajo')
SALIDA = os.path.join(RAIZ, 'salida')
DATASET = os.path.join(TRABAJO, 'competencia_01.parquet')
PROBAS = os.path.join(TRABAJO, 'probabilidades_202108.npz')
PREDICCION = os.path.join(SALIDA, 'prediccion_completa_202108.csv')


def leer_json(nombre):
    with open(os.path.join(CONFIG, nombre), encoding='utf-8') as f:
        return json.load(f)


ENTREGA = leer_json('entrega.json')
FEATURES = leer_json('features_crudas.json')['features']
SEMILLAS = ENTREGA['semillas']
MESES_TRAIN = ENTREGA['meses_entrenamiento']
MES_PRED = ENTREGA['mes_a_predecir']
N_ENVIOS = ENTREGA['n_envios']
ORDENES = ENTREGA['ordenes_lag_y_delta']
SUBMIT = os.path.join(SALIDA, ENTREGA['archivo_submit'])
REFERENCIA = os.path.join(RAIZ, 'referencia', ENTREGA['archivo_submit'])

_HIP = leer_json('hiperparametros_baseline.json')
LGBM_FIJOS = _HIP['params_fijos']
# Un juego por modelo, en el orden de la bayesiana (mejor AUC primero).
JUEGOS = [dict(n_estimators=j['num_iterations'], num_leaves=j['num_leaves'],
               min_sum_hessian_in_leaf=j['min_sum_hessian_in_leaf']) for j in _HIP['top10']]


def ruta_crudo():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crudo', default=os.environ.get(
        'DMEYF_CRUDO', os.path.join(RAIZ, 'datos', 'competencia_01_crudo.csv')))
    return ap.parse_args().crudo


def lag(clientes, mes_idx, v, k):
    """Valor de k meses antes del mismo cliente, o NaN.
    Se exige que la fila buscada sea exactamente mes_idx - k: si el cliente faltó un mes, no
    debe heredar el valor de otro mes. Con los datos ordenados por cliente y mes, esa fila está
    a lo sumo k posiciones atrás."""
    orden = np.lexsort((mes_idx, clientes))
    cli, mi, vs = clientes[orden], mes_idx[orden], np.asarray(v)[orden]
    res = np.full(len(vs), np.nan, dtype='float32')
    for j in range(1, k + 1):
        hay = np.zeros(len(vs), dtype=bool)
        hay[j:] = (cli[j:] == cli[:-j]) & (mi[j:] - mi[:-j] == k)
        previo = np.zeros(len(vs), dtype=bool)
        previo[:-j] = hay[j:]
        res[hay] = vs[previo]
    out = np.empty_like(res)
    out[orden] = res
    return out


def matriz(df, filas):
    """Columnas del LightGBM, en este orden: las 152 crudas; para cada cruda sus lags de orden
    1 y 2; para cada cruda sus deltas (valor - lag) de orden 1 y 2. Total 760.
    float32 en todo el armado para que los valores sean idénticos a los del entrenamiento original."""
    meses = sorted(df['foto_mes'].unique())
    mi = df['foto_mes'].map({m: i for i, m in enumerate(meses)}).to_numpy()
    cli = df['numero_de_cliente'].to_numpy()
    nf, no = len(FEATURES), len(ORDENES)
    X = np.empty((int(filas.sum()), nf * (1 + 2 * no)), dtype='float32')
    for j, f in enumerate(FEATURES):
        v = df[f].to_numpy()
        X[:, j] = v[filas]
        for i, k in enumerate(ORDENES):
            lk = lag(cli, mi, v, k)
            X[:, nf + j * no + i] = lk[filas]
            X[:, nf * (1 + no) + j * no + i] = (v - lk)[filas]
    return X
