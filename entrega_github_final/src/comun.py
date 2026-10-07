"""Configuración y funciones compartidas por los pasos 01-04.

Todo lo que define el modelo sale de config/, no del código: hiperparámetros (salida de las
búsquedas con Optuna), lista de variables, semillas, meses, cantidad de envíos y hash esperado.
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
SUBMIT = os.path.join(SALIDA, ENTREGA['archivo_submit'])
REFERENCIA = os.path.join(RAIZ, 'referencia', ENTREGA['archivo_submit'])
EVENTO = 'BAJA+2'


def params(archivo):
    # El random_state de los json es el de la búsqueda; cada modelo usa la semilla del loop.
    opt = leer_json(archivo)['optuna']
    fijos = {k: v for k, v in opt['params_fijos'].items() if k != 'random_state'}
    return {**opt['mejores_params'], **fijos}


RF_PARAMS = params('hiperparametros_rf.json')
LGBM_PARAMS = params('hiperparametros_lgbm.json')


def ruta_crudo():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crudo', default=os.environ.get(
        'DMEYF_CRUDO', os.path.join(RAIZ, 'datos', 'competencia_01_crudo.csv')))
    return ap.parse_args().crudo


def lag1(clientes, mes_idx, v):
    """Valor del mes anterior del mismo cliente, o NaN.
    Se exige que la fila anterior sea exactamente mes_idx - 1: si el cliente faltó un mes, no
    debe heredar un valor de dos meses atrás."""
    orden = np.lexsort((mes_idx, clientes))
    cli, mi, vs = clientes[orden], mes_idx[orden], np.asarray(v)[orden]
    hay = np.zeros(len(vs), dtype=bool)
    hay[1:] = (cli[1:] == cli[:-1]) & (mi[1:] - mi[:-1] == 1)
    previo = np.zeros(len(vs), dtype=bool)
    previo[:-1] = hay[1:]
    res = np.full(len(vs), np.nan, dtype='float32')
    res[hay] = vs[previo]
    out = np.empty_like(res)
    out[orden] = res
    return out


def matriz(df, filas):
    """Columnas del LightGBM: las 152 crudas, sus lag1 y sus deltas (valor - lag1), en ese orden.
    float32 en todo el armado para que los valores sean idénticos a los del entrenamiento original."""
    meses = sorted(df['foto_mes'].unique())
    mi = df['foto_mes'].map({m: i for i, m in enumerate(meses)}).to_numpy()
    cli = df['numero_de_cliente'].to_numpy()
    nf = len(FEATURES)
    X = np.empty((int(filas.sum()), 3 * nf), dtype='float32')
    for j, f in enumerate(FEATURES):
        v = df[f].to_numpy()
        l1 = lag1(cli, mi, v)
        X[:, j] = v[filas]
        X[:, nf + j] = l1[filas]
        X[:, 2 * nf + j] = (v - l1)[filas]
    return X
