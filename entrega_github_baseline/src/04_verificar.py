"""Paso 04: verifica el formato del archivo y que sea idéntico (sha256) al que envié al bot.
Termina con código 1 si algún chequeo falla."""
import hashlib
import re
import sys

import pandas as pd

import comun as cm

contenido = open(cm.SUBMIT, 'rb').read()
lineas = contenido.decode('ascii').split('\n')
ids = [int(x) for x in lineas[:-1] if re.fullmatch(r'\d+', x)]
crudo = pd.read_csv(cm.ruta_crudo(), usecols=['numero_de_cliente', 'foto_mes'])
ids_mes = set(crudo.loc[crudo['foto_mes'] == cm.MES_PRED, 'numero_de_cliente'].astype('int64'))
sha = hashlib.sha256(contenido).hexdigest()
sha_ref = hashlib.sha256(open(cm.REFERENCIA, 'rb').read()).hexdigest()

chequeos = [
    (lineas[-1] == '' and len(ids) == len(lineas) - 1, f'{len(lineas) - 1:,} lineas, todas enteras sin encabezado'),
    (len(ids) == len(set(ids)), 'sin duplicados'),
    (8000 <= len(ids) <= 15000, 'entre 8.000 y 15.000 lineas'),
    (b'\r' not in contenido, 'sin retornos de carro'),
    (set(ids) <= ids_mes, f'todos los IDs estan en {cm.MES_PRED}'),
    (sha == cm.ENTREGA['sha256_submit_esperado'] == sha_ref, 'identico al archivo enviado'),
]
for ok, msg in chequeos:
    print(('OK    ' if ok else 'FALLA ') + msg)
print(f'sha256 generado: {sha}\nsha256 esperado: {cm.ENTREGA["sha256_submit_esperado"]}')
sys.exit(0 if all(ok for ok, _ in chequeos) else 1)
