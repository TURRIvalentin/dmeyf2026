"""Corre los pasos 01 a 04 en orden.

    python correr_todo.py [--crudo RUTA/competencia_01_crudo.csv]

Sin --crudo usa la variable de entorno DMEYF_CRUDO o datos/competencia_01_crudo.csv.
"""
import os
import subprocess
import sys
from time import time

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
PASOS = ['01_preparar_datos.py', '02_entrenar_y_predecir.py', '03_generar_submit.py', '04_verificar.py']

t0 = time()
for paso in PASOS:
    print(f'{paso} ({(time() - t0) / 60:.1f} min)', flush=True)
    # Solo 01 y 04 leen el crudo; el argumento se les pasa tal cual.
    args = sys.argv[1:] if paso in (PASOS[0], PASOS[3]) else []
    if subprocess.run([sys.executable, os.path.join(SRC, paso), *args], cwd=SRC).returncode != 0:
        sys.exit(f'{paso} termino con error')
print(f'listo en {(time() - t0) / 60:.1f} min')
