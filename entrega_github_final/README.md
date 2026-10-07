# DMEyF 2026 — Primera Competencia

## El problema

Tengo que elegir a qué clientes del banco mandarles un estímulo en 202108 para retener a los
que se van a ir en dos meses (BAJA+2). Cada BAJA+2 acertado gana 1.072.500 y cada estímulo
cuesta 27.500, así que no busco el mejor clasificador sino la mejor lista: los clientes con
más chances de irse, cortando donde dejar de enviar conviene más que seguir.

Este paquete reproduce exactamente el archivo que entregué, `submit_03_k10000.csv`
(sha256 `8a68cae8543efaafaf6df9f905fad2845ec2387d4ed7ec3e6d437601b9728fde`), partiendo del
crudo de la cátedra. Hay una copia en `referencia/` y el último paso compara contra ella.

## Cómo correrlo

1. Bajo el crudo (no está en el repositorio):
   `https://storage.googleapis.com/open-courses/dmeyf2026-9c6f/competencia_01_crudo.csv`
   y lo dejo en `datos/competencia_01_crudo.csv`, o paso la ruta con `--crudo` o con la
   variable de entorno `DMEYF_CRUDO`.
2. Instalo las dependencias. Usé Python 3.11.6 en Windows 11:
   ```
   pip install -r requirements_minimos.txt
   ```
   (lightgbm 4.7.0, scikit-learn 1.8.0, numpy 2.4.4, pandas 2.3.3, scipy 1.17.1,
   pyarrow 24.0.0; `requirements.txt` tiene el `pip freeze` completo de mi entorno.)
3. Corro todo:
   ```
   python correr_todo.py --crudo RUTA/competencia_01_crudo.csv
   ```

Tarda entre 25 y 30 minutos con 16 núcleos y unos 4 GB de RAM; casi todo es el paso 02.

| Paso | Qué hace |
|---|---|
| `src/01_preparar_datos.py` | Lee el crudo, calcula `clase_ternaria`, verifica los conteos por mes y guarda el dataset en el orden de filas con el que entrené. |
| `src/02_entrenar_y_predecir.py` | Entrena 10 LightGBM y 10 Random Forest con 202103-202106 y guarda las probabilidades de 202108. Si se corta, retoma desde la última semilla. |
| `src/03_generar_submit.py` | Arma el blend y escribe los 10.000 clientes elegidos en formato del bot. |
| `src/04_verificar.py` | Chequea el formato y que el archivo sea idéntico, por hash, al que envié. |

Hiperparámetros, variables, semillas, meses, cantidad de envíos y hash esperado están en
`config/`; el código no tiene ninguno escrito a mano.

## Decisiones

**Validación temporal en tres ventanas.** Valido en 202104, 202105 y 202106, entrenando cada
vez solo con los meses anteriores. No mezclo filas de un mismo mes entre entrenamiento y
validación: el 98,5% de los clientes aparece en varios meses, y partir filas al azar mete al
mismo cliente de los dos lados. Uso tres meses porque 202106 tiene aguinaldo y se comporta
distinto: muchas ideas que mejoran 202104 y 202105 empeoran 202106. Mido la ganancia enviando
la misma fracción de clientes en cada mes y repito cada variante con 3 semillas (5 para
confirmar).

**BAJA+1 y BAJA+2 como positivos en el LightGBM.** Fue el cambio que más aportó. Contra el
LightGBM base con BAJA+2 solo, sumó +18,2M de ganancia promedio en las tres ventanas. La
ganancia la sigo midiendo contando solo BAJA+2: un BAJA+1 estimulado es un envío perdido.

**Lags y deltas de orden 1 de las 152 variables.** Solos sumaron +4,3M, y junto con el
target nuevo +26,4M sobre la base, positivos en todas las ventanas y semillas que medí. El
orden 2 no agregó nada (-0,5M). Una salvedad: en la ventana 202104 se entrena solo con
202103, donde los lags son todos nulos, así que esa ventana no puede medir su aporte.

**Blend de Random Forest y LightGBM por ranking.** El LightGBM solo rinde parecido en
promedio, pero pierde 24,6M en 202106; el RF (152 variables crudas, clase ternaria) es lo que
sostiene ese mes. Promedio rankings y no probabilidades porque los dos modelos estiman cosas
distintas. Le gana a mi entrega anterior (el mismo blend sin lags ni target nuevo) por +10,4M,
con un desvío entre semillas de 1,7M. Uso 10 semillas por modelo.

**Corte en 10.000 envíos.** La ganancia de validación es plana entre 9.500 y 11.000 envíos:
todo queda a ±2M de los 10.436 que usaba antes, menos que el ruido entre semillas. Elegí
10.000 dentro de esa meseta porque dio el mejor público entre los cortes que probé.

## Qué probé y no funcionó

Para cambiar el modelo exigí que una variante mejorara el promedio de las tres ventanas en al
menos 6M, que fuera positiva en dos de las tres y que no perdiera más de 5M en 202106. Diferencia
contra el modelo entregado, en millones:

| Idea | Promedio | 202106 |
|---|---|---|
| LightGBM con objetivo lambdarank, en lugar del binario dentro del blend | -2,6 | -1,8 |
| Lambdarank como tercer miembro del blend (1/3 cada uno; 5 semillas) | +2,1 | -2,0 |
| LightGBM con objetivo rank_xendcg en el blend | -17,5 | -30,1 |
| Selección de variables por importancia dentro de cada ventana, LightGBM con top 60 | -7,0 | -23,5 |
| Lo mismo con top 150 / RF con top 100 | -0,4 / -2,3 | -6,2 / -5,1 |
| LightGBM dart como tercer miembro | -8,8 | -27,9 |
| Regresión logística sobre percentiles por mes como tercer miembro | -7,1 | -39,2 |
| Historia nueva: meses consecutivos de caída en 10 variables | +1,1 | 0,0 |
| Historia nueva: huecos en la historia del cliente | 0,0 | 0,0 |
| RF + LightGBM solo lags y deltas + LightGBM completo (5 semillas) | +2,8 | -3,1 |
| `ccajas_depositos` en NA en 202105 (está toda en cero ese mes) | -1,3 | -4,0 |
| Régimen de hiperparámetros sugerido en clase (lr 0,005, max_bin 31, 2.000 árboles) en el blend | 0,0 a -3,8 | -8,8 a -13,9 |

Los huecos no aportan porque solo 23 clientes tienen algún mes faltante. Antes había
descartado undersampling de CONTINUA (-1 a -20M), `min_data_in_leaf` chico (-2 a -24M),
rankings por mes y deflactores (-1 a -2M), ExtraTrees en el blend (-30,8M) y XGBoost en el
blend (+4,4M, pero -11M en 202106).

## Límites

- El ruido es grande respecto de las diferencias. Entre semillas, la ganancia promedio varía
  unos 4M, y cambiar el sorteo de la validación la mueve otros 2M. Por eso puse el umbral en
  6M y no tomé como mejora nada que quedara por debajo.
- Los hiperparámetros salieron de búsquedas que miraron 202106, así que en ese mes la
  validación es algo optimista.
- El público es el 25% de 202108. No tuve acceso al privado, así que no sé cuánto de lo que
  vi en el público es ruido. El corte en 10.000 es la única decisión donde lo usé, y solo para
  elegir dentro de un rango que la validación ya mostraba equivalente.
- La reproducción exacta depende del orden de filas (`config/orden_filas.npz`) y de las
  versiones de librerías. El bagging del LightGBM y el bootstrap del RF eligen filas por
  posición, y el dataset con el que entrené salió de una consulta de DuckDB sin `ORDER BY`;
  por eso guardé la permutación que lleva el crudo a ese orden (solo posiciones, sin datos).
  LightGBM usa todos los núcleos, y con otra cantidad de hilos el resultado puede cambiar
  levemente.

## Herramientas

Usé Claude como asistente para escribir y revisar el código y para explorar variantes. Las
decisiones de validación y la elección de la entrega final las tomé yo.

## Verificación

Corrí el paquete desde cero (sin `trabajo/` ni `salida/`) en la misma máquina de la entrega.
Tardó 28,9 minutos y `04_verificar.py` dio:

```
OK    10,000 lineas, todas enteras sin encabezado
OK    sin duplicados
OK    entre 8.000 y 15.000 lineas
OK    sin retornos de carro
OK    todos los IDs estan en 202108
OK    identico al archivo enviado
sha256 generado: 8a68cae8543efaafaf6df9f905fad2845ec2387d4ed7ec3e6d437601b9728fde
sha256 esperado: 8a68cae8543efaafaf6df9f905fad2845ec2387d4ed7ec3e6d437601b9728fde
```
