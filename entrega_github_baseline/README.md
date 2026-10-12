# DMEyF 2026 — Primera Competencia: entrega final (baseline "En Limpio", 11.000 envíos)

## El problema

Tengo que elegir a qué clientes del banco mandarles un estímulo en 202108 para retener a los
que se van a ir en dos meses (BAJA+2). Cada BAJA+2 acertado gana 1.072.500 y cada estímulo
cuesta 27.500, así que no busco el mejor clasificador sino la mejor lista: los clientes con
más chances de irse, cortando donde dejar de enviar conviene más que seguir.

Este paquete reproduce exactamente mi entrega final, `base_k11000.csv`
(sha256 `6d2f0f0419890ec7087bffef0e03fc1d0499af349dce67eaa844bcce79fa3712`), partiendo del
crudo de la cátedra. Hay una copia en `referencia/` y el último paso compara contra ella.

## La entrega

Es el baseline del tópico "En Limpio" que recomendó el profesor, con un corte de 11.000 envíos:

- **Datos:** las 152 variables crudas más sus lags de orden 1 y 2 y sus deltas (valor − lag) de
  orden 1 y 2, en total 760 columnas. `ccajas_depositos` pasa a NA en 202105, porque ese mes
  viene toda en cero (dato roto); lo hago antes de calcular los lags, así no contamina los meses
  siguientes. No hago undersampling.
- **Target:** BAJA+1 y BAJA+2 como positivos. La ganancia la sigo midiendo contando solo BAJA+2.
- **LightGBM con parámetros fijos:** `max_bin` 31, `min_data_in_leaf` 0, `feature_fraction` 0,5,
  `feature_fraction_bynode` 0,2 y `learning_rate` 0,005. El resto queda en los valores por
  defecto de LightGBM.
- **Optimización bayesiana** (Optuna, TPE, 30 trials en 75 minutos) sobre `num_iterations`
  (500–2.000), `num_leaves` (16–1.024) y `min_sum_hessian_in_leaf` (0,001–100). Entrena con
  202103-202104 y mide AUC en 202106 completo. El mejor trial dio AUC 0,9202 (1.650
  iteraciones, 63 hojas, `min_sum_hessian_in_leaf` 11,6).
- **Modelo final:** 10 LightGBM, uno por cada uno de los 10 mejores juegos de hiperparámetros,
  cada uno con una semilla distinta. Se entrenan con 202103-202106, predicen 202108 y promedio
  sus probabilidades.
- **Corte:** los 11.000 clientes con mayor probabilidad promedio.

El código de la búsqueda bayesiana está en `mejoras_v11/01_bayesiana.py`. Este paquete no la
vuelve a correr: lee de `config/hiperparametros_baseline.json` los 10 juegos que salieron de
ella, con el AUC de cada uno.

## Por qué la elegí sobre mi blend

Mi entrega anterior era un blend de rankings entre un Random Forest y un LightGBM. Está
documentado y se reproduce en la carpeta `entrega_github_final/` de este mismo repositorio.
Elegí el baseline por dos razones:

1. **Rindió más en el público.** El baseline dio 106,8, 107,0 y 108,6 con 10.000, 10.500 y
   11.000 envíos. El blend había dado entre 97 y 99, y ninguna de las variantes que probé sobre
   él lo superó.
2. **Es la receta que recomendó el profesor.** No depende de decisiones mías que validé con
   pocas ventanas.

Elegí el corte de 11.000 porque fue el mejor de los tres en el público.

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

Tarda entre 80 y 90 minutos con 16 núcleos; casi todo es el paso 02.

| Paso | Qué hace |
|---|---|
| `src/01_preparar_datos.py` | Lee el crudo, calcula `clase_ternaria`, verifica los conteos por mes y guarda el dataset en el orden de filas con el que entrené. |
| `src/02_entrenar_y_predecir.py` | Pone `ccajas_depositos` en NA en 202105, arma las 760 columnas, entrena los 10 LightGBM con 202103-202106 y guarda las probabilidades de 202108. Si se corta, retoma desde el último modelo terminado. |
| `src/03_generar_submit.py` | Promedia las probabilidades de los 10 modelos y escribe los 11.000 clientes elegidos en el formato del bot. |
| `src/04_verificar.py` | Chequea el formato y que el archivo sea idéntico, por hash, al que envié. |

Los parámetros fijos, los 10 juegos de hiperparámetros, las variables, las semillas, los meses,
la cantidad de envíos y el hash esperado están en `config/`; el código no tiene ninguno escrito
a mano.

## Límites

- **El corte en 11.000 lo elegí mirando el público.** El público es el 25% de 202108 y no tuve
  acceso al privado, así que la diferencia entre 107,0 y 108,6 puede ser ruido.
- **Mi validación temporal no anticipaba esta mejora.** En la ventana 202106 de mi arnés
  (entrenando con 202103-202105, enviando el 6,34% del mes, 3 semillas), el mejor juego del
  baseline dio 398,5M contra 424,1M del blend. Esa medición además favorece al baseline,
  porque la bayesiana eligió hiperparámetros mirando 202106. Que en el público pase lo contrario
  sugiere que 202106 no representa bien a 202108, o que mi arnés le da ventaja al blend.
- **El tópico "En Limpio" no especificaba todo.** Algunas decisiones son mías:
  - el target BAJA+1 + BAJA+2;
  - los rangos de búsqueda;
  - la validación de la bayesiana (entrenar 202103-202104 y medir AUC en 202106);
  - dejar en sus valores por defecto los parámetros no listados.
- **La reproducción exacta depende del orden de filas y de las versiones de librerías.** El
  orden de filas está en `config/orden_filas.npz`. LightGBM arma los bins de cada variable con
  una muestra de 200.000 filas elegidas por posición, así que el resultado depende del orden del
  dataset con el que entrené. Ese dataset salió de una consulta de DuckDB sin `ORDER BY`; por eso
  guardé la permutación que lleva el crudo a ese orden (solo posiciones, sin datos). LightGBM usa todos los núcleos, y con otra cantidad de hilos el resultado puede
  cambiar levemente.

## Herramientas

Usé Claude como asistente para escribir y revisar el código y para explorar variantes. Las
decisiones de validación y la elección de la entrega final las tomé yo.

## Verificación

Corrí el paquete desde cero (sin `trabajo/` ni `salida/`) en la misma máquina de la entrega.
Tardó 88,4 minutos. Las probabilidades de los 10 modelos dieron idénticas a las de la corrida
original, y `04_verificar.py` dio:

```
OK    11,000 lineas, todas enteras sin encabezado
OK    sin duplicados
OK    entre 8.000 y 15.000 lineas
OK    sin retornos de carro
OK    todos los IDs estan en 202108
OK    identico al archivo enviado
sha256 generado: 6d2f0f0419890ec7087bffef0e03fc1d0499af349dce67eaa844bcce79fa3712
sha256 esperado: 6d2f0f0419890ec7087bffef0e03fc1d0499af349dce67eaa844bcce79fa3712
```
