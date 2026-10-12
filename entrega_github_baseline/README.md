# DMEyF 2026 - Primera Competencia

## Qué hay acá

El código para reproducir mi entrega final de la Primera Competencia: `base_k11000.csv`, los 11.000 clientes de 202108 a los que mando estímulo.

El problema es elegir a quién mandarle un estímulo para retener a los clientes que se van en dos meses (BAJA+2). Cada BAJA+2 acertado gana 1.072.500 y cada estímulo cuesta 27.500, así que lo que importa es armar una buena lista y saber dónde cortarla.

sha256 del archivo entregado:

```
6d2f0f0419890ec7087bffef0e03fc1d0499af349dce67eaa844bcce79fa3712
```

Hay una copia en `referencia/` y el último paso del código compara contra ella.

## El modelo

Seguí el baseline que el profesor dejó en el tópico "En Limpio":

- **Datos:** las 152 variables originales, más lags de orden 1 y 2 y sus deltas (760 columnas). `ccajas_depositos` viene toda en cero en 202105, así que la paso a NA en ese mes antes de calcular los lags. Sin undersampling.
- **Target:** BAJA+1 y BAJA+2 como positivos. La ganancia la mido solo con BAJA+2.
- **LightGBM**, con parámetros fijos: `max_bin` 31, `min_data_in_leaf` 0, `feature_fraction` 0,5, `feature_fraction_bynode` 0,2 y `learning_rate` 0,005.
- **Optimización bayesiana** de `num_iterations` (500 a 2.000), `num_leaves` (16 a 1.024) y `min_sum_hessian_in_leaf` (0,001 a 100), con 30 intentos. Entrené con 202103-202104 y medí AUC en 202106.
- **Modelo final:** 10 LightGBM, uno con cada uno de los 10 mejores juegos de hiperparámetros, entrenados con 202103-202106. Promedio sus probabilidades para 202108 y me quedo con los 11.000 clientes más altos.

El paquete no vuelve a correr la bayesiana: los 10 juegos de hiperparámetros están guardados en `config/hiperparametros_baseline.json`.

## Por qué esta entrega y no la anterior

Antes había entregado un blend de Random Forest y LightGBM, que sigue en la carpeta `entrega_github_final/` de este repo. Lo cambié por el baseline porque en el público rindió bastante más: 106,8, 107,0 y 108,6 con 10.000, 10.500 y 11.000 envíos, contra 97 a 99 del blend. Además es la receta que recomendó el profesor. Me quedé con 11.000 envíos porque fue el mejor de los tres cortes.

## Cómo correrlo

**1. Bajar el crudo**, que no está en el repo:

https://storage.googleapis.com/open-courses/dmeyf2026-9c6f/competencia_01_crudo.csv

Dejarlo en `datos/competencia_01_crudo.csv`, o indicar la ruta con `--crudo`.

**2. Instalar las dependencias.** Yo usé Python 3.11.6 en Windows 11:

```
pip install -r requirements_minimos.txt
```

`requirements.txt` tiene la lista completa de mi entorno.

**3. Correr:**

```
python correr_todo.py --crudo RUTA/competencia_01_crudo.csv
```

Tarda entre 80 y 90 minutos con 16 núcleos.

| Paso | Qué hace |
|---|---|
| `src/01_preparar_datos.py` | Lee el crudo, calcula `clase_ternaria` y deja las filas en el orden con el que entrené. |
| `src/02_entrenar_y_predecir.py` | Arregla `ccajas_depositos`, arma las 760 columnas, entrena los 10 modelos y predice 202108. |
| `src/03_generar_submit.py` | Promedia las probabilidades y escribe los 11.000 clientes. |
| `src/04_verificar.py` | Chequea el formato y compara el hash con el archivo que entregué. |

Todo lo configurable (hiperparámetros, semillas, meses, corte y hash esperado) está en `config/`.

## Limitaciones

- El corte de 11.000 lo elegí mirando el público, que es solo el 25% de 202108. La diferencia con 10.500 puede ser ruido.
- Mi propia validación no anticipaba esta mejora: en 202106 el baseline rendía menos que el blend. El público dijo lo contrario.
- "En Limpio" no detallaba algunas cosas, como el target, los rangos de búsqueda o cómo validar la bayesiana. Esas las decidí yo.
- Para que el resultado salga idéntico hay que respetar el orden de filas (`config/orden_filas.npz`) y las versiones de las librerías. Con otra cantidad de núcleos, LightGBM puede dar resultados levemente distintos.

## Verificación

Corrí el paquete desde cero y tardó 88 minutos. El hash del archivo generado coincide con el que entregué:

```
sha256 generado: 6d2f0f0419890ec7087bffef0e03fc1d0499af349dce67eaa844bcce79fa3712
sha256 esperado: 6d2f0f0419890ec7087bffef0e03fc1d0499af349dce67eaa844bcce79fa3712
```
