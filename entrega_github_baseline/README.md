# DMEyF 2026 - Primera Competencia: entrega final

**Entrega final:** `base_k11000.csv`, 11.000 clientes de 202108 a los que mando estímulo. Ganancia en el Public Leaderboard: **108,57**.

**El archivo de mi entrega final está en `referencia/base_k11000.csv`**: es el mismo que mandé al bot. Este paquete lo reproduce exactamente desde el crudo, y el último paso compara los hashes para comprobarlo.

sha256: `6d2f0f0419890ec7087bffef0e03fc1d0499af349dce67eaa844bcce79fa3712`

## El problema

Elegir a qué clientes mandarles un estímulo en 202108 para retener a los que se van en dos meses (BAJA+2). Cada BAJA+2 acertado gana 1.072.500 y cada estímulo cuesta 27.500. Con esa asimetría el punto de equilibrio es una probabilidad de 2,5%, así que no se trata de clasificar bien sino de **ordenar bien y cortar en el lugar justo**.

## El modelo final

Es el baseline que el profesor dejó en el tópico "En Limpio":

- **Datos:** 152 variables originales más lags y deltas de orden 1 y 2 (760 columnas). `ccajas_depositos` viene toda en cero en 202105 (dato roto), así que la paso a NA en ese mes antes de calcular los lags.
- **Target:** BAJA+1 y BAJA+2 como positivos. La ganancia se mide solo con BAJA+2.
- **LightGBM** sin undersampling, con `max_bin` 31, `min_data_in_leaf` 0, `feature_fraction` 0,5, `feature_fraction_bynode` 0,2 y `learning_rate` 0,005.
- **Optimización bayesiana** (30 trials) de `num_iterations`, `num_leaves` y `min_sum_hessian_in_leaf`, entrenando con 202103-202104 y midiendo AUC en 202106. El mejor trial: 1.650 iteraciones, 63 hojas y `min_sum_hessian_in_leaf` 11,6.
- **Ensamble:** 10 LightGBM, uno por cada uno de los 10 mejores juegos de hiperparámetros, entrenados con 202103-202106. Promedio sus probabilidades.
- **Corte:** 11.000 envíos.

## Cómo llegué hasta acá

### 1. Primeros modelos y validación

Arranqué con un Random Forest y un LightGBM optimizados con Optuna. Lo primero que corregí fue la validación: el mismo cliente aparece en varios meses, así que partir filas al azar filtra información. Pasé a una **validación temporal en tres ventanas** (202104, 202105 y 202106), entrenando siempre con meses anteriores, midiendo la ganancia al mismo porcentaje de envíos y repitiendo cada variante con varias semillas. El ruido entre semillas resultó de unos 4M, así que solo consideré mejoras de al menos 6M.

| Entrega | Public |
|---|---|
| Random Forest | 85,36 |
| LightGBM | 88,03 |
| Blend RF + LightGBM por ranking | 90,39 |

### 2. Target y variables históricas

Los dos cambios que más aportaron en validación fueron usar **BAJA+1 y BAJA+2 como positivos** (+18,2M) y agregar **lags y deltas de orden 1** (+26,4M combinados). Con eso el blend llegó a **97,00**, y ajustando el corte a 10.000 envíos, a **99,41**.

### 3. Lo que aprendí de los datos

- **202106 es atípico:** es el mes del aguinaldo (`mpayroll` +47%). Muchas variantes que mejoraban 202104 y 202105 empeoraban 202106.
- **No hay variables rotas en 202108** (PSI < 0,12), pero `Visa_fultimo_cierre` y `Master_fultimo_cierre` identifican el mes (AUC adversarial 0,9998). Sacarlas no cambió la ganancia.
- **La curva de ganancia es plana** entre 9.500 y 11.000 envíos. Por eso el corte importa poco dentro de ese rango.

### 4. Más de 100 variantes sin mejora

Probé objetivos de ranking (lambdarank, rank_xendcg), selección de variables, LightGBM dart, regresión logística, XGBoost, ExtraTrees, variables de historia, variables de tarjetas, pesos por mes, corte por probabilidad calibrada y usar 202107 con etiqueta parcial. Ninguna pasó el umbral de validación. También mandé 16 variantes al Public Leaderboard y ninguna superó al blend.

Una hipótesis que vale la pena contar: como 202106 tiene aguinaldo y 202108 no, probé variantes que ganaban en 202104/05 aunque perdieran en 202106. Todas dieron peor en el público (entre 92,8 y 96,3). También probé el peso de 202106 en el entrenamiento: 0 dio 95,65, 0,5 dio 94,02, 2 dio 96,31 y 3 dio 95,23. El óptimo era el peso normal. **202106, el mes más reciente, era el que mejor representaba a 202108.**

### 5. El baseline del profesor

El profesor mostró que elegir por el Public Leaderboard es peligroso: un público alto no garantiza superar la línea de 300 en el privado. Recomendó construir el baseline de "En Limpio" como base segura. Lo implementé y dio **106,8, 107,0 y 108,6** con 10.000, 10.500 y 11.000 envíos: alto en los tres cortes, no un pico aislado. Elegí 11.000.

## Por qué mi validación no lo anticipaba

En mi ventana de 202106 el baseline rendía 26M menos que el blend, y en el público rindió 9 puntos más. La explicación más probable es que **los hiperparámetros del blend se habían optimizado mirando 202106**, así que esa ventana era optimista para el blend. La diferencia clave del baseline con mis intentos parecidos está en **árboles chicos y muy regularizados** (63 hojas, `min_sum_hessian_in_leaf` alto) y en la **diversidad de 10 configuraciones distintas**, en lugar de 10 semillas de una sola.

## Cómo correrlo

**1. Bajar el crudo**, que no está en el repo:

https://storage.googleapis.com/open-courses/dmeyf2026-9c6f/competencia_01_crudo.csv

Dejarlo en `datos/competencia_01_crudo.csv`, o indicar la ruta con `--crudo`.

**2. Instalar las dependencias** (Python 3.11.6, Windows 11):

```
pip install -r requirements_minimos.txt
```

**3. Correr:**

```
python correr_todo.py --crudo RUTA/competencia_01_crudo.csv
```

Tarda entre 80 y 90 minutos con 16 núcleos. No vuelve a correr la bayesiana: los 10 juegos de hiperparámetros están en `config/hiperparametros_baseline.json`.

| Paso | Qué hace |
|---|---|
| `src/01_preparar_datos.py` | Lee el crudo, calcula `clase_ternaria` y deja las filas en el orden con el que entrené. |
| `src/02_entrenar_y_predecir.py` | Arregla `ccajas_depositos`, arma las 760 columnas, entrena los 10 modelos y predice 202108. |
| `src/03_generar_submit.py` | Promedia las probabilidades y escribe los 11.000 clientes. |
| `src/04_verificar.py` | Chequea el formato y compara el hash con el archivo entregado (`referencia/base_k11000.csv`). |

## Limitaciones

- El corte de 11.000 lo elegí mirando el público, que es solo el 25% de 202108.
- "En Limpio" no detallaba el target, los rangos de búsqueda ni la validación de la bayesiana. Esas decisiones son mías.
- La reproducción exacta depende del orden de filas (`config/orden_filas.npz`), de las versiones de las librerías y de la cantidad de núcleos.

## Herramientas

Usé Claude como asistente para escribir código y explorar variantes. Las decisiones de validación y la elección de la entrega final son mías.

## Verificación

Corrida desde cero en 88 minutos:

```
sha256 generado: 6d2f0f0419890ec7087bffef0e03fc1d0499af349dce67eaa844bcce79fa3712
sha256 esperado: 6d2f0f0419890ec7087bffef0e03fc1d0499af349dce67eaa844bcce79fa3712
```
