# Detección del estado afectivo del alumno en tutores inteligentes

**Autora:** Paula Lull Rios

Proyecto de Machine Learning supervisado sobre datos del tutor inteligente **ASSISTments** (matemáticas). El objetivo es detectar, **sin sensores** y solo a partir de cómo interactúa el alumno con el tutor, si está concentrado, aburrido, confundido o frustrado. Se entrenan y comparan dos modelos:

- **Modelo multiclase:** 4 estados (concentración, aburrimiento, confusión, frustración).
- **Modelo binario:** concentrado / no concentrado.

---

## Resumen de resultados

Todos los resultados de la tabla son sobre el **conjunto de test** (619 clips de alumnos que el modelo no ha visto), evaluado una sola vez por modelo.
 

| Modelo / Enfoque | Clase | % en dataset | F1-score | f1_macro |
| :--- | :--- | :---: | :---: | :---: |
| Multiclase (4 emociones) | concentration | ~82,5 % | 0,74 | 0,35 |
| Multiclase (4 emociones) | boredom | ~10,0 % | 0,29 | 0,35 |
| Multiclase (4 emociones) | confusion | ~3,9 % | 0,20 | 0,35 |
| Multiclase (4 emociones) | frustration | ~3,6 % | 0,19 | 0,35 |
| Binario (Concentración) | concentrado | ~82,5 % | 0,86 | 0,64 |
| Binario (Concentración) | no concentrado | ~17,5 % | 0,42 | 0,64 |

**Referencias de comparación (cálculo propio a partir de las proporciones):** un modelo que siempre dijera «concentración» tendría `f1_macro` ≈ 0,23 en el multiclase y ≈ 0,45 en el binario.


Tras un primer intento y enfoque para predecir el estado emocional diferenciando 4 posibles clases (tres de ellas siendo prácticamente ruido), se acaba demostrando que los modelos detectan algo de señal por encima del azar, pero con muchas falsas alarmas, dejando de ser utilizables y fiables a la hora de avisar sobre el estado de un alumno concreto. Es la misma conclusión a la que llega el artículo en el que se basa el proyecto.

No obstante, en un segundo enfoque con una formulación binaria centrada en la concentración (donde los estados de frustración, aburrimiento y confusión entran dentro del no estar concentrado), se obtienen resultados más relevantes, mejorando considerablemente las métricas de `f1_macro` y de las respectivas clases. Esto refleja tanto la reducción del ruido entre clases muy minoritarias como la menor complejidad de un problema de 2 clases frente a 4.

---

## 1. Contexto y origen del planteamiento

### El problema

Los tutores humanos observan el estado emocional del alumno y adaptan su forma de enseñar. Un tutor inteligente no puede hacerlo directamente, y los detectores que usan sensores físicos son difíciles de desplegar en un aula real. La alternativa es la **detección de afecto sin sensores** (*sensor-free*): inferir el estado emocional solo a partir del registro de acciones del alumno en la plataforma.

### El artículo de partida

El proyecto parte del artículo **Botelho, Baker y Heffernan, «Improving Sensor-Free Affect Detection Using Deep Learning»**, que trabaja con datos de ASSISTments:

- Las etiquetas de emoción vienen de **observadores humanos** en el aula, con el protocolo BROMP, que anotan el estado del alumno en ventanas de 20 segundos.
- Cada ventana (*clip*) se resume en variables agregadas de las acciones del alumno (media, mínimo, máximo y suma).
- Los autores comparan redes recurrentes (RNN, GRU y LSTM), que aprovechan la secuencia de clips, con los detectores clásicos anteriores.
- Concluyen que el AUC mejora, pero que el kappa no, y que el desbalanceo de clases (≈80 % concentración, ≈4 % confusión, ≈4 % frustración) sigue siendo el problema principal. En su caso, el remuestreo de las clases raras empeoró los resultados.

El detector clásico con el que se comparan es el de **Wang, Heffernan y Heffernan (2015)**, que usa algoritmos tabulares estándar (árboles, Naïve Bayes, regresión logística, entre otros) sobre las mismas variables agregadas.

### Qué pregunta plantea este proyecto

> ¿Hasta dónde se llega con **modelos tabulares clásicos**, sin modelar la secuencia temporal, sobre variables agregadas por clip?

### Qué se toma del artículo y qué cambia

| Se mantiene | Cambia |
|---|---|
| Etiquetas humanas (BROMP) en clips de 20 s | Modelos tabulares (regresión logística, Random Forest, LightGBM), sin RNN ni secuencias |
| Las cuatro emociones del artículo | Se trata el desbalanceo con **ponderación de clases** (`class_weight`), no con remuestreo de las secuencias|
| Validación por alumno (que un alumno no esté a la vez en entrenamiento y validación) | Métrica elegida a propósito para detectar las clases raras (ver apartado 3.6) |
| Evaluación centrada en las emociones poco frecuentes | Se añade un segundo modelo, el binario, para reducir el ruido de las clases raras |


---

## 2. Datos

**Fuente:** carpeta *Affect Data* de Anthony Botelho (autor del artículo), archivo `affect_features_and_labels_folded_all_timesteps.csv`:
https://drive.google.com/drive/folders/1ghIP22PevgoyogB-EppHIernFaOGWSY4

La carpeta incluye también `Affect_Clip_Labels.csv`, pero el README de los autores indica que **no se deben mezclar ni concatenar** los dos archivos (los identificadores se aleatorizaron por separado), así que solo se usa el primero.

### Estructura del archivo

- 48.556 filas y 117 columnas. Cada fila es un **clip de 20 segundos** de un alumno.
- Solo **3.111 clips tienen etiqueta de emoción** (de unos 471 alumnos). El resto son ventanas sin observación humana, y no sirven para aprendizaje supervisado.
- Las variables son resúmenes de lo que hizo el alumno en el clip: aciertos, intentos, pistas, tiempos, errores recientes, historial acumulado y si trabajaba en horario escolar. Cada variable base viene en cuatro versiones (`avg_`, `max_`, `min_`, `sum_`).

### Qué se descartó y por qué

| Elemento | Motivo |
|---|---|
| 12 columnas `confusion_1/2/3`, `boredom_1/2/3`, etc. | **Copian exactamente el target** (coinciden en el 100 % de las filas donde tienen valor) |
| Columna `fold` | No separa por alumno: unos 254 de los 471 alumnos etiquetados aparecen en más de un fold, y en este proyecto se agrupará por alumnos de cara a la validación |
| Identificadores (`rowid`, `clip_id`, `clip_sequence`, `clip_sequence_id`) | No describen el comportamiento del alumno. `user_id` se conserva solo para agrupar. |
| `on_task` / `off_task` | Es otra dimensión (describe conducta, no emoción), no mencionada en el artículo.
| 18 clips con más de una emoción a la vez (0,6 %) | Se descartan para tener una sola etiqueta por clip. |

### Conjunto resultante

**3.093 clips** con una sola emoción:

| Emoción | Clips | % |
|---|---|---|
| Concentración | 2.552 | 82,5 % |
| Aburrimiento | 308 | 10,0 % |
| Confusión | 121 | 3,9 % |
| Frustración | 112 | 3,6 % |

---

## 3. Metodología

### 3.1 Definición del target

- **Multiclase:** la emoción marcada en cada clip (`target`):  boredom=0, concentration=1, confusion=2, frustration=3
- **Binario:** concentración = 1, cualquier otra emoción = 0 (aburrimiento, confusión y frustración se agrupan en «no concentrado»).

### 3.2 Selección de variables

De las 92 columnas agregadas (23 variables base × 4 resúmenes) se llega a **32 columnas más `urbanicity`**:

- **Se descartan `max_` y `min_`.** En la mayoría de los clips coinciden con `avg_`, y generaban la mayoría de correlaciones altas y de columnas casi constantes.
- **Se descarta `timeSinceSkill`** por ser constante (un único valor).
- **Se descartan las variables de ventana de 5 problemas** (`frPast5...`), porque casi duplican las de ventana de 8 (correlación ≈ 0,99).
- **`avg_` se conserva para las 20 variables restantes. `sum_` se añade solo en 12**, donde cuenta algo distinto (por ejemplo, cuántas pistas se pidieron). En las variables de tiempo, de historial acumulado y de errores recientes, `sum_` se infla con el número de acciones del clip y no se usa.
- `urbanicity` (tipo de entorno escolar, con 3 valores) se incluye como categoría.

Todas las decisiones de exploración (constantes, correlaciones, histogramas) se tomaron **solo con los datos de entrenamiento**.

### 3.3 Preprocesado (dentro del pipeline)

| Para | Tratamiento |
|---|---|
| Regresión logística | `log1p` en las columnas con cola larga (máximo mayor que 20) → `StandardScaler` en todas las numéricas → `OneHotEncoder` en `urbanicity` |
| Random Forest y LightGBM | Columnas numéricas sin tocar (no necesitan escalado) + `OneHotEncoder` en `urbanicity` |

El preprocesado va **dentro del pipeline**, de modo que en cada partición de la validación se ajusta solo con los datos de entrenamiento de esa partición, sin fuga de información.

### 3.4 Partición y validación

- **Train / test:** 80 % / 20 % con `StratifiedGroupKFold` agrupando por alumno (`user_id`): **2.474 clips de entrenamiento y 619 de test, sin ningún alumno en común**, con proporciones de clase casi idénticas en ambos.
- **Validación cruzada:** 5 folds, también agrupados por alumno y estratificados, dentro del conjunto de entrenamiento.
- `random_state=42` en todo el proyecto.

### 3.5 Modelos e hiperparámetros

Tres familias de modelos, cada una con su búsqueda en rejilla (`GridSearchCV`):

| Modelo | Hiperparámetros explorados |
|---|---|
| Regresión logística | `C`, `class_weight` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight` |
| LightGBM | `n_estimators`, `learning_rate`, `max_depth`, `min_child_samples`, `class_weight` |

### 3.6 Métricas

La **exactitud (*accuracy*) engaña** en este problema: un modelo que siempre dijera «concentración» acertaría el 82,5 % sin aprender nada. Por eso:

- **Multiclase:** se optimiza el **F1 medio de las tres clases raras** (aburrimiento, confusión y frustración), llamado `f1_raras`, para que la clase mayoritaria no «ayude» a la nota. Se reportan además `f1_macro`, `AUC` (uno contra el resto) y precisión, recall y f1 por clase.
- **Binario:** se optimiza `f1_macro` (media del F1 de las dos clases). Se reporta además el `AUC` y precisión, recall y f1 por clase.

**Una lección del proyecto:** en el modelo multiclase, un Random Forest elegido por AUC (0,788) tenía f1_macro 0,25, es decir, el nivel de un modelo que siempre responde «concentración». Al elegir por `f1_raras` y con `class_weight` balanceado, el optimizador cambió a un Logistic Regressor subiendo el `f1_macro` a 0,35, así como las f1 de las clases raras. En este caso, el `AUC` mide si el modelo ordena bien los clips, pero no si decide bien.

---

## 4. Resultados

### 4.1 Modelo multiclase

**Comparación inicial, sin ajustar, con validación cruzada (f1_macro)**: regresión logística 0,320, LightGBM 0,289, Random Forest 0,288.

**Modelo final:** regresión logística con ponderación de clases.
**Hiperparámetros seleccionados:** C = 1 y class_weight = "balanced", elegidos por la búsqueda en rejilla. max_iter = 10000 y random_state = 42 quedaron fijos, sin explorar.

**Resultados por clase para el modelo final en test (619 clips):**

| Clase | Clips | Precisión | Recall | F1 |
|---|---|---|---|---|
| Aburrimiento | 61 | 0,23 | 0,38 | 0,29 |
| Concentración | 511 | 0,95 | 0,60 | 0,74 |
| Confusión | 24 | 0,12 | 0,58 | 0,20 |
| Frustración | 23 | 0,12 | 0,43 | 0,19 |

`f1_raras` = **0,227**, `f1_macro` = **0,355**, `AUC` (uno contra el resto) = **0,742**.

Los resultados en test son **casi idénticos a los de validación** (`f1_raras` ≈ 0,22, `f1_macro` ≈ 0,35), así que no se observa sobreajuste.

Las tres emociones raras se detectan con muy baja precisión. Se detectan muchas falsas alarmas por cada acierto, y estas vienen sobre todo de alumnos concentrados. Es el precio de dar más peso a las clases raras.

### 4.2 Modelo binario (concentrado / no concentrado)

**Modelo final:** Random Forest.

**Hiperparámetros:** 'class_weight': 'balanced_subsample', 'max_depth': None, 'min_samples_leaf': 5 y 'n_estimators': 500 elegidos por la búsqueda en rejilla, lo que viene siendo un bosque de 500 árboles con profundidad sin límite. n_jobs=1 y random_state=42 quedaron fijos sin explorar.

Comparación inicial, sin ajustar, con validación cruzada (F1 macro): LightGBM 0,661, Random Forest 0,659, regresión logística 0,637.

**Validación (predicciones de validación cruzada sobre entrenamiento):**

| Clase | Clips | Precisión | Recall | F1 |
|---|---|---|---|---|
| No concentrado | 433 | 0,46 | 0,59 | 0,52 |
| Concentrado | 2.041 | 0,91 | 0,85 | 0,88 |

`F1_macro` = **0,70.**, `AUC` = **0,742**.

**Test (619 clips):**

| Clase | Clips | Precisión | Recall | F1 |
|---|---|---|---|---|
| No concentrado | 108 | 0,37 | 0,48 | 0,42 |
| Concentrado | 511 | 0,88 | 0,83 | 0,86 |

`F1_macro` = **0,638.**, `AUC` = **0,79**.

- El F1 macro de 0,638 supera con claridad la referencia de ≈ 0,45 de un modelo que siempre dice «concentrado».
- La precisión de «no concentrado» (0,37) es algo mejor que en el caso de las clases raras, pero todavía sigue habiendo falsas alarmas.
- **Hay una caída ligera entre validación y test** (F1 macro de 0,70 a 0,64, y F1 de «no concentrado» de 0,52 a 0,42). Puede deberse en parte al optimismo de elegir el mejor en validación entre muchas combinaciones, o a que con solo 108 casos el margen de error sea amplio. Hay que tener en cuenta también, que en test, el modelo predice sobre alumnos nuevos, lo que puede resultar más costoso.



### 4.3 Comparación entre los dos modelos

- **Qué ocurre en validación:** en validación cruzada, el modelo binario (Random Forest) alcanza un `f1_macro` de **0,70** y un AUC de **0,79**. Por su parte, la regresión logística multiclase obtiene un `f1_macro` de **0,35** y un AUC de **0,73**, con valores bajos en las clases minoritarias (`boredom` 0,33, `confusion` 0,16 y `frustration` 0,17). Agrupar las tres emociones menos frecuentes en una única clase «no concentrado» reduce el ruido entre clases raras y hace más manejable la frontera de decisión.

- **Qué ocurre en test (619 clips):** en el conjunto de prueba sobre alumnos no vistos, el modelo binario pasa a un `f1_macro` de **0,638**, mientras que el multiclase (regresión logística) obtiene un `f1_macro` de **0,355**. Aunque en el binario hay una caída moderada respecto a validación, sigue superando con claridad la referencia de un modelo constante (~0,45) y mantiene un rendimiento más equilibrado que el multiclase en la detección de falta de concentración.

- **Qué se gana y qué se pierde:** el modelo binario ofrece mayor estabilidad estadística (108 clips de «no concentrado» en test frente a 23-24 de `confusion` y `frustration`) y un rendimiento más sólido para detectar si el alumno no está concentrado. A cambio, **pierde el detalle de qué emoción específica ocurre** (aburrimiento, confusión o frustración).

---

## 5. Comparación con el artículo de los modelos multiclase

| | Botelho et al. | Este proyecto |
|---|---|---|
| Datos | 7.663 clips de 646 alumnos, 204 variables por clip | 3.111 clips etiquetados (unos 471 alumnos), 92 variables agregadas |
| Modelos | RNN, GRU y LSTM (con secuencia) | Regresión logística, Random Forest y LightGBM (tabulares) |
| AUC | 0,73 – 0,78 | 0,742 en test (multiclase) |
| Kappa de Cohen | 0,11 – 0,21 | No calculado |
| Mejor / peor detectadas | Concentración y aburrimiento / confusión y frustración | Concentración y aburrimiento / confusión y frustración |
| Desbalanceo | El remuestreo de secuencias empeoró los resultados | La ponderación de clases mejoró el F1 de las clases raras |
| Conclusión práctica | Calidad insuficiente para intervenir en tiempo real | La misma |

**Qué coincide:** el AUC es aceptable pero las decisiones son muy flojas, el orden de dificultad por emoción es el mismo y la conclusión práctica es la misma.

**Qué no es comparable:** los datos no son los mismos (menos clips y menos variables) y el AUC no se calcula igual (aquí es la media de uno contra el resto para las cuatro clases; el artículo lo calcula con su propio protocolo por etiqueta). Las cifras sirven como **orientación**, no como para una comparación directa.

**La diferencia en el desbalanceo** no es una contradicción: el artículo remuestrea secuencias completas y aquí se ponderan las clases, que son técnicas distintas.

---

## 6. Conclusiones

1. **Hay señal real, pero débil.** Las emociones raras se detectan con una precisión extremadamente baja.
2. **Ninguno de los dos modelos sirve para avisar de un alumno concreto.** Con precisiones de entre 0,12 y 0,37, la mayoría de las alertas serían falsas. El artículo de Botelho señala que los detectores sin sensores han sido útiles en conjunto para investigación, pero no para intervenir. Este proyecto llega a la misma conclusión.
3. **El cuello de botella para la intervención está, fundamentalmente, en los datos.** Utilizar este tipo de modelos en intervención real sigue siendo igual de prometedor. Más allá de ajustar algoritmos o hiperparámetros, el principal límite del problema es la escasez de ejemplos etiquetados en las emociones minoritarias (como confusión y frustración), por lo que el punto clave para avanzar es recoger más datos y observaciones de calidad de esas clases raras.
4. **La métrica de optimización es clave.** Con clases tan desbalanceadas, elegir por AUC o por exactitud lleva a modelos que parecen buenos y no lo son. Cambiar la métrica de optimización cambió el modelo ganador y mejoró el F1 de las clases raras.
5. **Un modelo simple basta.** La regresión logística fue el mejor en el problema multiclase, así como el random forest en el problema binario, y las diferencias entre algoritmos son pequeñas.
6. **Simplificar a binario ayuda a medir, no a resolver.** Aporta más casos y métricas más estables, y una mejora moderada en validación, a costa de perder qué emoción es.


---

## 7. Limitaciones y trabajo futuro

### Limitaciones

- **Pocos casos de las clases raras:** 24 de confusión y 23 de frustración en test. Los resultados por clase son muy ruidosos, y no se calcularon intervalos de confianza.
- **Los datos no son exactamente los del artículo**.
- **Sesgo de selección leve:** el modelo final es el mejor de muchas combinaciones evaluadas con la misma validación, y la lista de variables se decidió con una exploración inicial antes de repetirla solo con entrenamiento.
- **Sin modelo secuencial:** no se aprovecha el orden de los clips de un mismo alumno, que es justo lo que explora el artículo.

### Trabajo futuro

- Cálculo de intervalos de confianza y errores.
- Probar una validación agrupada por secuencia (`clip_sequence_id`) frente a por alumno.
- Añadir como variables el contexto de los clips anteriores del mismo alumno, sin usar series temporales.
- Construir un modelo con dos pasos (¿concentrado?, y si no, ¿qué emoción?).

---

## 8. Estructura del repositorio

```
├── data/
│   ├── raw/          # CSV original, sin procesar
│   ├── processed/    # 3.093 clips filtrados, con las variables elegidas y el target
│   ├── train/        # 2.474 clips de entrenamiento
│   └── test/         # 619 clips de test
├── notebooks/
│   ├── main_binario.ipynb      # modelo binario
│   └── main_multiclase.ipynb   # modelo multiclase
├── src/
│   ├── pipe_preproc_training.py
│   └── evaluation.py
├── modelos/
│   ├── final_model_multiclase.pkl # modelo multiclase
│   └── final_model_binario.pkl    # modelo binario
├── imgs/
├── articulos_y_exploracion/
│   
└── README.md
```

Los modelos se guardan como **pipeline completo** (preprocesado + modelo) con `joblib`, así que se pueden usar directamente sobre datos nuevos.

---

## 9. Cómo ejecutar el proyecto

1. **Descargar los datos** desde el enlace del apartado 2 y colocar `affect_features_and_labels_folded_all_timesteps.csv` en `data/raw/`.
2. **Instalar las dependencias** (Python 3 con Jupyter):
   ```
   pip install pandas numpy scikit-learn lightgbm matplotlib joblib jupyter
   ```
3. **Ejecutar los main**: Celda por celda para ver todo el proceso en cada uno de los dos enfoques: predicción multiclase y binaria.
4. **Usar un modelo guardado:**
   ```python
   import joblib
   modelo = joblib.load("models/final_model_binario.pkl")
   prediccion = modelo.predict(X_nuevo)    # X_nuevo con las mismas columnas que train!!!
   ```

El modelo multiclase predice números (0, 1, 2 y 3), que corresponden a las emociones en orden alfabético: aburrimiento, concentración, confusión y frustración.

---

## 10. Referencias

- ASSISTments. *2012–2013 School Data with Affect*. Disponible en: https://sites.google.com/site/assistmentsdata/2012-13-school-data-with-affect
- Botelho, A. F., Baker, R. S., & Heffernan, N. T. (2017). *Improving Sensor-Free Affect Detection Using Deep Learning.* Artificial Intelligence in Education (AIED 2017). Springer. 
- Datos: carpeta *Affect Data* de Anthony Botelho (enlace en el apartado 2).
- Wang, Y., Heffernan, N. T., & Heffernan, C. (2015). *Towards Better Affect Detectors: Effect of Missing Skills, Class Features and Common Wrong Answers.* LAK '15. 

