# Predicción de ausentismo a turnos médicos (*No-Show*)

## Descripción del proyecto

El ausentismo de pacientes a turnos médicos ya agendados (*no-show*) genera huecos de agenda que no pueden reasignarse a tiempo. También aumenta las listas de espera y los costos operativos, y reduce la eficiencia en el uso de personal, equipamiento e insumos.

Este proyecto construye y compara **modelos de clasificación** que estiman, a partir de datos históricos, la probabilidad de que un paciente **no asista** a su turno. Con esa estimación anticipada, un centro de salud puede intervenir de forma proactiva con recordatorios adicionales, sobreturnos controlados o reprogramaciones.

| Aspecto | Detalle |
|---|---|
| **Tarea de minería de datos** | Clasificación binaria |
| **Variable objetivo** | `no_show` (1 = no asistió, 0 = asistió) |
| **Metodología** | CRISP-DM |
| **Desafío principal** | Desbalance de clases (79,8% asiste / 20,2% no asiste) |
| **Momento de la predicción** | Antes del día del turno, cuando el centro de salud todavía puede actuar |

---

## Conjunto de datos

| Campo | Valor |
|---|---|
| **Nombre** | Medical Appointment No Shows |
| **Fuente** | Kaggle — [joniarroba/noshowappointments](https://www.kaggle.com/datasets/joniarroba/noshowappointments) |
| **Origen** | Registros públicos de turnos médicos de Vitória, Brasil (2016) |
| **Instancias** | 110.527 turnos de 62.299 pacientes (110.516 tras la limpieza) |
| **Atributos** | 14 originales (13 predictores candidatos + variable objetivo) |
| **Período de turnos** | 29/04/2016 a 08/06/2016 (27 días con turnos) |

Cada fila representa un turno agendado. Incluye:

- datos demográficos del paciente (`Gender`, `Age`);
- el barrio de atención (`Neighbourhood`);
- condiciones de salud preexistentes (`Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`);
- inscripción en un programa social (`Scholarship`);
- recepción de recordatorio por SMS (`SMS_received`);
- las fechas de agendamiento (`ScheduledDay`) y del turno (`AppointmentDay`).

> **Nota:** el archivo original **no se versiona** en el repositorio. Ver [Instalación](#instalación-y-ejecución) para descargarlo.

---

## Enfoque metodológico

El trabajo sigue las fases de **CRISP-DM**:

```mermaid
flowchart LR
    A[Comprensión<br/>del negocio] --> B[Comprensión<br/>de los datos]
    B --> C[Preparación<br/>de los datos]
    C --> D[Modelado]
    D --> E[Evaluación]
    E --> F[Difusión de<br/>resultados]
    E -.-> B
```

### Decisiones clave

**Desbalance de clases.** Se analizó antes de entrenar. Un clasificador trivial que siempre predice "asiste" alcanza 79,8% de exactitud con recall 0, por lo que la exactitud **no** es la métrica principal. Se compararon cuatro estrategias aplicadas **solo sobre entrenamiento**: sin tratamiento, ponderación de clases, submuestreo aleatorio y SMOTE.

La evidencia mostró que el desbalance afecta a la **elección del umbral de decisión**, no a la capacidad de discriminación de los modelos. Por eso se resuelve ajustando el umbral, no remuestreando los datos (ver [Resultados](#modelado-baseline-y-tratamiento-del-desbalance-dataset-completo)).

**Métricas.** Se priorizan:

- PR-AUC, métrica principal (piso de referencia: 0,202);
- ROC-AUC (piso de referencia: 0,5);
- recall y F1 de la clase *no-show*;
- exactitud balanceada;
- Brier, para la calibración de las probabilidades.

**Prevención de fuga de información (*data leakage*).** Se aplican tres medidas:

1. **Partición agrupada por paciente.** El 39% de los pacientes tiene más de un turno, y el 82% de los grupos de turnos de un mismo paciente en un mismo día tiene igual resultado. Todos los turnos de un paciente quedan del mismo lado de la partición.
2. **Historial estrictamente anterior.** Los atributos de historial usan solo turnos de días anteriores, nunca del mismo día ni posteriores.
3. **Transformaciones aprendidas dentro del pipeline.** La codificación por objetivo, el escalado y el remuestreo se ajustan dentro de cada *fold*, solo con datos de entrenamiento.

La partición temporal se descartó como estrategia principal: el período abarca solo 27 días con turnos y el conjunto de prueba quedaría reducido a una semana.

**Evaluación honesta.** La validación cruzada y la optimización de hiperparámetros usan solo el conjunto de entrenamiento, con **5 *folds* fijos** guardados en el dataset para que todos los modelos se comparen con las mismas particiones. El conjunto de prueba se utiliza **una única vez**, al final.

### Modelos

| Modelo | Familia de preprocesamiento | Rol |
|---|---|---|
| `DummyClassifier` | — | Línea base de referencia |
| Regresión Logística | Lineal | Referencia lineal |
| Árbol de Decisión | Árboles | Interpretabilidad y extracción de reglas |
| Random Forest | Árboles | Ensamble por *bagging* |
| LightGBM / XGBoost | Árboles | Ensambles por *gradient boosting* |

### Preprocesamiento por familia de modelos

| Transformación | Árboles | Lineal |
|---|---|---|
| Atributos numéricos | Sin cambios | `log1p` en atributos asimétricos y estandarización |
| Edad | Numérica | Discretizada en 9 tramos con *one-hot* |
| Día de la semana | Ordinal (0 a 4) | *One-hot* |
| Barrio (81 categorías) | Codificación por objetivo (`TargetEncoder`) | Codificación por objetivo y estandarización |
| Binarias | Sin cambios | Sin cambios |
| **Columnas resultantes** | **18** | **28** |

### Optimización y trazabilidad

**Optuna** optimiza los hiperparámetros de los candidatos seleccionados en el modelado baseline. Usa el muestreador TPE con semilla fija, poda con `MedianPruner` y PR-AUC como métrica objetivo. La estrategia de desbalance se trata como un hiperparámetro más, entre "sin tratamiento" y "ponderación de clases".

**MLflow** registra cada corrida con base SQLite (`mlflow.db`) y artefactos en `mlartifacts/`. De cada corrida guarda:

- parámetros del modelo y del clasificador;
- media y desvío de cada métrica entre *folds*;
- métricas por *fold* como artefacto;
- etiquetas de trazabilidad: *hash* SHA-256 del dataset, semilla, *commit* de Git, datos utilizados y etapa.

La duración registrada de cada corrida incluye el entrenamiento. Los experimentos ejecutados con la muestra llevan el sufijo `_muestra`.

| Experimento MLflow | Contenido | Estado |
|---|---|---|
| `01_baseline` | Todos los modelos con parámetros por defecto | Completo (6 corridas) |
| `02_desbalance` | Modelos × estrategias de tratamiento del desbalance | Completo (15 corridas) |
| `03_optuna` | Estudios de optimización (corridas anidadas por *trial*) | Pendiente |
| `04_modelo_final` | Modelo seleccionado, umbral y evaluación en prueba | Pendiente |

---

## Flujo de trabajo: muestra de desarrollo y dataset completo

Para no procesar el dataset completo en cada prueba, el código se desarrolla sobre una **muestra representativa** y los resultados finales se obtienen siempre con el **dataset completo**.

| Paso | Descripción |
|---|---|
| 1 | `scripts/crear_muestra.py` genera `data/interim/muestra_noshow.csv` (~15% de los pacientes, ~16.700 turnos) |
| 2 | Cada notebook se desarrolla y prueba con `USAR_MUESTRA = True` |
| 3 | La ejecución definitiva se hace con `USAR_MUESTRA = False` (dataset completo) |
| 4 | Los resultados del informe provienen solo de la ejecución definitiva |

**Criterios de la muestra:**

- Se seleccionan **pacientes completos** (todos sus turnos), no filas sueltas, para conservar el historial.
- El muestreo se estratifica por "tuvo alguna ausencia" × "cantidad de turnos".
- Siempre se incluyen los pacientes con anomalías conocidas, para poder probar el código de limpieza.

La comparación entre muestra y original queda en `reports/tables/muestra_resumen.md`. Todos los indicadores difieren en menos de un punto porcentual.

Los archivos y experimentos generados a partir de la muestra llevan el sufijo `_muestra`, para que nunca se mezclen con los del dataset completo.

---

## Estructura del repositorio

```text
tfi-mineria-datos-noshow/
├── data/
│   ├── raw/                              # Dataset original (no versionado)
│   ├── interim/                          # Muestra de desarrollo (no versionada)
│   └── processed/                        # No versionado
│       ├── turnos_procesados.csv         # Dataset limpio, con atributos, partición y folds
│       └── metadatos_preparacion.json    # Atributos, conteos, semilla y hash SHA-256
├── docs/
│   ├── propuesta/                        # Propuesta de TFI aprobada
│   └── capturas/                         # Capturas de MLflow para el anexo del informe
│       ├── mlflow_01_corridas.png
│       ├── mlflow_02_comparacion_desbalance.png
│       └── mlflow_03_detalle_corrida.png
├── mlartifacts/                          # Artefactos de MLflow (no versionado)
├── models/                               # Modelos serializados (no versionados)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preparacion.ipynb
│   ├── 03_modelado_baseline.ipynb
│   ├── 04_optimizacion_optuna.ipynb
│   └── 05_evaluacion_final.ipynb
├── reports/
│   ├── figures/                          # Gráficos (prefijos eda_, prep_, mod_, ...)
│   ├── tables/                           # Tablas CSV y resúmenes Markdown por etapa
│   └── informe_final.md                  # Informe del TFI (se exporta a PDF)
├── scripts/
│   └── crear_muestra.py                  # Genera la muestra de desarrollo
├── src/
│   ├── __init__.py
│   ├── config.py                         # Rutas, semilla y constantes globales
│   ├── data.py                           # Carga, limpieza y carga del dataset procesado
│   ├── features.py                       # Atributos, historial, partición y preprocesadores
│   ├── modeling.py                       # Modelos y estrategias de desbalance
│   ├── evaluation.py                     # Métricas, validación cruzada y umbral
│   ├── reporting.py                      # Figuras, tablas, Markdown y tasas con IC
│   └── tracking.py                       # Registro de experimentos en MLflow
├── mlflow.db                             # Base de MLflow (no versionada)
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md
```

### Módulos de `src/`

| Módulo | Funciones principales |
|---|---|
| `data.py` | `cargar_datos_crudos`, `limpiar_datos`, `cargar_procesados`, `ruta_procesados` |
| `features.py` | `crear_atributos`, `agregar_historial`, `asignar_particiones`, `iterador_folds`, `construir_preprocesador`; listas de atributos por familia (`ATRIBUTOS_ARBOL`, `ATRIBUTOS_LINEAL`) |
| `modeling.py` | `crear_clasificador`, `crear_pipeline` (preprocesamiento → remuestreo opcional → clasificador), `atributos_de`; catálogos `MODELOS` y `ESTRATEGIAS` |
| `evaluation.py` | `calcular_metricas`, `evaluar_cv` (folds fijos y predicciones fuera de fold), `proba_entrenamiento`, `resumir_folds`, `umbral_f1_optimo` |
| `tracking.py` | `configurar_experimento`, `etiquetas_base`, `registrar_corrida`, `exportar_corridas` |
| `reporting.py` | `guardar_figura`, `guardar_tabla`, `tabla_markdown`, `tasa_con_ic`, `tasa_por` |

---

## Instalación y ejecución

### Requisitos

- Python 3.11
- Git

### 1. Clonar y preparar el entorno

**Windows (PowerShell):**

```powershell
git clone <URL-del-repositorio>
cd tfi-mineria-datos-noshow
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
git clone <URL-del-repositorio>
cd tfi-mineria-datos-noshow
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Dependencias de MLflow:** `requirements.txt` fija `sqlalchemy==2.0.36` y `alembic==1.14.0`. MLflow 2.19 admite cualquier SQLAlchemy menor a 3, pero la versión 2.1 eliminó una clase que MLflow utiliza, lo que impide crear la base de experimentos.

### 2. Descargar el dataset

1. Descargar el dataset desde [Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments).
2. Copiar el archivo `KaggleV2-May-2016.csv` en `data/raw/`.

### 3. Generar la muestra de desarrollo (opcional)

Solo es necesaria para desarrollar o probar código.

```powershell
python scripts/crear_muestra.py
```

| Opción | Descripción | Valor por defecto |
|---|---|---|
| `--frac` | Fracción de pacientes a muestrear | `0.15` |
| `--seed` | Semilla aleatoria | `42` |
| `--sin-anomalias` | No forzar la inclusión de pacientes con anomalías | Desactivada |

### 4. Ejecutar los notebooks en orden

```powershell
jupyter lab
```

Cada notebook tiene al inicio el interruptor `USAR_MUESTRA`. Debe estar en `False` para la ejecución definitiva. Conviene hacer un *commit* antes de ejecutar los notebooks de modelado, para que la etiqueta `git_commit` de MLflow identifique la versión exacta del código.

| Orden | Notebook | Fase CRISP-DM | Salidas principales |
|---|---|---|---|
| 1 | `01_EDA.ipynb` | Comprensión de los datos | 11 figuras `eda_*.png`, tablas `eda_*.csv`, `eda_resumen.md` |
| 2 | `02_preparacion.ipynb` | Preparación de los datos | `turnos_procesados.csv`, `metadatos_preparacion.json`, 3 figuras `prep_*.png`, tablas `prep_*.csv`, `prep_resumen.md` |
| 3 | `03_modelado_baseline.ipynb` | Modelado | Experimentos `01_baseline` y `02_desbalance`, 6 figuras `mod_*.png`, `mod_candidatos.csv`, `mod_resumen.md` |
| 4 | `04_optimizacion_optuna.ipynb` | Modelado (optimización) | Experimento `03_optuna` |
| 5 | `05_evaluacion_final.ipynb` | Evaluación | Experimento `04_modelo_final` |

> **Reejecutar un notebook de modelado:** cada ejecución agrega corridas nuevas a MLflow. Para evitar duplicados, borrar antes `mlflow.db` y la carpeta `mlartifacts/`; o, si ya hay experimentos posteriores que conservar, eliminar desde la interfaz de MLflow solo el experimento a reemplazar.

### 5. Consultar los experimentos en MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Luego abrir <http://127.0.0.1:5000> en el navegador.

---

## Reproducibilidad

- Semilla global fija (`SEED = 42`) definida en `src/config.py`.
- Versiones de librerías fijadas en `requirements.txt`, verificadas como compatibles entre sí.
- Muestra de desarrollo determinística (misma semilla, misma muestra).
- Partición y *folds* de validación cruzada guardados en el dataset procesado: todos los modelos usan exactamente las mismas divisiones.
- *Hash* SHA-256 del dataset procesado guardado en `metadatos_preparacion.json`, verificado al inicio de cada notebook de modelado y registrado en cada corrida de MLflow.
- Controles automáticos (`assert`) en los notebooks que verifican la limpieza, la ausencia de fuga en el historial, la separación de pacientes entre conjuntos y que todas las corridas usen el mismo dataset.

---

## Estado del proyecto

- [x] Propuesta de TFI aprobada
- [x] Estructura del repositorio
- [x] Muestra de desarrollo
- [x] Análisis exploratorio de datos
- [x] Limpieza y preparación
- [x] Modelado *baseline* y tratamiento del desbalance
- [ ] Optimización con Optuna
- [ ] Evaluación final y selección del modelo
- [ ] Informe final

---

## Resultados

### Análisis exploratorio (dataset completo)

**Calidad de los datos.** No hay valores faltantes ni duplicados exactos. Las inconsistencias detectadas son pocas:

- 1 edad negativa;
- 5 edades de 115 años;
- 5 turnos anteriores a su fecha de agendamiento.

Además, `Handcap` es un conteo de 0 a 4, no un atributo binario.

**Hallazgos principales:**

| Hallazgo | Evidencia |
|---|---|
| La anticipación es el atributo más asociado al no-show | Efecto 0,40 (biserial de rangos); no-show de 4,6% en turnos del mismo día frente a 28,5% con anticipación |
| El efecto aparente del SMS es una variable de confusión | Globalmente los turnos con SMS faltan más, pero dentro de cada tramo de anticipación el SMS **reduce** el no-show (por ejemplo, de 36,8% a 29,7% con 16 a 30 días) |
| Las ausencias previas del paciente predicen su comportamiento | Ver la corrección en la sección de preparación |
| La edad tiene una relación no lineal | Máximo entre 13 y 30 años (~25%), mínimo entre 61 y 75 (~15%) |
| El barrio aporta señal | 31 de 81 barrios difieren significativamente de la tasa global |
| Sin asociación significativa | Género, alcoholismo y discapacidad (p > 0,1) |

### Preparación de los datos (dataset completo)

**Limpieza.** Se eliminaron 11 filas (0,010%) por valores imposibles o inconsistentes; la tasa de no-show no cambió (20,19%). Se renombraron las columnas y `discapacidad` se convirtió en binaria.

**Atributos construidos:**

| Grupo | Atributos |
|---|---|
| Temporales | `anticipacion_dias`, `mismo_dia`, `dia_semana` (sábado agrupado con viernes), `hora_agenda` |
| Demográficos | `mujer`, `grupo_edad` (solo para el modelo lineal) |
| Del paciente | `turnos_mismo_dia`, `turnos_previos`, `ausencias_previas`, `tasa_ausencias_previas`, `tiene_historial` |

**Corrección de fuga de información en el historial.** En el EDA, el "turno anterior" podía ser otro turno del mismo día, cuyo resultado está muy correlacionado con el actual. Eso inflaba el efecto (34,4% de no-show si el turno anterior fue ausencia). Con la definición estricta (solo días anteriores), el efecto real es:

| Historial del paciente | Turnos | Tasa de no-show |
|---|---:|---:|
| Sin historial | 66.286 | 19,4% |
| Historial sin ausencias | 30.119 | 18,8% |
| 1 ausencia previa | 10.472 | 25,0% |
| 2 o más ausencias previas | 3.639 | 32,1% |

El efecto lo dan las ausencias acumuladas, no el hecho de tener historial.

**Partición:**

| Conjunto | Turnos | Pacientes | Tasa de no-show |
|---|---:|---:|---:|
| Entrenamiento | 88.301 | 49.737 | 20,10% |
| Prueba | 22.215 | 12.559 | 20,55% |

Ningún paciente aparece en ambos conjuntos. La tasa de no-show de los 5 *folds* va de 19,55% a 20,32%.

**Relevancia de los atributos.** Se midió con información mutua corregida por cardinalidad (Miller-Madow), calculada solo sobre entrenamiento:

- Los atributos derivados de la anticipación (`anticipacion_dias`, `mismo_dia`) ocupan los dos primeros lugares, muy por encima del resto.
- Los atributos de historial superan al barrio y a todos los atributos clínicos originales.
- Alcoholismo, género, discapacidad y diabetes tienen relevancia prácticamente nula.

Se conservan todos los atributos; su aporte se evaluará con importancia por permutación.

**Trazabilidad.** *Hash* SHA-256 del dataset procesado: `40d361e54c4cd001e1c062e326c199206e44089db0d0ceb24d7879bd8a0413e6`.

### Modelado baseline y tratamiento del desbalance (dataset completo)

**Experimento 1: parámetros por defecto, sin tratamiento.** Validación cruzada con los 5 *folds* fijos.

| Modelo | PR-AUC (media ± desvío) | ROC-AUC | F1 con umbral óptimo | Brier |
|---|---:|---:|---:|---:|
| LightGBM | 0,381 ± 0,014 | 0,744 | 0,457 | 0,142 |
| XGBoost | 0,366 ± 0,009 | 0,735 | 0,450 | 0,144 |
| Regresión Logística | 0,366 ± 0,015 | 0,736 | 0,451 | 0,143 |
| Random Forest | 0,354 ± 0,010 | 0,724 | 0,444 | 0,147 |
| Árbol de Decisión | 0,233 ± 0,005 | 0,567 | 0,335 | 0,289 |
| Dummy (clase a priori) | 0,201 ± 0,003 | 0,500 | — | 0,161 |

- Todos los modelos superan al clasificador trivial. LightGBM supera a XGBoost y a la Regresión Logística en los 5 *folds*.
- Con umbral 0,5, el recall de los mejores modelos es menor al 8%: el efecto del desbalance anticipado en el EDA.
- El Árbol de Decisión y Random Forest, sin límite de profundidad, memorizan el entrenamiento. La Regresión Logística prácticamente no sobreajusta, y los modelos de *boosting* se ubican en un punto intermedio.

**Experimento 2: estrategias de desbalance.** Cada modelo se reentrenó con ponderación de clases, submuestreo aleatorio y SMOTE, y se comparó fold a fold contra "sin tratamiento":

| Estrategia | Δ PR-AUC (promedio entre modelos) | Δ Recall con umbral 0,5 | Δ Brier | Tiempo relativo |
|---|---:|---:|---:|---:|
| Ponderación de clases | −0,002 | +0,43 | +0,036 | 0,96× |
| Submuestreo aleatorio | −0,005 | +0,63 | +0,078 | 0,54× |
| SMOTE | −0,006 | +0,19 | +0,017 | 2,27× |

**Conclusiones del experimento 2:**

- **Ninguna estrategia mejora la PR-AUC.** En la Regresión Logística, Random Forest, LightGBM y XGBoost, las tres estrategias la reducen levemente, en la mayoría de los casos en todos o casi todos los *folds*.
- **Las estrategias solo desplazan las probabilidades.** El recall con umbral 0,5 sube mucho, pero el umbral que maximiza el F1 pasa de ~0,2 a ~0,5, y el F1 con umbral óptimo queda prácticamente igual. Es un efecto equivalente a bajar el umbral.
- **La calibración empeora** con ponderación y submuestreo.
- **SMOTE no funciona con atributos discretos.** Genera valores imposibles, como "diabetes = 0,4", y los modelos de árboles aprenden a reconocer los turnos sintéticos en lugar del patrón de ausentismo. Un clasificador distingue los sintéticos de los reales con una ROC-AUC cercana a 1, demostrado en la sección 4.3 del notebook.
- **Excepción: el Árbol de Decisión con submuestreo** mejora de forma pequeña pero consistente, sin un mecanismo concluyente identificado.

**Decisión:** el desbalance se resuelve **ajustando el umbral de decisión** (notebook 05), no remuestreando. En Optuna, la estrategia se incluye como hiperparámetro solo entre "sin tratamiento" y "ponderación de clases"; para el Árbol de Decisión, también el submuestreo.

**Candidatos seleccionados:**

| Rol | Modelo | Estrategia | PR-AUC |
|---|---|---|---:|
| Candidato a optimizar | LightGBM | Sin tratamiento | 0,381 |
| Candidato a optimizar | XGBoost | Sin tratamiento | 0,366 |
| Candidato a optimizar | Regresión Logística | Sin tratamiento | 0,366 |
| Referencia interpretable | Árbol de Decisión | Submuestreo aleatorio | 0,242 |

**Trazabilidad.** Se registraron 21 corridas en MLflow, todas verificadas con el mismo *hash* de dataset. Las capturas de la interfaz están en `docs/capturas/`.

*Las secciones de optimización y evaluación final se completan al finalizar las etapas correspondientes.*

---

## Registro de cambios

| Etapa | Cambios |
|---|---|
| Estructura inicial | Script `crear_estructura.ps1`, configuración (`config.py`), `requirements.txt` con versiones fijas, `.gitignore` y `.gitattributes` |
| Muestra de desarrollo | Carpeta `scripts/` con `crear_muestra.py` (muestreo estratificado por paciente con inclusión de anomalías) y reporte `muestra_resumen.md` |
| Análisis exploratorio | Notebook `01_EDA.ipynb` con interruptor `USAR_MUESTRA`; módulos `src/data.py` (carga de datos) y `src/reporting.py` (figuras, tablas y Markdown) |
| Decisiones metodológicas | Partición agrupada por paciente como estrategia principal (se descarta la temporal por el período corto); codificación del barrio por objetivo dentro de la validación cruzada; eliminación de edades implausibles (115) conservando 102 |
| Preparación de los datos | Notebook `02_preparacion.ipynb`. `src/data.py` suma limpieza y carga de procesados. `src/features.py` implementa atributos, historial sin fuga, partición con *folds* fijos y preprocesadores por familia. `src/reporting.py` suma tasas con IC de Wilson. Archivos con sufijo `_muestra` para separar muestra y completo. Información mutua con corrección de Miller-Madow |
| Modelado baseline | Notebook `03_modelado_baseline.ipynb` con experimentos `01_baseline` y `02_desbalance` en MLflow. Nuevos módulos `src/modeling.py`, `src/evaluation.py` y `src/tracking.py`. Control del *hash* del dataset al inicio. Selección de candidatos en `mod_candidatos.csv`. Capturas de MLflow en `docs/capturas/` |
| Correcciones del modelado | La PR-AUC de entrenamiento se calcula con la codificación del barrio que el modelo vio al ajustarse (antes se subestimaba el sobreajuste de los modelos de árboles). La duración de las corridas en MLflow incluye el entrenamiento. Nueva sección con la evidencia sobre SMOTE. Lecturas reescritas con los resultados del dataset completo. Se fijaron `sqlalchemy==2.0.36` y `alembic==1.14.0` por incompatibilidad con MLflow 2.19 |

---

## Referencias

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631. https://doi.org/10.1145/3292500.3330701
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS Inc.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357. https://doi.org/10.1613/jair.953
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
- Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations Newsletter*, 3(1), 27–32. https://doi.org/10.1145/507533.507538
- Miller, G. A. (1955). Note on the bias of information estimates. En H. Quastler (Ed.), *Information theory in psychology: Problems and methods* (pp. 95–100). Free Press.
- Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., Murching, S., Nykodym, T., Ogilvie, P., Parkhe, M., Xie, F., & Zumar, C. (2018). Accelerating the machine learning lifecycle with MLflow. *IEEE Data Engineering Bulletin*, 41(4), 39–45.

---

## Licencia y uso

Proyecto de carácter académico desarrollado para la Maestría en Ciencia de Datos de UCASAL. El dataset pertenece a sus autores originales y se utiliza según los términos publicados en Kaggle.
