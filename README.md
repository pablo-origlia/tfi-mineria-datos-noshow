# Predicción de ausentismo a turnos médicos (*No-Show*)

## Descripción del proyecto

El ausentismo de pacientes a turnos médicos ya agendados (*no-show*) genera huecos de agenda que no pueden reasignarse a tiempo. También aumenta las listas de espera y los costos operativos, y reduce la eficiencia en el uso de personal, equipamiento e insumos.

Este proyecto construye y compara **modelos de clasificación** que estiman, a partir de datos históricos, la probabilidad de que un paciente **no asista** a su turno. Con esa estimación anticipada, un centro de salud puede intervenir de forma proactiva con recordatorios adicionales, sobreturnos controlados o reprogramaciones.

| Aspecto | Detalle |
|---|---|
| **Tarea de minería de datos** | Clasificación binaria |
| **Variable objetivo** | `No-show` (1 = no asistió, 0 = asistió) |
| **Metodología** | CRISP-DM |
| **Desafío principal** | Desbalance de clases (~80% asiste / ~20% no asiste) |

---

## Conjunto de datos

| Campo | Valor |
|---|---|
| **Nombre** | Medical Appointment No Shows |
| **Fuente** | Kaggle — [joniarroba/noshowappointments](https://www.kaggle.com/datasets/joniarroba/noshowappointments) |
| **Origen** | Registros públicos de turnos médicos de Vitória, Brasil (2016) |
| **Instancias** | 110.527 turnos |
| **Atributos** | 14 (13 predictores candidatos + variable objetivo) |

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

**Desbalance de clases.** Se analiza antes de entrenar. Un clasificador trivial que siempre predice "asiste" alcanza ~80% de exactitud, por lo que la exactitud **no** es la métrica principal. Se comparan cuatro estrategias aplicadas **solo sobre entrenamiento**: sin tratamiento, ponderación de clases (`class_weight`), submuestreo aleatorio y SMOTE.

**Métricas.** Se priorizan:

- PR-AUC;
- ROC-AUC;
- recall y F1 de la clase *no-show*;
- exactitud balanceada.

Además se ajusta el umbral de decisión sobre validación.

**Prevención de fuga de información (*data leakage*).** Un mismo paciente tiene varios turnos, por lo que la partición respeta la estructura temporal o agrupada por paciente. Los atributos de historial se calculan solo con turnos estrictamente anteriores.

**Evaluación honesta.** La validación cruzada y la optimización de hiperparámetros usan solo el conjunto de entrenamiento. El conjunto de prueba se utiliza **una única vez**, al final.

### Modelos

| Modelo | Rol |
|---|---|
| `DummyClassifier` | Línea base de referencia |
| Regresión Logística | Referencia lineal |
| Árbol de Decisión | Interpretabilidad y extracción de reglas |
| Random Forest | Ensamble por *bagging* |
| LightGBM / XGBoost | Ensambles por *gradient boosting* |

### Optimización y trazabilidad

**Optuna** optimiza los hiperparámetros de los mejores candidatos. Usa el muestreador TPE con semilla fija, poda con `MedianPruner` y PR-AUC como métrica objetivo. La estrategia de desbalance se trata como un hiperparámetro más.

**MLflow** registra cada corrida: parámetros, métricas, gráficos, modelo, semilla y *hash* del dataset. Esto permite comparar modelos de forma trazable y reproducible.

| Experimento MLflow | Contenido |
|---|---|
| `01_baseline` | Todos los modelos con parámetros por defecto |
| `02_desbalance` | Modelos × estrategias de tratamiento del desbalance |
| `03_optuna` | Estudios de optimización (corridas anidadas por *trial*) |
| `04_modelo_final` | Modelo seleccionado, umbral y evaluación en prueba |

---

## Estructura del repositorio

```text
tfi-mineria-datos-noshow/
├── data/
│   ├── raw/              # Dataset original (no versionado)
│   ├── interim/          # Datos intermedios
│   └── processed/        # Datos listos para modelar
├── docs/
│   ├── propuesta/        # Propuesta de TFI aprobada
│   └── capturas/         # Capturas de MLflow y otras evidencias
├── models/               # Modelos serializados (no versionados)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preparacion.ipynb
│   ├── 03_modelado_baseline.ipynb
│   ├── 04_optimizacion_optuna.ipynb
│   └── 05_evaluacion_final.ipynb
├── reports/
│   ├── figures/          # Gráficos generados por los notebooks
│   ├── tables/           # Tablas exportadas (comparaciones, métricas)
│   └── informe_final.md  # Informe del TFI (se exporta a PDF)
├── src/
│   ├── __init__.py
│   ├── config.py         # Rutas, semilla y constantes globales
│   ├── data.py           # Carga y limpieza
│   ├── features.py       # Ingeniería de atributos
│   ├── evaluation.py     # Métricas y gráficos de evaluación
│   └── tracking.py       # Utilidades de registro en MLflow
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md
```

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

### 2. Descargar el dataset

1. Descargar el dataset desde [Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments).
2. Copiar el archivo `KaggleV2-May-2016.csv` en `data/raw/`.

### 3. Ejecutar los notebooks en orden

```powershell
jupyter lab
```

| Orden | Notebook | Fase CRISP-DM |
|---|---|---|
| 1 | `01_EDA.ipynb` | Comprensión de los datos |
| 2 | `02_preparacion.ipynb` | Preparación de los datos |
| 3 | `03_modelado_baseline.ipynb` | Modelado |
| 4 | `04_optimizacion_optuna.ipynb` | Modelado (optimización) |
| 5 | `05_evaluacion_final.ipynb` | Evaluación |

### 4. Consultar los experimentos en MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Luego abrir <http://127.0.0.1:5000> en el navegador.

---

## Reproducibilidad

- Semilla global fija (`SEED = 42`) definida en `src/config.py`.
- Versiones de librerías fijadas en `requirements.txt`.
- Cada corrida de MLflow registra parámetros, semilla y *hash* del dataset utilizado.

---

## Estado del proyecto

- [x] Propuesta de TFI aprobada
- [x] Estructura del repositorio
- [ ] Análisis exploratorio de datos
- [ ] Limpieza y preparación
- [ ] Modelado *baseline* y tratamiento del desbalance
- [ ] Optimización con Optuna
- [ ] Evaluación final y selección del modelo
- [ ] Informe final

---

## Resultados

*Sección a completar al finalizar la evaluación de los modelos.*

---

## Referencias

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631. https://doi.org/10.1145/3292500.3330701
- Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS Inc.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357. https://doi.org/10.1613/jair.953
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
- Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., Murching, S., Nykodym, T., Ogilvie, P., Parkhe, M., Xie, F., & Zumar, C. (2018). Accelerating the machine learning lifecycle with MLflow. *IEEE Data Engineering Bulletin*, 41(4), 39–45.

---

## Licencia y uso

Proyecto de carácter académico desarrollado para la Maestría en Ciencia de Datos de UCASAL. El dataset pertenece a sus autores originales y se utiliza según los términos publicados en Kaggle.
