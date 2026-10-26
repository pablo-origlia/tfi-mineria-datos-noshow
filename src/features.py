"""Ingeniería de atributos, partición de los datos y preprocesamiento por familia de modelos.

Regla general para evitar fuga de información:
- Los atributos que se calculan fila a fila o con información del pasado se crean
  antes de la partición (no usan datos de otros turnos futuros ni el objetivo actual).
- Las transformaciones que aprenden de los datos (codificación por objetivo, escalado)
  se ajustan dentro del pipeline de cada modelo, solo con datos de entrenamiento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

from src import config

GRUPO = "paciente_id"

# Discretización de la edad (mismos tramos del EDA; límites fijos, no aprendidos de los datos)
BINS_EDAD = [-np.inf, 0, 5, 12, 18, 30, 45, 60, 75, np.inf]
ETIQUETAS_EDAD = ["<1", "1-5", "6-12", "13-18", "19-30", "31-45", "46-60", "61-75", "76+"]
DIAS_SEMANA = [0, 1, 2, 3, 4]  # lunes a viernes (el sábado se agrupa con el viernes)

# --- Conjuntos de atributos ---
NUMERICAS = ["edad", "anticipacion_dias", "hora_agenda", "turnos_mismo_dia",
             "turnos_previos", "ausencias_previas", "tasa_ausencias_previas"]
BINARIAS = ["mujer", "beca", "hipertension", "diabetes", "alcoholismo", "discapacidad",
            "sms_recibido", "mismo_dia", "tiene_historial"]
CATEGORICAS = ["dia_semana", "grupo_edad", "barrio"]

ATRIBUTOS_ARBOL = NUMERICAS + BINARIAS + ["dia_semana", "barrio"]
ATRIBUTOS_LINEAL = [c for c in NUMERICAS if c != "edad"] + BINARIAS + CATEGORICAS


# ---------------------------------------------------------------------------
# Ingeniería de atributos
# ---------------------------------------------------------------------------

def crear_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Crea los atributos derivados a partir del dataset limpio."""
    d = df.copy()

    # Temporales
    d["anticipacion_dias"] = (d["fecha_turno"] - d["fecha_agenda"].dt.normalize()).dt.days
    d["mismo_dia"] = (d["anticipacion_dias"] == 0).astype(int)
    d["dia_semana"] = d["fecha_turno"].dt.dayofweek.clip(upper=4)  # sábado (5) -> viernes (4)
    d["hora_agenda"] = d["fecha_agenda"].dt.hour

    # Demográficos
    d["mujer"] = (d["genero"] == "F").astype(int)
    d["grupo_edad"] = pd.cut(d["edad"], bins=BINS_EDAD, labels=ETIQUETAS_EDAD).astype(str)

    # Turnos del mismo paciente en el mismo día
    d["turnos_mismo_dia"] = d.groupby([GRUPO, "fecha_turno"])["turno_id"].transform("size")

    return agregar_historial(d)


def agregar_historial(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el historial del paciente usando solo turnos de días estrictamente anteriores.

    Los turnos del mismo día no se cuentan entre sí, porque su resultado no se conoce
    al momento de la predicción.
    """
    por_dia = (
        df.groupby([GRUPO, "fecha_turno"], as_index=False)
        .agg(n=("turno_id", "size"), ausencias=(config.TARGET, "sum"))
        .sort_values([GRUPO, "fecha_turno"])
    )
    por_paciente = por_dia.groupby(GRUPO)
    por_dia["turnos_previos"] = por_paciente["n"].cumsum() - por_dia["n"]
    por_dia["ausencias_previas"] = por_paciente["ausencias"].cumsum() - por_dia["ausencias"]

    d = df.merge(por_dia[[GRUPO, "fecha_turno", "turnos_previos", "ausencias_previas"]],
                 on=[GRUPO, "fecha_turno"], how="left", validate="many_to_one")
    d["tiene_historial"] = (d["turnos_previos"] > 0).astype(int)
    d["tasa_ausencias_previas"] = np.where(
        d["turnos_previos"] > 0, d["ausencias_previas"] / d["turnos_previos"].clip(lower=1), 0.0
    )
    return d


# ---------------------------------------------------------------------------
# Partición
# ---------------------------------------------------------------------------

def asignar_particiones(df: pd.DataFrame, n_folds: int = 5, seed: int = config.SEED) -> pd.DataFrame:
    """Asigna entrenamiento/prueba (80/20) y folds de validación cruzada.

    Ambas divisiones son estratificadas por la variable objetivo y agrupadas por paciente:
    todos los turnos de un paciente quedan del mismo lado.
    Columnas agregadas: 'particion' ('train'/'test') y 'fold' (0..n_folds-1; -1 en prueba).
    """
    d = df.copy()
    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, idx_test = next(divisor.split(d, d[config.TARGET], d[GRUPO]))
    d["particion"] = "train"
    d.loc[d.index[idx_test], "particion"] = "test"

    d["fold"] = -1
    train = d[d["particion"] == "train"]
    divisor_cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, idx_val) in enumerate(divisor_cv.split(train, train[config.TARGET], train[GRUPO])):
        d.loc[train.index[idx_val], "fold"] = k
    return d


def iterador_folds(train: pd.DataFrame):
    """Genera (índices de entrenamiento, índices de validación) a partir de la columna 'fold'.

    Se pasa como 'cv' a las funciones de scikit-learn para usar siempre los mismos folds.
    """
    folds = train["fold"].to_numpy()
    for k in np.unique(folds):
        yield np.where(folds != k)[0], np.where(folds == k)[0]


# ---------------------------------------------------------------------------
# Preprocesamiento por familia de modelos
# ---------------------------------------------------------------------------

def construir_preprocesador(familia: str = "arbol") -> ColumnTransformer:
    """Devuelve el preprocesador adecuado para cada familia de modelos.

    - 'arbol' (Árbol, Random Forest, LightGBM, XGBoost): atributos numéricos sin escalar
      y barrio codificado por objetivo.
    - 'lineal' (Regresión Logística): logaritmo y escalado de atributos asimétricos,
      edad discretizada en tramos, one-hot del día de la semana y barrio codificado por
      objetivo y escalado.

    El TargetEncoder usa validación cruzada interna al ajustarse, lo que evita que
    cada turno se codifique con su propio resultado.
    """
    barrio = TargetEncoder(target_type="binary", random_state=config.SEED)

    if familia == "arbol":
        return ColumnTransformer(
            [("directo", "passthrough", NUMERICAS + BINARIAS + ["dia_semana"]),
             ("barrio", barrio, ["barrio"])],
            verbose_feature_names_out=False,
        )

    if familia == "lineal":
        log_y_escala = Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("escala", StandardScaler()),
        ])
        return ColumnTransformer(
            [("log", log_y_escala, ["anticipacion_dias", "turnos_mismo_dia",
                                    "turnos_previos", "ausencias_previas"]),
             ("escala", StandardScaler(), ["hora_agenda", "tasa_ausencias_previas"]),
             ("onehot", OneHotEncoder(categories=[ETIQUETAS_EDAD, DIAS_SEMANA], drop="first",
                                      sparse_output=False), ["grupo_edad", "dia_semana"]),
             ("binarias", "passthrough", BINARIAS),
             ("barrio", Pipeline([("objetivo", barrio), ("escala", StandardScaler())]), ["barrio"])],
            verbose_feature_names_out=False,
        )

    raise ValueError("familia debe ser 'arbol' o 'lineal'")
