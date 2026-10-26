"""Definición de los modelos y de las estrategias de tratamiento del desbalance."""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import config
from src.features import ATRIBUTOS_ARBOL, ATRIBUTOS_LINEAL, construir_preprocesador

SEED = config.SEED

# Nombre del modelo -> (familia de preprocesamiento, nombre para mostrar)
MODELOS = {
    "dummy": ("arbol", "Dummy (clase a priori)"),
    "logistica": ("lineal", "Regresión Logística"),
    "arbol": ("arbol", "Árbol de Decisión"),
    "random_forest": ("arbol", "Random Forest"),
    "lightgbm": ("arbol", "LightGBM"),
    "xgboost": ("arbol", "XGBoost"),
}

ESTRATEGIAS = {
    "ninguna": "Sin tratamiento",
    "pesos": "Ponderación de clases",
    "submuestreo": "Submuestreo aleatorio",
    "smote": "SMOTE",
}


def crear_clasificador(nombre: str, estrategia: str = "ninguna", ratio_clases: float = 4.0, **params):
    """Crea el clasificador con parámetros por defecto (salvo semilla, paralelismo y verbosidad).

    Con estrategia 'pesos' se activa la ponderación de clases propia de cada algoritmo.
    `ratio_clases` (negativos / positivos en entrenamiento) se usa en XGBoost.
    """
    pesos = estrategia == "pesos"
    if nombre == "dummy":
        return DummyClassifier(strategy="prior")
    if nombre == "logistica":
        return LogisticRegression(max_iter=2000, class_weight="balanced" if pesos else None,
                                  random_state=SEED, **params)
    if nombre == "arbol":
        return DecisionTreeClassifier(class_weight="balanced" if pesos else None, random_state=SEED, **params)
    if nombre == "random_forest":
        return RandomForestClassifier(class_weight="balanced" if pesos else None, n_jobs=-1,
                                      random_state=SEED, **params)
    if nombre == "lightgbm":
        return LGBMClassifier(class_weight="balanced" if pesos else None, random_state=SEED,
                              n_jobs=-1, verbose=-1, **params)
    if nombre == "xgboost":
        return XGBClassifier(scale_pos_weight=ratio_clases if pesos else 1.0, random_state=SEED,
                             n_jobs=-1, eval_metric="logloss", **params)
    raise ValueError(f"Modelo desconocido: {nombre}")


def crear_pipeline(nombre: str, estrategia: str = "ninguna", ratio_clases: float = 4.0, **params) -> Pipeline:
    """Pipeline completo: preprocesamiento -> (remuestreo) -> clasificador.

    El remuestreo (submuestreo o SMOTE) se aplica solo al ajustar, es decir, solo sobre
    los datos de entrenamiento de cada fold. Las predicciones de validación y de prueba
    se hacen siempre sobre datos reales sin remuestrear.
    """
    familia = MODELOS[nombre][0]
    pasos = [("preprocesador", construir_preprocesador(familia))]
    if estrategia == "submuestreo":
        pasos.append(("remuestreo", RandomUnderSampler(random_state=SEED)))
    elif estrategia == "smote":
        pasos.append(("remuestreo", SMOTE(random_state=SEED)))
    pasos.append(("clasificador", crear_clasificador(nombre, estrategia, ratio_clases, **params)))
    return Pipeline(pasos)


def atributos_de(nombre: str) -> list[str]:
    """Lista de atributos de entrada según la familia del modelo."""
    return ATRIBUTOS_LINEAL if MODELOS[nombre][0] == "lineal" else ATRIBUTOS_ARBOL
