"""Espacios de búsqueda de hiperparámetros para la optimización con Optuna.

Cada espacio define los rangos que Optuna explora para un algoritmo. Los rangos se
eligieron con dos criterios:
- incluir los valores por defecto, que se encolan como primer trial (CONFIG_INICIAL),
  para que la optimización pueda, como mínimo, igualar al baseline;
- acotar la complejidad del modelo, que fue la principal fuente de sobreajuste
  detectada en el notebook 03 (profundidad, tamaño mínimo de hojas, regularización).

La estrategia de desbalance se trata como un hiperparámetro más. Según el
experimento 2 del notebook 03, solo se consideran "sin tratamiento" y "ponderación
de clases"; para el Árbol de Decisión se agrega el submuestreo, la única estrategia
que lo mejoró de forma consistente.
"""

from __future__ import annotations

import optuna

ESTRATEGIAS_POR_MODELO = {
    "lightgbm": ["ninguna", "pesos"],
    "xgboost": ["ninguna", "pesos"],
    "logistica": ["ninguna", "pesos"],
    "random_forest": ["ninguna", "pesos"],
    "arbol": ["ninguna", "pesos", "submuestreo"],
}

# Descripción legible de cada espacio (para el informe y la documentación)
DESCRIPCION_ESPACIOS = {
    "lightgbm": {
        "n_estimators": "100 a 1000 (log)", "learning_rate": "0,01 a 0,2 (log)",
        "num_leaves": "8 a 128 (log)", "min_child_samples": "10 a 300 (log)",
        "subsample": "0,5 a 1", "colsample_bytree": "0,4 a 1",
        "reg_alpha": "0,001 a 10 (log)", "reg_lambda": "0,001 a 10 (log)",
    },
    "xgboost": {
        "n_estimators": "100 a 1000 (log)", "learning_rate": "0,01 a 0,3 (log)",
        "max_depth": "3 a 10", "min_child_weight": "1 a 50 (log)",
        "subsample": "0,5 a 1", "colsample_bytree": "0,4 a 1",
        "reg_alpha": "0,001 a 10 (log)", "reg_lambda": "0,001 a 10 (log)",
    },
    "logistica": {
        "C": "0,001 a 10 (log)", "penalty": "l1 o l2",
    },
    "random_forest": {
        "n_estimators": "100 a 500 (log)", "max_depth": "4 a 30",
        "min_samples_leaf": "1 a 200 (log)", "max_features": "0,2 a 1",
    },
    "arbol": {
        "max_depth": "3 a 20", "min_samples_leaf": "5 a 500 (log)",
        "criterion": "gini o entropy",
    },
}


# Configuración de partida de cada estudio: los valores por defecto de cada algoritmo
# (los usados en el baseline del notebook 03). Se encola como primer trial para que la
# búsqueda parta del baseline y el resultado optimizado no pueda quedar por debajo de él.
# Aproximaciones necesarias para que los valores caigan dentro del espacio de búsqueda:
# - reg_alpha = 0 se representa como 0,001 (límite inferior de la escala logarítmica);
# - max_depth = None (sin límite) se representa con el máximo del rango;
# - max_features = "sqrt" en Random Forest equivale a sqrt(18) / 18 ≈ 0,236;
# - la Regresión Logística usa el solver liblinear (admite l1 y l2) en lugar de lbfgs.
CONFIG_INICIAL = {
    "lightgbm": {"estrategia": "ninguna", "n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31,
                 "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                 "reg_alpha": 1e-3, "reg_lambda": 1e-3},
    "xgboost": {"estrategia": "ninguna", "n_estimators": 100, "learning_rate": 0.3, "max_depth": 6,
                "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 1e-3, "reg_lambda": 1.0},
    "logistica": {"estrategia": "ninguna", "C": 1.0, "penalty": "l2"},
    "random_forest": {"estrategia": "ninguna", "n_estimators": 100, "max_depth": 30,
                      "min_samples_leaf": 1, "max_features": 0.236},
    "arbol": {"estrategia": "ninguna", "max_depth": 20, "min_samples_leaf": 5, "criterion": "gini"},
}


def sugerir_parametros(trial: optuna.Trial, modelo: str) -> tuple[str, dict]:
    """Devuelve (estrategia, parámetros del clasificador) sugeridos por Optuna para un trial."""
    estrategia = trial.suggest_categorical("estrategia", ESTRATEGIAS_POR_MODELO[modelo])

    if modelo == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,  # necesario para que LightGBM aplique el submuestreo de filas
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    elif modelo == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    elif modelo == "logistica":
        params = {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": "liblinear",  # admite penalización l1 y l2
        }
    elif modelo == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 200, log=True),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        }
    elif modelo == "arbol":
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 500, log=True),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }
    else:
        raise ValueError(f"Modelo sin espacio de búsqueda: {modelo}")

    return estrategia, params


def parametros_desde_trial(modelo: str, valores: dict) -> tuple[str, dict]:
    """Reconstruye (estrategia, parámetros) a partir de los valores de un trial (por ejemplo, el mejor).

    Se reutiliza `sugerir_parametros` con un trial fijo, de modo que los parámetros
    constantes (como subsample_freq o solver) se agregan exactamente igual que durante
    la búsqueda.
    """
    return sugerir_parametros(optuna.trial.FixedTrial(valores), modelo)
