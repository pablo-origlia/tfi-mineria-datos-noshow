"""Métricas, validación cruzada con folds fijos y ajuste de umbral."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)

# Nombre interno (usado en MLflow) -> nombre para tablas y gráficos
NOMBRES_METRICAS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "recall": "Recall (no-show)",
    "precision": "Precisión (no-show)",
    "f1": "F1 (no-show)",
    "balanced_accuracy": "Exactitud balanceada",
    "accuracy": "Exactitud",
    "brier": "Brier",
}


def calcular_metricas(y_true, proba, umbral: float = 0.5) -> dict[str, float]:
    """Métricas independientes del umbral (PR-AUC, ROC-AUC, Brier) y con umbral (resto)."""
    pred = (proba >= umbral).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
    }


def evaluar_cv(modelo, X: pd.DataFrame, y: pd.Series, folds: np.ndarray):
    """Validación cruzada con los folds fijos del dataset.

    Devuelve:
    - DataFrame con las métricas de validación de cada fold, la PR-AUC sobre los datos de
      entrenamiento del fold (para detectar sobreajuste) y el tiempo de ajuste.
    - Probabilidades fuera de fold (OOF): cada turno predicho por un modelo que no lo vio.
    """
    y = np.asarray(y)
    oof = np.zeros(len(y), dtype=float)
    filas = []
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        m = clone(modelo)
        inicio = time.perf_counter()
        m.fit(X.iloc[tr], y[tr])
        segundos = time.perf_counter() - inicio
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        pr_auc_train = average_precision_score(y[tr], proba_entrenamiento(m, X.iloc[tr], y[tr]))
        filas.append({"fold": int(k), **calcular_metricas(y[va], oof[va]),
                      "pr_auc_train": pr_auc_train, "segundos_ajuste": segundos})
    return pd.DataFrame(filas), oof


def proba_entrenamiento(pipeline, X_tr: pd.DataFrame, y_tr) -> np.ndarray:
    """Probabilidades sobre los datos de entrenamiento tal como el clasificador los vio al ajustarse.

    El TargetEncoder codifica el barrio con validación cruzada interna al ajustarse
    (fit_transform), pero con las tasas de todo el entrenamiento al predecir (transform).
    Si se midiera el desempeño de entrenamiento con `pipeline.predict_proba`, los modelos de
    árboles recibirían una codificación distinta de la que usaron para aprender sus cortes y
    el sobreajuste quedaría subestimado. Por eso se reconstruye la codificación de ajuste
    (el preprocesador es determinístico: misma semilla, mismos folds internos).
    Se usan los datos reales de entrenamiento, sin remuestrear.
    """
    preprocesador = clone(pipeline.steps[0][1])
    clasificador = pipeline.steps[-1][1]
    return clasificador.predict_proba(preprocesador.fit_transform(X_tr, y_tr))[:, 1]


def resumir_folds(por_fold: pd.DataFrame) -> dict[str, float]:
    """Media y desvío estándar de cada métrica entre folds."""
    res = {}
    for col in por_fold.columns.drop("fold"):
        res[f"{col}_media"] = por_fold[col].mean()
        res[f"{col}_desvio"] = por_fold[col].std(ddof=1)
    return res


def umbral_f1_optimo(y_true, proba) -> tuple[float, float]:
    """Umbral que maximiza el F1 de la clase positiva y el F1 alcanzado."""
    precision, recall, umbrales = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    i = int(np.argmax(f1[:-1]))
    return float(umbrales[i]), float(f1[i])


# ---------------------------------------------------------------------------
# Umbral de decisión y evaluación final con intervalos de confianza
# ---------------------------------------------------------------------------

def umbral_para_recall(y_true, proba, recall_minimo: float) -> float:
    """Mayor umbral que alcanza al menos el recall indicado (maximiza la precisión con ese recall)."""
    precision, recall, umbrales = precision_recall_curve(y_true, proba)
    validos = np.where(recall[:-1] >= recall_minimo)[0]
    return float(umbrales[validos[-1]]) if len(validos) else float(umbrales[0])


def umbral_para_capacidad(proba, fraccion: float) -> float:
    """Umbral que marca como riesgo a la fracción indicada de turnos (los de mayor probabilidad)."""
    return float(np.quantile(proba, 1 - fraccion))


def metricas_ponderadas(y_true, proba, umbral: float, pesos=None) -> dict[str, float]:
    """Métricas con pesos por fila (los pesos permiten un bootstrap rápido sin copiar datos)."""
    pred = (proba >= umbral).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, proba, sample_weight=pesos),
        "roc_auc": roc_auc_score(y_true, proba, sample_weight=pesos),
        "recall": recall_score(y_true, pred, sample_weight=pesos, zero_division=0),
        "precision": precision_score(y_true, pred, sample_weight=pesos, zero_division=0),
        "f1": f1_score(y_true, pred, sample_weight=pesos, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred, sample_weight=pesos),
        "brier": brier_score_loss(y_true, proba, sample_weight=pesos),
    }


def bootstrap_por_grupo(y_true, probas: dict, umbrales: dict, grupos, n: int = 1000, seed: int = 42):
    """Bootstrap agrupado por paciente sobre el conjunto de prueba.

    En cada réplica se remuestrean pacientes con reposición; cada turno recibe como peso la
    cantidad de veces que salió sorteado su paciente. Todos los modelos se evalúan sobre las
    mismas réplicas, lo que permite comparar diferencias pareadas.
    Devuelve un DataFrame largo con columnas: replica, modelo, métrica, valor.
    """
    y_true = np.asarray(y_true)
    codigos, indice = np.unique(np.asarray(grupos), return_inverse=True)
    rng = np.random.default_rng(seed)
    filas = []
    for r in range(n):
        conteo = rng.multinomial(len(codigos), np.full(len(codigos), 1 / len(codigos)))
        pesos = conteo[indice]
        for nombre, proba in probas.items():
            for metrica, valor in metricas_ponderadas(y_true, proba, umbrales[nombre], pesos).items():
                filas.append((r, nombre, metrica, valor))
    return pd.DataFrame(filas, columns=["replica", "modelo", "metrica", "valor"])
