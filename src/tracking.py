"""Utilidades de registro de experimentos en MLflow."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path

import mlflow
import pandas as pd

from src import config
from src.data import ruta_procesados

ARTEFACTOS_DIR = config.ROOT_DIR / "mlartifacts"


def configurar_experimento(clave: str, usar_muestra: bool = False, reiniciar: bool = False) -> str:
    """Configura MLflow (base SQLite en la raíz del repo) y activa el experimento.

    Los experimentos con la muestra llevan el sufijo '_muestra' para no mezclarse con
    los del dataset completo. Los artefactos se guardan en 'mlartifacts/'.
    Con reiniciar=True, las corridas existentes del experimento se marcan como eliminadas
    (borrado lógico de MLflow), para que una reejecución no duplique corridas.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    nombre = config.MLFLOW_EXPERIMENTS[clave] + ("_muestra" if usar_muestra else "")
    # La primera conexión crea la base SQLite y MLflow imprime sus migraciones: se silencian
    logging.disable(logging.INFO)
    try:
        experimento = mlflow.get_experiment_by_name(nombre)
    finally:
        logging.disable(logging.NOTSET)
    if experimento is None:
        ubicacion = (ARTEFACTOS_DIR / nombre).resolve()
        ubicacion.mkdir(parents=True, exist_ok=True)
        id_exp = mlflow.create_experiment(nombre, artifact_location=ubicacion.as_uri())
    else:
        id_exp = experimento.experiment_id
    mlflow.set_experiment(nombre)
    if reiniciar:
        anteriores = mlflow.search_runs(experiment_ids=[id_exp])
        for run_id in anteriores.get("run_id", []):
            mlflow.delete_run(run_id)
        if len(anteriores):
            print(f"Se archivaron {len(anteriores)} corridas anteriores del experimento {nombre}.")
    return id_exp


def _commit_git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT_DIR,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "sin_git"


def etiquetas_base(usar_muestra: bool = False) -> dict[str, str]:
    """Etiquetas comunes a todas las corridas: datos, hash del dataset, semilla y commit."""
    ruta_meta = ruta_procesados(usar_muestra).with_name(
        ruta_procesados(usar_muestra).stem.replace("turnos_procesados", "metadatos_preparacion") + ".json")
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    return {
        "datos": meta["datos"],
        "dataset_sha256": meta["sha256"],
        "semilla": str(config.SEED),
        "git_commit": _commit_git(),
    }


def registrar_corrida(nombre_corrida: str, parametros: dict, metricas: dict, etiquetas: dict,
                      tablas: dict[str, pd.DataFrame] | None = None, figuras: dict | None = None) -> str:
    """Registra parámetros, métricas y artefactos y devuelve el run_id.

    Si hay una corrida activa (por ejemplo, abierta antes de entrenar para que la duración
    registrada incluya el entrenamiento), se registra en ella; si no, se abre una nueva.
    """
    activa = mlflow.active_run()
    contexto = nullcontext(activa) if activa is not None else mlflow.start_run(run_name=nombre_corrida)
    with contexto as corrida:
        mlflow.set_tags(etiquetas)
        mlflow.log_params({k: (v if v is not None else "None") for k, v in parametros.items()})
        mlflow.log_metrics({k: float(v) for k, v in metricas.items()})
        with tempfile.TemporaryDirectory() as tmp:
            for nombre, tabla in (tablas or {}).items():
                ruta = Path(tmp) / f"{nombre}.csv"
                tabla.to_csv(ruta, index=False)
                mlflow.log_artifact(str(ruta))
        for nombre, fig in (figuras or {}).items():
            mlflow.log_figure(fig, f"{nombre}.png")
        return corrida.info.run_id


def exportar_corridas(claves: list[str], usar_muestra: bool = False) -> pd.DataFrame:
    """Devuelve todas las corridas de los experimentos indicados como DataFrame."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    nombres = [config.MLFLOW_EXPERIMENTS[c] + ("_muestra" if usar_muestra else "") for c in claves]
    return mlflow.search_runs(experiment_names=nombres)
