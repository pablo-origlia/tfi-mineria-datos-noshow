"""Configuración central del proyecto: rutas, semilla y constantes globales."""

from pathlib import Path

# --- Rutas ---
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_INTERIM = ROOT_DIR / "data" / "interim"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
TABLES_DIR = ROOT_DIR / "reports" / "tables"

RAW_FILE = DATA_RAW / "KaggleV2-May-2016.csv"

# --- Reproducibilidad ---
SEED = 42

# --- Variable objetivo ---
TARGET = "no_show"
POSITIVE_CLASS = 1  # 1 = el paciente NO asistió al turno

# --- MLflow ---
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENTS = {
    "baseline": "01_baseline",
    "desbalance": "02_desbalance",
    "optuna": "03_optuna",
    "final": "04_modelo_final",
}