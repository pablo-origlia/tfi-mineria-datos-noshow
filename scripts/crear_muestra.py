"""Genera una muestra representativa del dataset original de no-show.

La muestra se usa para desarrollar y probar el código sin procesar el dataset
completo. Los resultados del informe se obtienen siempre con el dataset completo.

Criterios de muestreo:
- Se seleccionan PACIENTES completos (todos sus turnos), no filas sueltas,
  para conservar el historial de cada paciente.
- Muestreo estratificado por (tuvo alguna ausencia) x (cantidad de turnos),
  para conservar la proporción de no-show y la estructura de turnos repetidos.
- Se incluyen siempre los pacientes con anomalías conocidas (edad fuera de
  rango, Handcap > 1, turno anterior al agendamiento), para poder probar el
  código de limpieza.

Uso (desde la raíz del repositorio):
    python scripts/crear_muestra.py
    python scripts/crear_muestra.py --frac 0.10 --seed 42
    python scripts/crear_muestra.py --sin-anomalias
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import config  # noqa: E402

COLUMNAS_ESPERADAS = [
    "PatientId", "AppointmentID", "Gender", "ScheduledDay", "AppointmentDay",
    "Age", "Neighbourhood", "Scholarship", "Hipertension", "Diabetes",
    "Alcoholism", "Handcap", "SMS_received", "No-show",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crea una muestra estratificada por paciente.")
    parser.add_argument("--entrada", type=Path, default=config.RAW_FILE,
                        help="CSV original (por defecto: data/raw/KaggleV2-May-2016.csv)")
    parser.add_argument("--salida", type=Path, default=config.DATA_INTERIM / "muestra_noshow.csv",
                        help="CSV de salida (por defecto: data/interim/muestra_noshow.csv)")
    parser.add_argument("--frac", type=float, default=0.15,
                        help="Fracción de pacientes a muestrear (por defecto: 0.15)")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Semilla aleatoria")
    parser.add_argument("--sin-anomalias", action="store_true",
                        help="No forzar la inclusión de pacientes con anomalías")
    return parser.parse_args()


def cargar(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        sys.exit(f"ERROR: no se encontró el archivo {ruta}")
    # PatientId como texto para no alterar su formato al volver a guardar
    df = pd.read_csv(ruta, dtype={"PatientId": str})
    faltantes = set(COLUMNAS_ESPERADAS) - set(df.columns)
    if faltantes:
        sys.exit(f"ERROR: faltan columnas en el CSV: {sorted(faltantes)}")
    return df


def marcar_anomalias(df: pd.DataFrame) -> pd.Series:
    """Devuelve una máscara booleana de filas con anomalías conocidas."""
    agendado = pd.to_datetime(df["ScheduledDay"], utc=True).dt.normalize()
    turno = pd.to_datetime(df["AppointmentDay"], utc=True).dt.normalize()
    return (
        (df["Age"] < 0)
        | (df["Age"] > 100)
        | (df["Handcap"] > 1)
        | (turno < agendado)
    )


def estratos_por_paciente(df: pd.DataFrame) -> pd.DataFrame:
    pac = df.groupby("PatientId").agg(
        n_turnos=("AppointmentID", "size"),
        alguna_ausencia=("No-show", lambda s: (s == "Yes").any()),
    )
    pac["tramo_turnos"] = pd.cut(pac["n_turnos"], bins=[0, 1, 2, 5, float("inf")],
                                 labels=["1", "2", "3-5", "6+"])
    pac["estrato"] = pac["alguna_ausencia"].astype(str) + "_" + pac["tramo_turnos"].astype(str)
    return pac


def resumen(df: pd.DataFrame) -> dict[str, str]:
    agendado = pd.to_datetime(df["ScheduledDay"], utc=True)
    turno = pd.to_datetime(df["AppointmentDay"], utc=True)
    anticipacion = (turno.dt.normalize() - agendado.dt.normalize()).dt.days
    return {
        "Filas (turnos)": f"{len(df):,}",
        "Pacientes únicos": f"{df['PatientId'].nunique():,}",
        "Turnos por paciente (media)": f"{len(df) / df['PatientId'].nunique():.2f}",
        "% no-show": f"{(df['No-show'] == 'Yes').mean() * 100:.2f}%",
        "% mujeres": f"{(df['Gender'] == 'F').mean() * 100:.2f}%",
        "Edad media": f"{df['Age'].mean():.2f}",
        "% SMS recibido": f"{df['SMS_received'].mean() * 100:.2f}%",
        "% Scholarship": f"{df['Scholarship'].mean() * 100:.2f}%",
        "Anticipación media (días)": f"{anticipacion.mean():.2f}",
        "Barrios distintos": f"{df['Neighbourhood'].nunique()}",
        "Rango de fechas de turno": f"{turno.min().date()} a {turno.max().date()}",
        "Filas con edad < 0": f"{(df['Age'] < 0).sum()}",
        "Filas con edad > 100": f"{(df['Age'] > 100).sum()}",
        "Filas con Handcap > 1": f"{(df['Handcap'] > 1).sum()}",
        "Filas con turno anterior al agendamiento": f"{(anticipacion < 0).sum()}",
    }


def tabla_markdown(original: dict[str, str], muestra: dict[str, str]) -> str:
    lineas = ["| Indicador | Original | Muestra |", "|---|---:|---:|"]
    for clave in original:
        lineas.append(f"| {clave} | {original[clave]} | {muestra[clave]} |")
    return "\n".join(lineas)


def main() -> None:
    args = parse_args()
    if not 0 < args.frac <= 1:
        sys.exit("ERROR: --frac debe estar entre 0 y 1.")

    df = cargar(args.entrada)
    pac = estratos_por_paciente(df)

    seleccion = (
        pac.groupby("estrato", group_keys=False, observed=True)
        .sample(frac=args.frac, random_state=args.seed)
        .index
    )
    ids = set(seleccion)

    n_forzados = 0
    if not args.sin_anomalias:
        anomalos = set(df.loc[marcar_anomalias(df), "PatientId"])
        n_forzados = len(anomalos - ids)
        ids |= anomalos

    muestra = df[df["PatientId"].isin(ids)].copy()

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    muestra.to_csv(args.salida, index=False)

    # --- Reporte de representatividad ---
    tabla = tabla_markdown(resumen(df), resumen(muestra))
    reporte = "\n".join([
        "# Reporte de la muestra de desarrollo",
        "",
        "## Parámetros",
        "",
        f"- **Archivo original:** `{args.entrada.name}`",
        f"- **Fracción de pacientes:** {args.frac}",
        f"- **Semilla:** {args.seed}",
        f"- **Estratos:** {pac['estrato'].nunique()} (alguna ausencia × tramo de turnos)",
        f"- **Pacientes agregados por anomalías:** {n_forzados}",
        f"- **Tamaño del archivo de salida:** {args.salida.stat().st_size / 1024 ** 2:.2f} MB",
        "",
        "## Comparación original vs. muestra",
        "",
        tabla,
        "",
    ])
    ruta_reporte = config.TABLES_DIR / "muestra_resumen.md"
    ruta_reporte.parent.mkdir(parents=True, exist_ok=True)
    ruta_reporte.write_text(reporte, encoding="utf-8")

    print(reporte)
    print(f"Muestra guardada en: {args.salida}")
    print(f"Reporte guardado en: {ruta_reporte}")


if __name__ == "__main__":
    main()
