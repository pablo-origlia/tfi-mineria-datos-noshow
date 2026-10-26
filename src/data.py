"""Carga del dataset original y de los datos procesados, y funciones de limpieza."""

from __future__ import annotations

import pandas as pd

from src import config

SAMPLE_FILE = config.DATA_INTERIM / "muestra_noshow.csv"

# Nombres originales -> nombres en español, en minúsculas y sin errores de escritura
RENOMBRES = {
    "PatientId": "paciente_id",
    "AppointmentID": "turno_id",
    "Gender": "genero",
    "ScheduledDay": "fecha_agenda",
    "AppointmentDay": "fecha_turno",
    "Age": "edad",
    "Neighbourhood": "barrio",
    "Scholarship": "beca",
    "Hipertension": "hipertension",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholismo",
    "Handcap": "discapacidad",
    "SMS_received": "sms_recibido",
    "No-show": "no_show",
}

# Edad máxima plausible: se conserva 102 años y se elimina 115 (valor implausible repetido)
EDAD_MAXIMA = 110


def cargar_datos_crudos(usar_muestra: bool = False) -> pd.DataFrame:
    """Carga el dataset original o la muestra de desarrollo, sin modificaciones.

    PatientId se lee como texto para no alterar su formato original.
    """
    ruta = SAMPLE_FILE if usar_muestra else config.RAW_FILE
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. "
            + ("Generá la muestra con scripts/crear_muestra.py." if usar_muestra
               else "Descargá el dataset de Kaggle y copialo en data/raw/.")
        )
    return pd.read_csv(ruta, dtype={"PatientId": str})


def limpiar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica la limpieza definida en el EDA.

    Devuelve el dataset limpio y un registro con las filas eliminadas en cada paso.
    """
    d = df.rename(columns=RENOMBRES).copy()

    # Tipos: objetivo 0/1 y fechas sin zona horaria (la fecha del turno no tiene hora)
    d[config.TARGET] = (d[config.TARGET] == "Yes").astype(int)
    d["fecha_agenda"] = pd.to_datetime(d["fecha_agenda"], utc=True).dt.tz_localize(None)
    d["fecha_turno"] = pd.to_datetime(d["fecha_turno"], utc=True).dt.tz_localize(None).dt.normalize()

    anticipacion = (d["fecha_turno"] - d["fecha_agenda"].dt.normalize()).dt.days
    filtros = [
        ("Edad negativa", d["edad"] < 0,
         "Valor imposible"),
        (f"Edad mayor a {EDAD_MAXIMA} años", d["edad"] > EDAD_MAXIMA,
         "Valor implausible (115 años, repetido en varios registros)"),
        ("Turno anterior al agendamiento", anticipacion < 0,
         "Inconsistencia temporal: un turno no puede agendarse después de ocurrir"),
    ]

    registro = []
    eliminar = pd.Series(False, index=d.index)
    filas = len(d)
    for paso, mascara, justificacion in filtros:
        nuevas = mascara & ~eliminar
        eliminar |= mascara
        filas -= int(nuevas.sum())
        registro.append({"Paso": paso, "Filas eliminadas": int(nuevas.sum()),
                         "Filas restantes": filas, "Justificación": justificacion})

    d = d.loc[~eliminar].reset_index(drop=True)

    # Discapacidad: conteo 0-4 con valores 2-4 muy poco frecuentes -> indicador binario
    d["discapacidad"] = (d["discapacidad"] > 0).astype(int)

    return d, pd.DataFrame(registro)


def ruta_procesados(usar_muestra: bool = False):
    """Ruta del dataset procesado (archivo distinto para la muestra y para el completo)."""
    nombre = "turnos_procesados_muestra.csv" if usar_muestra else "turnos_procesados.csv"
    return config.DATA_PROCESSED / nombre


def cargar_procesados(usar_muestra: bool = False) -> pd.DataFrame:
    """Carga el dataset procesado por el notebook 02."""
    ruta = ruta_procesados(usar_muestra)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró {ruta}. Ejecutá primero el notebook 02_preparacion.")
    return pd.read_csv(ruta, dtype={"paciente_id": str, "barrio": str, "grupo_edad": str},
                       parse_dates=["fecha_agenda", "fecha_turno"])
