"""Utilidades para guardar figuras y tablas y generar resúmenes en Markdown."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src import config


def guardar_figura(fig: plt.Figure, nombre: str) -> None:
    """Guarda una figura en reports/figures/<nombre>.png."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")


def guardar_tabla(df: pd.DataFrame, nombre: str, index: bool = False) -> None:
    """Guarda una tabla en reports/tables/<nombre>.csv (UTF-8, compatible con Excel)."""
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TABLES_DIR / f"{nombre}.csv", index=index, encoding="utf-8-sig")


def tabla_markdown(df: pd.DataFrame, index: bool = False, decimales: int = 4) -> str:
    """Convierte un DataFrame en una tabla Markdown sin dependencias externas."""
    datos = df.reset_index() if index else df

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.{decimales}g}" if abs(v) < 1e-3 and v != 0 else f"{v:,.{decimales}f}".rstrip("0").rstrip(".")
        return str(v)

    encabezado = "| " + " | ".join(map(str, datos.columns)) + " |"
    separador = "|" + "|".join(["---"] * len(datos.columns)) + "|"
    filas = ["| " + " | ".join(fmt(v) for v in fila) + " |" for fila in datos.itertuples(index=False)]
    return "\n".join([encabezado, separador, *filas])


def tasa_con_ic(exitos, n, z: float = 1.96):
    """Proporción con intervalo de confianza de Wilson (95% por defecto)."""
    import numpy as np

    exitos, n = np.asarray(exitos, float), np.asarray(n, float)
    p = exitos / n
    denom = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / denom
    margen = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, centro - margen, centro + margen


def tasa_por(data: pd.DataFrame, col: str, objetivo: str = "no_show", orden=None) -> pd.DataFrame:
    """Tabla de tasa del objetivo por categoría, con n e intervalo de confianza."""
    g = data.groupby(col, observed=True)[objetivo].agg(n="size", ausencias="sum")
    if orden is not None:
        g = g.reindex(orden)
    g["tasa"], g["ic_inf"], g["ic_sup"] = tasa_con_ic(g["ausencias"], g["n"])
    return g
