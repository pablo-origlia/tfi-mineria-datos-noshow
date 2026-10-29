"""Verifica que todas las salidas del proyecto estén completas y sean consistentes.

Revisa, etapa por etapa:
- versiones instaladas frente a requirements.txt;
- archivos esperados (figuras, tablas, dataset procesado, metadatos);
- notebooks ejecutados por completo, sin errores y con el dataset completo;
- integridad del dataset procesado (hash, particiones, folds);
- corridas de MLflow (cantidad, duplicados, estado, mismo dataset).

Uso (desde la raíz del repositorio):
    python scripts/verificar_salidas.py
    python scripts/verificar_salidas.py --hasta 2      # verifica solo hasta el notebook 02

Devuelve código de salida 1 si algún control falla.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import config  # noqa: E402

resultados: list[tuple[str, str, bool, str]] = []


def control(etapa: str, descripcion: str, ok: bool, detalle: str = "") -> bool:
    resultados.append((etapa, descripcion, bool(ok), detalle))
    return bool(ok)


# ---------------------------------------------------------------------------
# Controles generales
# ---------------------------------------------------------------------------

def verificar_entorno() -> None:
    etapa = "Entorno"
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    distintas = []
    for linea in req:
        linea = linea.split("#")[0].strip()
        if "==" not in linea:
            continue
        paquete, version = linea.split("==")
        try:
            instalada = metadata.version(paquete)
        except metadata.PackageNotFoundError:
            instalada = "no instalado"
        if instalada != version:
            distintas.append(f"{paquete} {instalada} (esperada {version})")
    control(etapa, "Versiones instaladas iguales a requirements.txt", not distintas,
            "; ".join(distintas) if distintas else "todas coinciden")
    control(etapa, "Dataset original en data/raw/", config.RAW_FILE.exists(), config.RAW_FILE.name)


def verificar_notebook(etapa: str, nombre: str) -> None:
    ruta = ROOT / "notebooks" / nombre
    if not control(etapa, f"Notebook {nombre} existe", ruta.exists()):
        return
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    celdas = [c for c in nb["cells"] if c["cell_type"] == "code"]
    sin_ejecutar = sum(c.get("execution_count") is None for c in celdas)
    errores = [o.get("ename", "") for c in celdas for o in c.get("outputs", []) if o.get("output_type") == "error"]
    fuente = "".join("".join(c["source"]) for c in celdas)
    muestra = re.search(r"USAR_MUESTRA\s*=\s*(True|False)", fuente)
    control(etapa, f"{nombre}: todas las celdas ejecutadas", sin_ejecutar == 0, f"{sin_ejecutar} sin ejecutar")
    control(etapa, f"{nombre}: sin errores", not errores, ", ".join(errores) if errores else "")
    control(etapa, f"{nombre}: ejecutado con el dataset completo",
            muestra is not None and muestra.group(1) == "False",
            f"USAR_MUESTRA = {muestra.group(1) if muestra else '?'}")


def verificar_archivos(etapa: str, carpeta: Path, patron: str, esperados: int) -> None:
    encontrados = sorted(carpeta.glob(patron))
    control(etapa, f"{esperados} archivos {patron} en {carpeta.relative_to(ROOT)}",
            len(encontrados) >= esperados, f"{len(encontrados)} encontrados")


def verificar_existe(etapa: str, ruta: Path) -> None:
    control(etapa, f"Existe {ruta.relative_to(ROOT)}", ruta.exists())


# ---------------------------------------------------------------------------
# Controles por etapa
# ---------------------------------------------------------------------------

def etapa_01() -> None:
    etapa = "01 EDA"
    verificar_notebook(etapa, "01_EDA.ipynb")
    verificar_archivos(etapa, config.FIGURES_DIR, "eda_*.png", 11)
    for t in ["eda_resumen.md", "eda_diccionario_datos.csv", "eda_calidad_problemas.csv",
              "eda_baseline_dummy.csv", "eda_pruebas_asociacion.csv"]:
        verificar_existe(etapa, config.TABLES_DIR / t)
    resumen = config.TABLES_DIR / "eda_resumen.md"
    if resumen.exists():
        control(etapa, "eda_resumen.md generado con el dataset completo",
                "Dataset completo" in resumen.read_text(encoding="utf-8"))


def etapa_02() -> None:
    import pandas as pd

    etapa = "02 Preparación"
    verificar_notebook(etapa, "02_preparacion.ipynb")
    verificar_archivos(etapa, config.FIGURES_DIR, "prep_*.png", 3)
    verificar_existe(etapa, config.TABLES_DIR / "prep_resumen.md")

    ruta = config.DATA_PROCESSED / "turnos_procesados.csv"
    ruta_meta = config.DATA_PROCESSED / "metadatos_preparacion.json"
    if not (control(etapa, "Existe el dataset procesado completo", ruta.exists())
            and control(etapa, "Existen los metadatos de la preparación", ruta_meta.exists())):
        return
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    hash_actual = hashlib.sha256(ruta.read_bytes()).hexdigest()
    control(etapa, "Hash SHA-256 coincide con los metadatos", hash_actual == meta["sha256"], hash_actual[:16] + "...")
    control(etapa, "Metadatos generados con el dataset completo", meta["datos"] == "Dataset completo", meta["datos"])

    d = pd.read_csv(ruta, dtype={"paciente_id": str})
    control(etapa, "Filas finales = originales - eliminadas",
            len(d) == meta["filas_originales"] - meta["filas_eliminadas"] == meta["filas_finales"],
            f"{len(d):,} filas")
    solapados = set(d.loc[d.particion == "train", "paciente_id"]) & set(d.loc[d.particion == "test", "paciente_id"])
    control(etapa, "Ningún paciente en entrenamiento y prueba a la vez", not solapados, f"{len(solapados)} solapados")
    folds = sorted(d.loc[d.particion == "train", "fold"].unique())
    control(etapa, "Folds 0 a 4 en entrenamiento y -1 en prueba",
            folds == [0, 1, 2, 3, 4] and (d.loc[d.particion == "test", "fold"] == -1).all(), str(folds))
    control(etapa, "Sin valores faltantes", int(d.isna().sum().sum()) == 0)


def etapa_03() -> None:
    etapa = "03 Modelado"
    verificar_notebook(etapa, "03_modelado_baseline.ipynb")
    verificar_archivos(etapa, config.FIGURES_DIR, "mod_*.png", 6)
    for t in ["mod_resumen.md", "mod_candidatos.csv", "mod_ranking.csv", "mod_baseline.csv",
              "mod_desbalance.csv", "mod_desbalance_diferencias.csv", "mod_analisis_smote.csv",
              "mod_corridas_mlflow.csv"]:
        verificar_existe(etapa, config.TABLES_DIR / t)

    if not control(etapa, "Existe la base de MLflow (mlflow.db)", (ROOT / "mlflow.db").exists()):
        return
    logging.disable(logging.INFO)
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    meta = json.loads((config.DATA_PROCESSED / "metadatos_preparacion.json").read_text(encoding="utf-8"))
    for clave, esperadas in [("baseline", 6), ("desbalance", 15)]:
        nombre = config.MLFLOW_EXPERIMENTS[clave]
        corridas = mlflow.search_runs(experiment_names=[nombre])
        if not control(etapa, f"Experimento {nombre} existe", len(corridas) > 0):
            continue
        control(etapa, f"{nombre}: {esperadas} corridas", len(corridas) == esperadas, f"{len(corridas)} corridas")
        control(etapa, f"{nombre}: sin corridas duplicadas",
                corridas["tags.mlflow.runName"].nunique() == len(corridas))
        control(etapa, f"{nombre}: todas finalizadas", (corridas["status"] == "FINISHED").all())
        control(etapa, f"{nombre}: todas con el hash del dataset actual",
                (corridas["tags.dataset_sha256"] == meta["sha256"]).all())
        commits = corridas["tags.git_commit"].unique()
        control(etapa, f"{nombre}: commit de Git registrado", "sin_git" not in commits, ", ".join(commits))
        if clave == "baseline":
            arbol = corridas[corridas["tags.mlflow.runName"] == "arbol__ninguna"]
            if len(arbol):
                pr_train = float(arbol["metrics.pr_auc_train_media"].iloc[0])
                control(etapa, "PR-AUC de entrenamiento del árbol sin poda ~1 (codificación corregida)",
                        pr_train > 0.99, f"{pr_train:.4f}")
    otros = [e.name for e in mlflow.search_experiments() if e.name.endswith("_muestra")]
    control(etapa, "Sin experimentos de la muestra mezclados", not otros, ", ".join(otros))


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica las salidas del proyecto.")
    parser.add_argument("--hasta", type=int, default=3, help="Último notebook a verificar (1 a 3)")
    args = parser.parse_args()

    verificar_entorno()
    for numero, funcion in [(1, etapa_01), (2, etapa_02), (3, etapa_03)]:
        if numero <= args.hasta:
            funcion()

    fallas = [r for r in resultados if not r[2]]
    lineas = ["# Verificación de salidas del proyecto", "",
              f"**Controles:** {len(resultados)} | **Superados:** {len(resultados) - len(fallas)} | "
              f"**Fallidos:** {len(fallas)}", "",
              "| Etapa | Control | Resultado | Detalle |", "|---|---|---|---|"]
    lineas += [f"| {e} | {d} | {'OK' if ok else 'FALLA'} | {det} |" for e, d, ok, det in resultados]
    texto = "\n".join(lineas) + "\n"
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (config.TABLES_DIR / "verificacion.md").write_text(texto, encoding="utf-8")

    ancho = max(len(d) for _, d, _, _ in resultados)
    for e, d, ok, det in resultados:
        print(f"[{'OK   ' if ok else 'FALLA'}] {e:<15} {d:<{ancho}}  {det}")
    print(f"\n{len(resultados) - len(fallas)} de {len(resultados)} controles superados. "
          f"Detalle en reports/tables/verificacion.md")
    sys.exit(1 if fallas else 0)


if __name__ == "__main__":
    main()
