# Verificación de salidas del proyecto

**Controles:** 80 | **Superados:** 80 | **Fallidos:** 0

| Etapa | Control | Resultado | Detalle |
|---|---|---|---|
| Entorno | Versiones instaladas iguales a requirements.txt | OK | todas coinciden |
| Entorno | Dataset original en data/raw/ | OK | KaggleV2-May-2016.csv |
| 01 EDA | Notebook 01_EDA.ipynb existe | OK |  |
| 01 EDA | 01_EDA.ipynb: todas las celdas ejecutadas | OK | 0 sin ejecutar |
| 01 EDA | 01_EDA.ipynb: sin errores | OK |  |
| 01 EDA | 01_EDA.ipynb: ejecutado con el dataset completo | OK | USAR_MUESTRA = False |
| 01 EDA | 11 archivos eda_*.png en reports\figures | OK | 11 encontrados |
| 01 EDA | Existe reports\tables\eda_resumen.md | OK |  |
| 01 EDA | Existe reports\tables\eda_diccionario_datos.csv | OK |  |
| 01 EDA | Existe reports\tables\eda_calidad_problemas.csv | OK |  |
| 01 EDA | Existe reports\tables\eda_baseline_dummy.csv | OK |  |
| 01 EDA | Existe reports\tables\eda_pruebas_asociacion.csv | OK |  |
| 01 EDA | eda_resumen.md generado con el dataset completo | OK |  |
| 02 Preparación | Notebook 02_preparacion.ipynb existe | OK |  |
| 02 Preparación | 02_preparacion.ipynb: todas las celdas ejecutadas | OK | 0 sin ejecutar |
| 02 Preparación | 02_preparacion.ipynb: sin errores | OK |  |
| 02 Preparación | 02_preparacion.ipynb: ejecutado con el dataset completo | OK | USAR_MUESTRA = False |
| 02 Preparación | 3 archivos prep_*.png en reports\figures | OK | 3 encontrados |
| 02 Preparación | Existe reports\tables\prep_resumen.md | OK |  |
| 02 Preparación | Existe el dataset procesado completo | OK |  |
| 02 Preparación | Existen los metadatos de la preparación | OK |  |
| 02 Preparación | Hash SHA-256 coincide con los metadatos | OK | 40d361e54c4cd001... |
| 02 Preparación | Metadatos generados con el dataset completo | OK | Dataset completo |
| 02 Preparación | Filas finales = originales - eliminadas | OK | 110,516 filas |
| 02 Preparación | Ningún paciente en entrenamiento y prueba a la vez | OK | 0 solapados |
| 02 Preparación | Folds 0 a 4 en entrenamiento y -1 en prueba | OK | [0, 1, 2, 3, 4] |
| 02 Preparación | Sin valores faltantes | OK |  |
| 03 Modelado | Notebook 03_modelado_baseline.ipynb existe | OK |  |
| 03 Modelado | 03_modelado_baseline.ipynb: todas las celdas ejecutadas | OK | 0 sin ejecutar |
| 03 Modelado | 03_modelado_baseline.ipynb: sin errores | OK |  |
| 03 Modelado | 03_modelado_baseline.ipynb: ejecutado con el dataset completo | OK | USAR_MUESTRA = False |
| 03 Modelado | 6 archivos mod_*.png en reports\figures | OK | 6 encontrados |
| 03 Modelado | Existe reports\tables\mod_resumen.md | OK |  |
| 03 Modelado | Existe reports\tables\mod_candidatos.csv | OK |  |
| 03 Modelado | Existe reports\tables\mod_ranking.csv | OK |  |
| 03 Modelado | Existe reports\tables\mod_baseline.csv | OK |  |
| 03 Modelado | Existe reports\tables\mod_desbalance.csv | OK |  |
| 03 Modelado | Existe reports\tables\mod_desbalance_diferencias.csv | OK |  |
| 03 Modelado | Existe reports\tables\mod_analisis_smote.csv | OK |  |
| 03 Modelado | Existe reports\tables\mod_corridas_mlflow.csv | OK |  |
| 03 Modelado | Existe la base de MLflow (mlflow.db) | OK |  |
| 03 Modelado | Metadatos de la preparación disponibles | OK |  |
| 03 Modelado | Experimento 01_baseline existe | OK |  |
| 03 Modelado | 01_baseline: 6 corridas | OK | 6 corridas |
| 03 Modelado | 01_baseline: sin corridas duplicadas | OK |  |
| 03 Modelado | 01_baseline: todas finalizadas | OK |  |
| 03 Modelado | 01_baseline: todas con el hash del dataset actual | OK |  |
| 03 Modelado | 01_baseline: commit de Git registrado | OK | 768108c |
| 03 Modelado | PR-AUC de entrenamiento del árbol sin poda ~1 (codificación corregida) | OK | 1.0000 |
| 03 Modelado | Experimento 02_desbalance existe | OK |  |
| 03 Modelado | 02_desbalance: 15 corridas | OK | 15 corridas |
| 03 Modelado | 02_desbalance: sin corridas duplicadas | OK |  |
| 03 Modelado | 02_desbalance: todas finalizadas | OK |  |
| 03 Modelado | 02_desbalance: todas con el hash del dataset actual | OK |  |
| 03 Modelado | 02_desbalance: commit de Git registrado | OK | 768108c |
| 03 Modelado | Sin experimentos de la muestra mezclados | OK |  |
| 04 Optuna | Notebook 04_optimizacion_optuna.ipynb existe | OK |  |
| 04 Optuna | 04_optimizacion_optuna.ipynb: todas las celdas ejecutadas | OK | 0 sin ejecutar |
| 04 Optuna | 04_optimizacion_optuna.ipynb: sin errores | OK |  |
| 04 Optuna | 04_optimizacion_optuna.ipynb: ejecutado con el dataset completo | OK | USAR_MUESTRA = False |
| 04 Optuna | 4 archivos opt_*.png en reports\figures | OK | 4 encontrados |
| 04 Optuna | Existe reports\tables\opt_resumen.md | OK |  |
| 04 Optuna | Existe reports\tables\opt_espacios_busqueda.csv | OK |  |
| 04 Optuna | Existe reports\tables\opt_mejores_configuraciones.csv | OK |  |
| 04 Optuna | Existe reports\tables\opt_comparacion_baseline.csv | OK |  |
| 04 Optuna | Existe reports\tables\opt_mejora_pareada.csv | OK |  |
| 04 Optuna | Existe reports\tables\opt_importancia_hiperparametros.csv | OK |  |
| 04 Optuna | Metadatos de la preparación disponibles | OK |  |
| 04 Optuna | Existe opt_seleccion.json | OK |  |
| 04 Optuna | Selección hecha con el dataset completo | OK | Dataset completo |
| 04 Optuna | Selección asociada al hash del dataset actual | OK |  |
| 04 Optuna | Modelo final definido (modelo, versión, estrategia, parámetros) | OK | lightgbm (optimizado) |
| 04 Optuna | Referencia interpretable definida | OK |  |
| 04 Optuna | Experimento 03_optuna existe | OK |  |
| 04 Optuna | 03_optuna: una corrida padre por candidato | OK | 4 estudios, 4 candidatos |
| 04 Optuna | 03_optuna: un trial registrado por cada trial ejecutado | OK | 240 registrados, 240 ejecutados |
| 04 Optuna | 03_optuna: todos los trials con corrida padre | OK |  |
| 04 Optuna | 03_optuna: todas finalizadas | OK |  |
| 04 Optuna | 03_optuna: todas con el hash del dataset actual | OK |  |
| 04 Optuna | 03_optuna: commit de Git registrado | OK | 78c8b43 |
