# Resumen del modelado baseline y del tratamiento del desbalance

**Datos utilizados:** Dataset completo

## Cifras clave

| Indicador | Valor |
|---|---|
| Datos utilizados | Dataset completo |
| Turnos de entrenamiento | 88,301 |
| Tasa de no-show (entrenamiento) | 20.10% |
| SMOTE: sintéticos con valores fraccionarios en atributos enteros | 98.6% |
| SMOTE: ROC-AUC para distinguir sintéticos de reales | 0.996 |
| Mejor configuración | LightGBM - Sin tratamiento |
| Mejor PR-AUC (validación cruzada) | 0.3814 ± 0.0143 |
| Mejor ROC-AUC asociada | 0.7442 |
| Candidatos para Optuna | LightGBM, XGBoost, Regresión Logística |
| Corridas registradas en MLflow | 21 |

## Experimento 1: parámetros por defecto, sin tratamiento

| Modelo | PR-AUC | ROC-AUC | Recall (no-show) | Precisión (no-show) | F1 (no-show) | Exactitud balanceada | Exactitud | Brier | PR-AUC (desvío) | PR-AUC entrenamiento | Segundos (5 folds) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.3814 | 0.7442 | 0.038 | 0.5181 | 0.0708 | 0.5146 | 0.7996 | 0.1415 | 0.0143 | 0.4898 | 1.366 |
| XGBoost | 0.3659 | 0.7352 | 0.0788 | 0.451 | 0.1341 | 0.5274 | 0.7957 | 0.1441 | 0.0093 | 0.5907 | 1.2867 |
| Regresión Logística | 0.3658 | 0.7361 | 0.0302 | 0.4983 | 0.057 | 0.5113 | 0.799 | 0.1429 | 0.0153 | 0.3676 | 1.2016 |
| Random Forest | 0.3536 | 0.7244 | 0.1139 | 0.4265 | 0.1796 | 0.5377 | 0.7911 | 0.1469 | 0.0099 | 1 | 6.0772 |
| Árbol de Decisión | 0.2329 | 0.5668 | 0.3264 | 0.2984 | 0.3118 | 0.5667 | 0.7104 | 0.2892 | 0.0051 | 1 | 1.4732 |
| Dummy (clase a priori) | 0.201 | 0.5 | 0 | 0 | 0 | 0.5 | 0.799 | 0.1606 | 0.0031 | 0.201 | 0.2563 |

## Diagnóstico de sobreajuste

| Modelo | PR-AUC entrenamiento | PR-AUC validación | Brecha |
|---|---|---|---|
| LightGBM | 0.4898 | 0.3814 | 0.1084 |
| XGBoost | 0.5907 | 0.3659 | 0.2248 |
| Regresión Logística | 0.3676 | 0.3658 | 0.0018 |
| Random Forest | 1 | 0.3536 | 0.6464 |
| Árbol de Decisión | 1 | 0.2329 | 0.7671 |
| Dummy (clase a priori) | 0.201 | 0.201 | 0 |

## Experimento 2: diferencias pareadas respecto de sin tratamiento

| Modelo | Estrategia | Δ PR-AUC (media) | Δ PR-AUC (desvío) | Folds con mejora | Δ Brier (media) | Δ Recall@0,5 (media) | Tiempo relativo |
|---|---|---|---|---|---|---|---|
| Regresión Logística | Ponderación de clases | -0.0021 | 0.0011 | 0 | 0.0686 | 0.7885 | 0.8291 |
| Regresión Logística | Submuestreo aleatorio | -0.0023 | 0.0009 | 0 | 0.0688 | 0.7862 | 0.6049 |
| Regresión Logística | SMOTE | -0.0043 | 0.0016 | 0 | 0.0689 | 0.7852 | 2.0682 |
| Árbol de Decisión | Ponderación de clases | -0.0013 | 0.0029 | 2 | -0.0102 | -0.027 | 0.9919 |
| Árbol de Decisión | Submuestreo aleatorio | 0.0095 | 0.0054 | 5 | 0.1123 | 0.2652 | 0.5033 |
| Árbol de Decisión | SMOTE | -0.0008 | 0.002 | 2 | 0.0066 | 0.0094 | 2.3989 |
| Random Forest | Ponderación de clases | -0.0014 | 0.003 | 2 | -0.0002 | -0.018 | 1.0615 |
| Random Forest | Submuestreo aleatorio | -0.0074 | 0.0024 | 0 | 0.0684 | 0.6447 | 0.4293 |
| Random Forest | SMOTE | -0.0114 | 0.0065 | 0 | 0.0046 | 0.0653 | 2.6831 |
| LightGBM | Ponderación de clases | -0.002 | 0.0024 | 1 | 0.0646 | 0.7678 | 0.9636 |
| LightGBM | Submuestreo aleatorio | -0.0122 | 0.0056 | 0 | 0.0692 | 0.7727 | 0.6792 |
| LightGBM | SMOTE | -0.0082 | 0.003 | 0 | 0.0022 | 0.0521 | 2.56 |
| XGBoost | Ponderación de clases | -0.0042 | 0.0037 | 1 | 0.0586 | 0.646 | 1.0647 |
| XGBoost | Submuestreo aleatorio | -0.0136 | 0.0055 | 0 | 0.0717 | 0.6829 | 0.7638 |
| XGBoost | SMOTE | -0.005 | 0.0062 | 1 | 0.0025 | 0.0593 | 2.6558 |

## Resumen por estrategia (promedio entre modelos)

| Estrategia | Δ PR-AUC (media) | Δ Brier (media) | Δ Recall@0,5 (media) | Tiempo relativo |
|---|---|---|---|---|
| Ponderación de clases | -0.0022 | 0.0363 | 0.4314 | 0.9822 |
| SMOTE | -0.0059 | 0.017 | 0.1942 | 2.4732 |
| Submuestreo aleatorio | -0.0052 | 0.0781 | 0.6303 | 0.5961 |

## Ranking de todas las configuraciones

| Posición | Modelo | Estrategia | PR-AUC | PR-AUC (desvío) | ROC-AUC | Recall (no-show) | F1 con umbral óptimo | Umbral óptimo (F1) | Brier | PR-AUC entrenamiento | Segundos (5 folds) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LightGBM | Sin tratamiento | 0.3814 | 0.0143 | 0.7442 | 0.038 | 0.4566 | 0.2173 | 0.1415 | 0.4898 | 1.366 |
| 2 | LightGBM | Ponderación de clases | 0.3794 | 0.0137 | 0.7428 | 0.8058 | 0.4547 | 0.518 | 0.2061 | 0.4661 | 1.3163 |
| 3 | LightGBM | SMOTE | 0.3732 | 0.0127 | 0.7411 | 0.0902 | 0.4535 | 0.2525 | 0.1437 | 0.4412 | 3.4969 |
| 4 | LightGBM | Submuestreo aleatorio | 0.3692 | 0.0099 | 0.7394 | 0.8108 | 0.453 | 0.5317 | 0.2107 | 0.4451 | 0.9278 |
| 5 | XGBoost | Sin tratamiento | 0.3659 | 0.0093 | 0.7352 | 0.0788 | 0.45 | 0.2162 | 0.1441 | 0.5907 | 1.2867 |
| 6 | Regresión Logística | Sin tratamiento | 0.3658 | 0.0153 | 0.7361 | 0.0302 | 0.4509 | 0.2332 | 0.1429 | 0.3676 | 1.2016 |
| 7 | Regresión Logística | Ponderación de clases | 0.3637 | 0.0154 | 0.736 | 0.8188 | 0.4513 | 0.5394 | 0.2116 | 0.3652 | 0.9962 |
| 8 | Regresión Logística | Submuestreo aleatorio | 0.3635 | 0.0148 | 0.7358 | 0.8165 | 0.4511 | 0.5423 | 0.2117 | 0.3652 | 0.7268 |
| 9 | XGBoost | Ponderación de clases | 0.3617 | 0.0102 | 0.7292 | 0.7248 | 0.4457 | 0.4521 | 0.2028 | 0.5795 | 1.3699 |
| 10 | Regresión Logística | SMOTE | 0.3615 | 0.0152 | 0.7342 | 0.8154 | 0.4506 | 0.5391 | 0.2118 | 0.363 | 2.4851 |
| 11 | XGBoost | SMOTE | 0.3609 | 0.011 | 0.7327 | 0.1381 | 0.4484 | 0.2185 | 0.1466 | 0.5374 | 3.4172 |
| 12 | Random Forest | Sin tratamiento | 0.3536 | 0.0099 | 0.7244 | 0.1139 | 0.4439 | 0.2 | 0.1469 | 1 | 6.0772 |
| 13 | Random Forest | Ponderación de clases | 0.3523 | 0.0082 | 0.7243 | 0.0958 | 0.4429 | 0.178 | 0.1466 | 0.9999 | 6.4507 |
| 14 | XGBoost | Submuestreo aleatorio | 0.3522 | 0.0124 | 0.7285 | 0.7617 | 0.4466 | 0.4876 | 0.2158 | 0.4905 | 0.9827 |
| 15 | Random Forest | Submuestreo aleatorio | 0.3463 | 0.0095 | 0.7233 | 0.7586 | 0.444 | 0.44 | 0.2153 | 0.9196 | 2.6088 |
| 16 | Random Forest | SMOTE | 0.3422 | 0.0077 | 0.7185 | 0.1791 | 0.4397 | 0.19 | 0.1515 | 1 | 16.3059 |
| 17 | Árbol de Decisión | Submuestreo aleatorio | 0.2424 | 0.0059 | 0.5958 | 0.5916 | 0.3719 | 0.5 | 0.4015 | 0.4625 | 0.7414 |
| 18 | Árbol de Decisión | Sin tratamiento | 0.2329 | 0.0051 | 0.5668 | 0.3264 | 0.3347 | 0 | 0.2892 | 1 | 1.4732 |
| 19 | Árbol de Decisión | SMOTE | 0.2321 | 0.0049 | 0.5662 | 0.3357 | 0.3347 | 0 | 0.2957 | 1 | 3.5341 |
| 20 | Árbol de Decisión | Ponderación de clases | 0.2316 | 0.0023 | 0.5632 | 0.2994 | 0.3347 | 0 | 0.279 | 1 | 1.4614 |

## Candidatos seleccionados

| Rol | Modelo | Estrategia | PR-AUC | PR-AUC (desvío) | ROC-AUC | F1 con umbral óptimo | Δ vs. mejor (media) | Folds en que supera al mejor |
|---|---|---|---|---|---|---|---|---|
| Candidato a optimizar | LightGBM | Sin tratamiento | 0.3814 | 0.0143 | 0.7442 | 0.4566 | 0 | 0 |
| Candidato a optimizar | XGBoost | Sin tratamiento | 0.3659 | 0.0093 | 0.7352 | 0.45 | -0.0156 | 0 |
| Candidato a optimizar | Regresión Logística | Sin tratamiento | 0.3658 | 0.0153 | 0.7361 | 0.4509 | -0.0157 | 0 |
| Referencia interpretable | Árbol de Decisión | Submuestreo aleatorio | 0.2424 | 0.0059 | 0.5958 | 0.3719 | -0.139 | 0 |
