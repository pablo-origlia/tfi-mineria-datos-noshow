# Resumen de la optimización con Optuna

**Datos utilizados:** Dataset completo

## Cifras clave

| Indicador | Valor |
|---|---|
| Datos utilizados | Dataset completo |
| Trials por modelo | 60 |
| Modelo final | LightGBM (versión optimizado) |
| PR-AUC del modelo final (validación cruzada) | 0.3837 ± 0.0142 |
| ROC-AUC del modelo final | 0.7451 |
| Ventaja sobre el segundo | +0.0010 (XGBoost; gana en 3/5 folds) |
| Corridas registradas en MLflow (03_optuna) | 4 estudios + 240 trials |

## Espacios de búsqueda

| Modelo | Hiperparámetro | Rango |
|---|---|---|
| LightGBM | estrategia | ninguna o pesos |
| LightGBM | n_estimators | 100 a 1000 (log) |
| LightGBM | learning_rate | 0,01 a 0,2 (log) |
| LightGBM | num_leaves | 8 a 128 (log) |
| LightGBM | min_child_samples | 10 a 300 (log) |
| LightGBM | subsample | 0,5 a 1 |
| LightGBM | colsample_bytree | 0,4 a 1 |
| LightGBM | reg_alpha | 0,001 a 10 (log) |
| LightGBM | reg_lambda | 0,001 a 10 (log) |
| XGBoost | estrategia | ninguna o pesos |
| XGBoost | n_estimators | 100 a 1000 (log) |
| XGBoost | learning_rate | 0,01 a 0,3 (log) |
| XGBoost | max_depth | 3 a 10 |
| XGBoost | min_child_weight | 1 a 50 (log) |
| XGBoost | subsample | 0,5 a 1 |
| XGBoost | colsample_bytree | 0,4 a 1 |
| XGBoost | reg_alpha | 0,001 a 10 (log) |
| XGBoost | reg_lambda | 0,001 a 10 (log) |
| Regresión Logística | estrategia | ninguna o pesos |
| Regresión Logística | C | 0,001 a 10 (log) |
| Regresión Logística | penalty | l1 o l2 |
| Árbol de Decisión | estrategia | ninguna o pesos o submuestreo |
| Árbol de Decisión | max_depth | 3 a 20 |
| Árbol de Decisión | min_samples_leaf | 5 a 500 (log) |
| Árbol de Decisión | criterion | gini o entropy |

## Mejores configuraciones

| Modelo | Mejor trial | Estrategia | Hiperparámetros |
|---|---|---|---|
| LightGBM | 56 | Sin tratamiento | n_estimators=108, learning_rate=0.05195, num_leaves=50, min_child_samples=22, subsample=0.9181, subsample_freq=1, colsample_bytree=0.9533, reg_alpha=0.02007, reg_lambda=3.945 |
| XGBoost | 44 | Sin tratamiento | n_estimators=384, learning_rate=0.01111, max_depth=7, min_child_weight=12.94, subsample=0.7628, colsample_bytree=0.7331, reg_alpha=0.3723, reg_lambda=0.001506 |
| Regresión Logística | 51 | Sin tratamiento | C=0.01296, penalty=l2, solver=liblinear |
| Árbol de Decisión | 37 | Sin tratamiento | max_depth=18, min_samples_leaf=364, criterion=gini |

## Importancia de los hiperparámetros (fANOVA)

| Hiperparámetro | LightGBM | XGBoost | Regresión Logística | Árbol de Decisión |
|---|---|---|---|---|
| learning_rate | 0.256 | 0.726 | - | - |
| num_leaves | 0.223 | - | - | - |
| min_child_samples | 0.197 | - | - | - |
| n_estimators | 0.223 | 0.075 | - | - |
| colsample_bytree | 0.06 | 0.03 | - | - |
| subsample | 0.024 | 0.06 | - | - |
| reg_alpha | 0.013 | 0.007 | - | - |
| estrategia | 0.003 | 0.012 | 0.944 | 0.03 |
| reg_lambda | 0.001 | 0.004 | - | - |
| min_child_weight | - | 0.039 | - | - |
| max_depth | - | 0.048 | - | 0.519 |
| C | - | - | 0.044 | - |
| penalty | - | - | 0.011 | - |
| min_samples_leaf | - | - | - | 0.449 |
| criterion | - | - | - | 0.002 |

## Baseline frente a optimizado

| Modelo | Versión | Estrategia | PR-AUC | PR-AUC (desvío) | ROC-AUC | F1 con umbral óptimo | Brier | PR-AUC entrenamiento | Brecha de sobreajuste | Segundos (5 folds) |
|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM | Baseline | Sin tratamiento | 0.3814 | 0.0143 | 0.7442 | 0.4566 | 0.1415 | 0.4898 | 0.1084 | 1.2055 |
| LightGBM | Optimizado | Sin tratamiento | 0.3837 | 0.0142 | 0.7451 | 0.457 | 0.1413 | 0.4725 | 0.0888 | 2.1921 |
| XGBoost | Baseline | Sin tratamiento | 0.3659 | 0.0093 | 0.7352 | 0.45 | 0.1441 | 0.5907 | 0.2248 | 1.2228 |
| XGBoost | Optimizado | Sin tratamiento | 0.3827 | 0.0154 | 0.745 | 0.4578 | 0.1413 | 0.4535 | 0.0707 | 5.8323 |
| Regresión Logística | Baseline | Sin tratamiento | 0.3658 | 0.0153 | 0.7361 | 0.4509 | 0.1429 | 0.3676 | 0.0018 | 1.2507 |
| Regresión Logística | Optimizado | Sin tratamiento | 0.3663 | 0.0145 | 0.7357 | 0.451 | 0.1429 | 0.3679 | 0.0016 | 1.2955 |
| Árbol de Decisión | Baseline | Submuestreo aleatorio | 0.2424 | 0.0059 | 0.5958 | 0.3719 | 0.4015 | 0.4625 | 0.22 | 0.7744 |
| Árbol de Decisión | Optimizado | Sin tratamiento | 0.3573 | 0.0091 | 0.7315 | 0.4482 | 0.1435 | 0.3799 | 0.0226 | 0.6783 |

## Mejora pareada por fold

| Modelo | Δ PR-AUC (media) | Δ PR-AUC (desvío) | Folds con mejora | Supera al baseline |
|---|---|---|---|---|
| LightGBM | 0.0023 | 0.001 | 5 | Sí |
| XGBoost | 0.0169 | 0.0077 | 5 | Sí |
| Regresión Logística | 0.0005 | 0.0014 | 4 | Sí |
| Árbol de Decisión | 0.1149 | 0.0068 | 5 | Sí |

## Trials por estado

| tags.modelo | completo | podado |
|---|---|---|
| Árbol de Decisión | 54 | 6 |
| LightGBM | 59 | 1 |
| Regresión Logística | 56 | 4 |
| XGBoost | 59 | 1 |
