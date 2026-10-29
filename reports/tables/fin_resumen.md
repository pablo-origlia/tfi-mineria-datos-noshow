# Resumen de la evaluación final

**Datos utilizados:** Dataset completo

## Cifras clave

| Indicador | Valor |
|---|---|
| Datos utilizados | Dataset completo |
| Turnos de entrenamiento / prueba | 88,301 / 22,215 |
| Tasa de no-show en prueba | 20.55% |
| Umbral de decisión (máximo F1 en OOF) | 0.2301 |
| PR-AUC en prueba (IC 95%) | 0.3878 (0.3694 - 0.4060) |
| ROC-AUC en prueba (IC 95%) | 0.7406 (0.7320 - 0.7485) |
| Recall / precisión / F1 en prueba (umbral elegido) | 0.745 / 0.330 / 0.457 |
| PR-AUC en validación cruzada | 0.3837 ± 0.0142 |
| Piso sin información (tasa de no-show en prueba) | 0.2055 |
| Cada 1.000 turnos: alertas / ausencias detectadas / ausencias totales | 465 / 153 / 206 |
| Contactando al 20% de mayor riesgo se captura | 38.1% de las ausencias (lift 1.91) |
| Grupo de atributos más importante | Anticipación del turno (caída de PR-AUC 0.1547) |

## Puntos de operación (predicciones fuera de fold del entrenamiento)

| Punto de operación | Umbral | Recall | Precisión | F1 | % de turnos alertados |
|---|---|---|---|---|---|
| Máximo F1 (principal) | 0.2301 | 0.7413 | 0.3303 | 0.457 | 45.0991 |
| Recall mínimo 50% | 0.2977 | 0.5 | 0.3711 | 0.426 | 27.079 |
| Recall mínimo 70% | 0.2419 | 0.7 | 0.3362 | 0.4543 | 41.8376 |
| Capacidad 20% de los turnos | 0.3253 | 0.3879 | 0.3897 | 0.3888 | 20.0009 |

## Resultados en prueba

| Modelo | Umbral | PR-AUC | ROC-AUC | Recall (no-show) | Precisión (no-show) | F1 (no-show) | Exactitud balanceada | Brier | PR-AUC (CV) |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM optimizado (final) | 0.2301 | 0.3878 | 0.7406 | 0.7453 | 0.3297 | 0.4571 | 0.6766 | 0.1441 | 0.3837 |
| Dummy (clase a priori) | 0.5 | 0.2055 | 0.5 | 0 | 0 | 0 | 0.5 | 0.1633 | 0.201 |
| LightGBM por defecto | 0.2173 | 0.3844 | 0.739 | 0.7753 | 0.3225 | 0.4555 | 0.677 | 0.1444 | 0.3814 |
| Árbol de Decisión optimizado | 0.2123 | 0.36 | 0.7248 | 0.7665 | 0.3122 | 0.4437 | 0.6648 | 0.1467 | 0.3573 |
| Árbol de Decisión (profundidad 3) | 0.2135 | 0.3243 | 0.7103 | 0.9244 | 0.2857 | 0.4365 | 0.6632 | 0.1479 | 0.3242 |

## Resultados en prueba con IC 95% (bootstrap por paciente)

| Modelo | PR-AUC | ROC-AUC | Recall (no-show) | Precisión (no-show) | F1 (no-show) | Brier |
|---|---|---|---|---|---|---|
| LightGBM optimizado (final) | 0.3878 (0.3694 - 0.4060) | 0.7406 (0.7320 - 0.7485) | 0.7453 (0.7327 - 0.7583) | 0.3297 (0.3181 - 0.3394) | 0.4571 (0.4447 - 0.4679) | 0.1441 (0.1413 - 0.1470) |
| LightGBM por defecto | 0.3844 (0.3665 - 0.4036) | 0.7390 (0.7304 - 0.7465) | 0.7753 (0.7621 - 0.7876) | 0.3225 (0.3116 - 0.3320) | 0.4555 (0.4439 - 0.4663) | 0.1444 (0.1415 - 0.1473) |
| Árbol de Decisión optimizado | 0.3600 (0.3465 - 0.3739) | 0.7248 (0.7165 - 0.7330) | 0.7665 (0.7531 - 0.7796) | 0.3122 (0.3019 - 0.3219) | 0.4437 (0.4313 - 0.4542) | 0.1467 (0.1437 - 0.1497) |
| Árbol de Decisión (profundidad 3) | 0.3243 (0.3133 - 0.3354) | 0.7103 (0.7024 - 0.7185) | 0.9244 (0.9154 - 0.9336) | 0.2857 (0.2769 - 0.2940) | 0.4365 (0.4259 - 0.4462) | 0.1479 (0.1449 - 0.1508) |

## Diferencias pareadas de PR-AUC en prueba

| Comparación | Δ PR-AUC (prueba) | IC 95% inferior | IC 95% superior | Réplicas en que el final es mejor | ¿Diferencia consistente? |
|---|---|---|---|---|---|
| Final - LightGBM por defecto | 0.0034 | -0.0002 | 0.0075 | 96.1% | No |
| Final - Árbol de Decisión optimizado | 0.0279 | 0.0168 | 0.0394 | 100.0% | Sí |
| Final - Árbol de Decisión (profundidad 3) | 0.0635 | 0.0501 | 0.0781 | 100.0% | Sí |

## Validación cruzada frente a prueba

| Modelo | PR-AUC (CV) | PR-AUC (CV, desvío) | PR-AUC (prueba) | Diferencia (prueba - CV) |
|---|---|---|---|---|
| LightGBM optimizado (final) | 0.3837 | 0.0142 | 0.3878 | 0.0041 |
| Dummy (clase a priori) | 0.201 | 0.0031 | 0.2055 | 0.0046 |
| LightGBM por defecto | 0.3814 | 0.0143 | 0.3844 | 0.003 |
| Árbol de Decisión optimizado | 0.3573 | 0.0091 | 0.36 | 0.0027 |
| Árbol de Decisión (profundidad 3) | 0.3242 | 0.0072 | 0.3243 | 0.0001 |

## Impacto operativo cada 1.000 turnos

| Concepto | Cada 1.000 turnos |
|---|---|
| Turnos alertados como riesgo | 465 |
| Ausencias detectadas (verdaderos positivos) | 153 |
| Falsas alarmas (asisten pese a la alerta) | 311 |
| Ausencias no detectadas (falsos negativos) | 52 |
| Ausencias totales | 206 |

## Curva de ganancia

| % de turnos contactados (mayor riesgo) | % de ausencias capturadas | Lift (respecto del azar) |
|---|---|---|
| 10% | 21.53 | 2.15 |
| 20% | 38.11 | 1.91 |
| 30% | 54.58 | 1.82 |
| 50% | 78.49 | 1.57 |

## Importancia por permutación (grupos)

| Grupo | Atributos | Caída de PR-AUC (media) | Caída de PR-AUC (desvío) |
|---|---|---|---|
| Anticipación del turno | anticipacion_dias, mismo_dia | 0.1547 | 0.0037 |
| Demográficos | edad, mujer, barrio, beca | 0.0571 | 0.003 |
| Historial del paciente | turnos_previos, ausencias_previas, tasa_ausencias_previas, tiene_historial | 0.0286 | 0.0029 |
| Agenda del día | turnos_mismo_dia, dia_semana, hora_agenda | 0.0147 | 0.0023 |
| Recordatorio SMS | sms_recibido | 0.0106 | 0.0011 |
| Clínicos | hipertension, diabetes, alcoholismo, discapacidad | 0.0025 | 0.0015 |

## Importancia por permutación (atributos)

| Atributo | Caída de PR-AUC (media) | Caída de PR-AUC (desvío) |
|---|---|---|
| anticipacion_dias | 0.1234 | 0.0038 |
| edad | 0.0435 | 0.0023 |
| tasa_ausencias_previas | 0.0145 | 0.0022 |
| barrio | 0.0107 | 0.0012 |
| ausencias_previas | 0.0104 | 0.0008 |
| sms_recibido | 0.0104 | 0.0017 |
| hora_agenda | 0.0083 | 0.0012 |
| turnos_previos | 0.006 | 0.0008 |
| turnos_mismo_dia | 0.0044 | 0.0011 |
| mismo_dia | 0.0043 | 0.0015 |
| dia_semana | 0.0032 | 0.001 |
| beca | 0.0026 | 0.0007 |
| mujer | 0.002 | 0.0005 |
| alcoholismo | 0.0018 | 0.0008 |
| hipertension | 0.0013 | 0.0003 |
| diabetes | 0.0002 | 0.0004 |
| discapacidad | 0.0001 | 0.0002 |
| tiene_historial | -0.0001 | 0 |

## Reglas del árbol de profundidad 3

| N.º | Regla | Turnos (entrenamiento) | No-show (entrenamiento) | Turnos (prueba) | No-show (prueba) |
|---|---|---|---|---|---|
| 1 | mismo_dia <= 0.50 y edad <= 44.50 y anticipacion_dias > 9.50 | 17427 | 0.367 | 4579 | 0.3678 |
| 2 | mismo_dia > 0.50 y ausencias_previas > 2.50 | 426 | 0.3357 | 108 | 0.4167 |
| 3 | mismo_dia <= 0.50 y edad > 44.50 y tasa_ausencias_previas > 0.26 | 2392 | 0.314 | 545 | 0.2862 |
| 4 | mismo_dia <= 0.50 y edad <= 44.50 y anticipacion_dias <= 9.50 | 15451 | 0.2842 | 3896 | 0.2831 |
| 5 | mismo_dia <= 0.50 y edad > 44.50 y tasa_ausencias_previas <= 0.26 | 22018 | 0.2183 | 5647 | 0.2183 |
| 6 | mismo_dia > 0.50 y ausencias_previas <= 2.50 y tasa_ausencias_previas > 0.14 | 2677 | 0.09 | 739 | 0.0853 |
| 7 | mismo_dia > 0.50 y ausencias_previas <= 2.50 y tasa_ausencias_previas <= 0.14 | 27910 | 0.0365 | 6701 | 0.0421 |
