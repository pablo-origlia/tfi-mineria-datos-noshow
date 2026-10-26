# Resumen del análisis exploratorio (EDA)

**Datos utilizados:** Dataset completo

## Cifras clave

| Indicador | Valor |
|---|---|
| Datos utilizados | Dataset completo |
| Instancias (turnos) | 110,527 |
| Atributos | 14 |
| Tasa de no-show | 20.19% |
| Relación de clases (asiste : no asiste) | 3.95 : 1 |
| Exactitud del clasificador trivial (clase mayoritaria) | 79.81% |
| Turnos agendados para el mismo día | 34.9% |
| Edad mediana | 37 años |
| Anticipación mediana | 4 días |
| Turnos concentrados en los 5 barrios más frecuentes | 22.9% |
| No-show en turnos del mismo día | 4.6% |
| No-show en turnos con 1 o más días de anticipación | 28.5% |
| Turnos del mismo día con SMS | 0.0% |
| Período de turnos | 29/04/2016 a 08/06/2016 |
| Pacientes únicos | 62,299 |
| Pacientes con más de un turno | 39.1% (concentran el 65.7% de los turnos) |
| No-show si el turno anterior fue ausencia | 34.4% |
| No-show si el turno anterior fue asistencia | 17.5% |

## Problemas de calidad

| Problema | Evidencia | Acción propuesta |
|---|---|---|
| Valores faltantes | 0 valores | No requiere acción |
| Filas duplicadas exactas | 0 filas | No requiere acción |
| Varios turnos del mismo paciente el mismo día | 16,200 filas (14.7%); 82% de los grupos con igual resultado | Conservar como turnos distintos; partición agrupada por paciente para evitar fuga |
| Edad negativa | 1 fila(s) (0.001%) | Eliminar |
| Edad mayor a 100 años | 7 filas (0.006%) | Revisar plausibilidad; eliminar valores imposibles |
| Handcap con valores 2 a 4 | 199 filas (0.180%) | Es un conteo, no un error: conservar o binarizar (> 0) |
| Turno anterior al agendamiento | 5 filas (0.005%) | Eliminar (inconsistencia temporal) |
| AppointmentDay sin hora | 110,527 filas (100%) | No se puede usar la hora del turno como atributo |
| Turnos en sábado | 39 filas (0.035%) | Categoría casi vacía: agrupar o tratar con cuidado |
| PatientId con formato irregular | 5 fila(s) | Conservar (solo se usa como identificador de grupo) |
| Género inconsistente por paciente | 0 pacientes | No requiere acción |
| Barrios con menos de 50 turnos | 5 barrios (56 turnos) | Agrupar en 'Otros' o usar codificación por frecuencia/objetivo |
| Nombres de columnas con errores | Hipertension, Handcap, No-show | Renombrar |

## Clasificadores triviales (validación cruzada estratificada, 5 particiones)

| Clasificador | Exactitud | Exactitud balanceada | Precisión (no-show) | Recall (no-show) | F1 (no-show) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Clase mayoritaria | 0.7981 | 0.5 | 0 | 0 | 0 | 0.5 | 0.2019 |
| Aleatorio estratificado | 0.6782 | 0.5015 | 0.2043 | 0.205 | 0.2047 | 0.5015 | 0.2024 |

## Pruebas de asociación con la variable objetivo

| Atributo | Prueba | p-valor | Tamaño del efecto | Medida |
|---|---|---|---|---|
| anticipacion_dias | Mann-Whitney U | 0 | 0.3965 | Biserial de rangos |
| SMS_received | Chi-cuadrado | 0 | 0.1264 | V de Cramér |
| Age | Mann-Whitney U | 4.152e-92 | -0.088 | Biserial de rangos |
| Neighbourhood | Chi-cuadrado | 1.542e-60 | 0.0667 | V de Cramér |
| Hipertension | Chi-cuadrado | 1.918e-32 | 0.0357 | V de Cramér |
| Scholarship | Chi-cuadrado | 3.907e-22 | 0.0291 | V de Cramér |
| dia_semana | Chi-cuadrado | 4.623e-05 | 0.0158 | V de Cramér |
| Diabetes | Chi-cuadrado | 4.85e-07 | 0.0151 | V de Cramér |
| Handcap | Chi-cuadrado | 0.1341 | 0.008 | V de Cramér |
| Gender | Chi-cuadrado | 0.1734 | 0.0041 | V de Cramér |
| Alcoholism | Chi-cuadrado | 0.9654 | 0.0001304 | V de Cramér |

## Tasa de no-show por anticipación

| tramo_anticipacion | n | tasa | ic_inf | ic_sup |
|---|---|---|---|---|
| 0 (mismo día) | 38563 | 0.0465 | 0.0444 | 0.0486 |
| 1 | 5213 | 0.2135 | 0.2026 | 0.2248 |
| 2-3 | 9462 | 0.2374 | 0.2289 | 0.246 |
| 4-7 | 17510 | 0.252 | 0.2457 | 0.2585 |
| 8-15 | 13528 | 0.308 | 0.3002 | 0.3158 |
| 16-30 | 15868 | 0.3251 | 0.3179 | 0.3324 |
| 31-60 | 8283 | 0.3415 | 0.3314 | 0.3518 |
| 61+ | 2095 | 0.2845 | 0.2656 | 0.3042 |

## Tasa de no-show por grupo de edad

| grupo_edad | n | tasa | ic_inf | ic_sup |
|---|---|---|---|---|
| <1 | 3540 | 0.1805 | 0.1682 | 0.1935 |
| 1-5 | 8192 | 0.1887 | 0.1804 | 0.1973 |
| 6-12 | 9305 | 0.228 | 0.2196 | 0.2367 |
| 13-18 | 7830 | 0.2605 | 0.2509 | 0.2704 |
| 19-30 | 16765 | 0.2471 | 0.2407 | 0.2537 |
| 31-45 | 21954 | 0.2137 | 0.2083 | 0.2192 |
| 46-60 | 23179 | 0.1782 | 0.1733 | 0.1832 |
| 61-75 | 14370 | 0.1486 | 0.1429 | 0.1546 |
| 76+ | 5392 | 0.1614 | 0.1518 | 0.1714 |

## Tasa de no-show según el turno anterior

| historial | n | tasa | ic_inf | ic_sup |
|---|---|---|---|---|
| Primer turno | 62299 | 0.1953 | 0.1922 | 0.1984 |
| Anterior: asistió | 38190 | 0.1754 | 0.1717 | 0.1793 |
| Anterior: no asistió | 10038 | 0.344 | 0.3348 | 0.3533 |

## No-show con y sin SMS por anticipación

| tramo_anticipacion | n_sms0 | n_sms1 | tasa_sms0 | tasa_sms1 |
|---|---|---|---|---|
| 0 (mismo día) | 38,563 | nan | 0.0465 | nan |
| 1 | 5,213 | nan | 0.2135 | nan |
| 2-3 | 8,556 | 906 | 0.2399 | 0.213 |
| 4-7 | 6,868 | 10,642 | 0.2711 | 0.2397 |
| 8-15 | 5,601 | 7,927 | 0.3412 | 0.2845 |
| 16-30 | 6,271 | 9,597 | 0.3684 | 0.2969 |
| 31-60 | 3,224 | 5,059 | 0.3837 | 0.3147 |
| 61+ | 744 | 1,351 | 0.3387 | 0.2546 |
