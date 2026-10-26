# Resumen de la preparación de los datos

**Datos utilizados:** Dataset completo

## Cifras clave

| Indicador | Valor |
|---|---|
| Datos utilizados | Dataset completo |
| Instancias originales | 110,527 |
| Filas eliminadas en la limpieza | 11 (0.010%) |
| No-show sin historial | 19.4% |
| Turnos con historial previo | 40.0% |
| No-show con 2 o más ausencias previas | 32.1% |
| No-show con historial sin ausencias | 18.8% |
| Atributos para modelos de árboles | 18 |
| Atributos para el modelo lineal (antes de codificar) | 18 |
| Entrenamiento | 88,301 turnos (20.10% no-show) |
| Prueba | 22,215 turnos (20.55% no-show) |
| Instancias finales | 110,516 |
| Hash SHA-256 del dataset procesado | `40d361e54c4cd001...` |

## Registro de limpieza

| Paso | Filas eliminadas | Filas restantes | Justificación |
|---|---|---|---|
| Edad negativa | 1 | 110526 | Valor imposible |
| Edad mayor a 110 años | 5 | 110521 | Valor implausible (115 años, repetido en varios registros) |
| Turno anterior al agendamiento | 5 | 110516 | Inconsistencia temporal: un turno no puede agendarse después de ocurrir |

## Partición entrenamiento / prueba

| particion | turnos | pacientes | tasa_no_show | % turnos |
|---|---|---|---|---|
| train | 88301 | 49737 | 0.201 | 79.8988 |
| test | 22215 | 12559 | 0.2055 | 20.1012 |

## Folds de validación cruzada (entrenamiento)

| fold | turnos | pacientes | tasa_no_show |
|---|---|---|---|
| 0 | 18010 | 10015 | 0.2022 |
| 1 | 17760 | 10014 | 0.201 |
| 2 | 17521 | 9877 | 0.1955 |
| 3 | 17482 | 9916 | 0.2029 |
| 4 | 17528 | 9915 | 0.2032 |

## Tasa de no-show según el historial del paciente

| historial | n | tasa | ic_inf | ic_sup |
|---|---|---|---|---|
| Sin historial | 66286 | 0.1938 | 0.1908 | 0.1968 |
| 0 ausencias | 30119 | 0.1885 | 0.1841 | 0.1929 |
| 1 ausencia | 10472 | 0.2504 | 0.2422 | 0.2588 |
| 2+ ausencias | 3639 | 0.3212 | 0.3063 | 0.3366 |

## Tasa de no-show según los turnos del paciente ese día

| turnos_mismo_dia | n | tasa | ic_inf | ic_sup |
|---|---|---|---|---|
| 1 | 94320 | 0.1987 | 0.1962 | 0.2013 |
| 2 | 13060 | 0.224 | 0.2169 | 0.2312 |
| 3 | 2217 | 0.2183 | 0.2016 | 0.236 |
| 4+ | 919 | 0.1741 | 0.151 | 0.2 |

## Información mutua con el objetivo (entrenamiento)

| atributo | valores_distintos | im_bruta | im_corregida |
|---|---|---|---|
| anticipacion_dias | 126 | 0.053 | 0.0523 |
| mismo_dia | 2 | 0.0492 | 0.0492 |
| sms_recibido | 2 | 0.0079 | 0.0079 |
| edad | 102 | 0.0046 | 0.004 |
| tasa_ausencias_previas | 63 | 0.0039 | 0.0036 |
| ausencias_previas | 15 | 0.0028 | 0.0027 |
| hora_agenda | 16 | 0.0027 | 0.0026 |
| barrio | 81 | 0.0025 | 0.002 |
| turnos_previos | 75 | 0.0019 | 0.0015 |
| hipertension | 2 | 0.00058 | 0.00057 |
| turnos_mismo_dia | 8 | 0.00041 | 0.00037 |
| beca | 2 | 0.00036 | 0.00035 |
| tiene_historial | 2 | 0.00032 | 0.00031 |
| diabetes | 2 | 9e-05 | 8e-05 |
| dia_semana | 5 | 0.0001 | 8e-05 |
| discapacidad | 2 | 3e-05 | 2e-05 |
| mujer | 2 | 2e-05 | 1e-05 |
| alcoholismo | 2 | 0 | 0 |

## Preprocesadores

| Familia | Atributos de entrada | Columnas tras preprocesar | Faltantes (train + test) |
|---|---|---|---|
| arbol | 18 | 18 | 0 |
| lineal | 18 | 28 | 0 |
