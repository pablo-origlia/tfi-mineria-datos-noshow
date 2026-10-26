# Reporte de la muestra de desarrollo

## Parámetros

- **Archivo original:** `KaggleV2-May-2016.csv`
- **Fracción de pacientes:** 0.15
- **Semilla:** 42
- **Estratos:** 8 (alguna ausencia × tramo de turnos)
- **Pacientes agregados por anomalías:** 103
- **Tamaño del archivo de salida:** 1.56 MB

## Comparación original vs. muestra

| Indicador | Original | Muestra |
|---|---:|---:|
| Filas (turnos) | 110,527 | 16,695 |
| Pacientes únicos | 62,299 | 9,448 |
| Turnos por paciente (media) | 1.77 | 1.77 |
| % no-show | 20.19% | 20.40% |
| % mujeres | 65.00% | 65.86% |
| Edad media | 37.09 | 37.12 |
| % SMS recibido | 32.10% | 32.14% |
| % Scholarship | 9.83% | 10.30% |
| Anticipación media (días) | 10.18 | 10.28 |
| Barrios distintos | 81 | 81 |
| Rango de fechas de turno | 2016-04-29 a 2016-06-08 | 2016-04-29 a 2016-06-08 |
| Filas con edad < 0 | 1 | 1 |
| Filas con edad > 100 | 7 | 7 |
| Filas con Handcap > 1 | 199 | 199 |
| Filas con turno anterior al agendamiento | 5 | 5 |
