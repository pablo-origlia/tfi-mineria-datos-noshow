<#
.SYNOPSIS
    Borra todas las salidas generadas y vuelve a ejecutar el proyecto desde cero.

.DESCRIPTION
    1. Controla el entorno (entorno virtual activo, jupyter disponible, cambios sin commit).
    2. Borra las salidas generadas: datos procesados, figuras, tablas, modelos, mlflow.db y mlartifacts/.
       NO borra el dataset original (data/raw), las capturas (docs/capturas) ni los notebooks.
    3. Regenera la muestra de desarrollo (opcional, con -IncluirMuestra).
    4. Ejecuta los notebooks en orden con el dataset completo, guardando las salidas en el mismo
       notebook, y se detiene ante el primer error.
    5. Ejecuta scripts/verificar_salidas.py para comprobar que no falte nada.

.PARAMETER Hasta
    Último notebook a ejecutar (1 a 4). Por defecto, 4.

.PARAMETER IncluirMuestra
    Borra y regenera también la muestra de desarrollo (data/interim).

.PARAMETER SoloLimpiar
    Solo borra las salidas, sin ejecutar nada.

.PARAMETER Forzar
    No pide confirmación antes de borrar.

.EXAMPLE
    .\scripts\reiniciar_y_ejecutar.ps1

.EXAMPLE
    .\scripts\reiniciar_y_ejecutar.ps1 -IncluirMuestra -Forzar

.EXAMPLE
    .\scripts\reiniciar_y_ejecutar.ps1 -Hasta 3   # regenera solo hasta el notebook 03
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 4)][int]$Hasta = 4,
    [switch]$IncluirMuestra,
    [switch]$SoloLimpiar,
    [switch]$Forzar
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Notebooks = @("01_EDA.ipynb", "02_preparacion.ipynb", "03_modelado_baseline.ipynb", "04_optimizacion_optuna.ipynb")

function Write-Paso([string]$Texto) {
    Write-Host ""
    Write-Host "=== $Texto ===" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 1. Controles previos
# ---------------------------------------------------------------------------
Write-Paso "1. Controles previos"

$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { throw "No se encontró 'python'. Activá el entorno virtual: .\.venv\Scripts\Activate.ps1" }
if (-not $env:VIRTUAL_ENV) {
    Write-Warning "No hay un entorno virtual activo. Se usará: $python"
} else {
    Write-Host "Entorno virtual: $env:VIRTUAL_ENV" -ForegroundColor Green
}
if (-not $SoloLimpiar) {
    & python -m jupyter nbconvert --version *> $null
    if ($LASTEXITCODE -ne 0) { throw "jupyter nbconvert no está disponible. Ejecutá: pip install -r requirements.txt" }
    if (-not (Test-Path "data/raw/KaggleV2-May-2016.csv")) { throw "Falta data/raw/KaggleV2-May-2016.csv" }
}

if (Get-Command git -ErrorAction SilentlyContinue) {
    $cambios = git status --porcelain 2>$null
    if ($cambios) {
        Write-Warning "Hay cambios sin commit. La etiqueta git_commit de MLflow no reflejará exactamente el código ejecutado."
        Write-Host "Se recomienda hacer un commit antes de continuar." -ForegroundColor Yellow
    } else {
        Write-Host "Repositorio sin cambios pendientes (commit $(git rev-parse --short HEAD))." -ForegroundColor Green
    }
}

# ---------------------------------------------------------------------------
# 2. Limpieza
# ---------------------------------------------------------------------------
Write-Paso "2. Limpieza de salidas generadas"

$aBorrar = @(
    "data/processed/*",
    "reports/figures/*",
    "reports/tables/*",
    "models/*",
    "mlflow.db",
    "mlartifacts"
)
if ($IncluirMuestra) { $aBorrar += "data/interim/*" }

Write-Host "Se borrará:"
$aBorrar | ForEach-Object { Write-Host "  - $_" }
Write-Host "Se conserva: data/raw, docs/capturas, notebooks y código." -ForegroundColor Green

if (-not $Forzar) {
    $respuesta = Read-Host "¿Continuar? (s/n)"
    if ($respuesta -notin @("s", "S", "si", "sí")) { Write-Host "Cancelado."; exit 0 }
}

foreach ($patron in $aBorrar) {
    Get-ChildItem -Path $patron -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne ".gitkeep" } |
        Remove-Item -Recurse -Force
}
Write-Host "Limpieza completa." -ForegroundColor Green

if ($SoloLimpiar) { exit 0 }

# ---------------------------------------------------------------------------
# 3. Muestra de desarrollo (opcional)
# ---------------------------------------------------------------------------
if ($IncluirMuestra) {
    Write-Paso "3. Muestra de desarrollo"
    & python scripts/crear_muestra.py
    if ($LASTEXITCODE -ne 0) { throw "Falló la generación de la muestra." }
}

# ---------------------------------------------------------------------------
# 4. Ejecución de los notebooks
# ---------------------------------------------------------------------------
Write-Paso "4. Ejecución de los notebooks (dataset completo)"

$tiempos = @()
for ($i = 0; $i -lt $Hasta; $i++) {
    $nb = $Notebooks[$i]
    $ruta = Join-Path "notebooks" $nb
    $contenido = Get-Content $ruta -Raw -Encoding UTF8
    if ($contenido -notmatch 'USAR_MUESTRA = False') {
        throw "$nb no tiene USAR_MUESTRA = False. Corregilo antes de ejecutar."
    }
    Write-Host ""
    Write-Host "Ejecutando $nb ..." -ForegroundColor Yellow
    $inicio = Get-Date
    & python -m jupyter nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.timeout=-1 $ruta
    if ($LASTEXITCODE -ne 0) { throw "Falló la ejecución de $nb. Abrilo en Jupyter para ver la celda con error." }
    $minutos = [math]::Round(((Get-Date) - $inicio).TotalMinutes, 1)
    $tiempos += [pscustomobject]@{ Notebook = $nb; Minutos = $minutos }
    Write-Host "$nb completado en $minutos min." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 5. Verificación
# ---------------------------------------------------------------------------
Write-Paso "5. Verificación de salidas"
$tiempos | Format-Table -AutoSize
& python scripts/verificar_salidas.py --hasta $Hasta
$verificacion = $LASTEXITCODE

Write-Host ""
if ($verificacion -eq 0) {
    Write-Host "Proyecto regenerado y verificado correctamente." -ForegroundColor Green
} else {
    Write-Host "La verificación encontró problemas. Revisá reports/tables/verificacion.md" -ForegroundColor Red
}
Write-Host "Recordá volver a tomar las capturas de MLflow (los identificadores de las corridas cambiaron)." -ForegroundColor Yellow
exit $verificacion
