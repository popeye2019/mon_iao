<#
Start-Ollama-Streamlit.ps1  (v10 - auto start Ollama + auto venv)
- Verifie si l'API Ollama est up.
- Demarre Ollama automatiquement (mode auto CPU/GPU) si non lance.
- Verifie et pull les modeles requis (LLM + embedding) si absents.
- Cree et/ou active un venv local 'env' et installe requirements si besoin.
- Lance Streamlit.
Place ce script a la racine du projet.
#>

[CmdletBinding()]
param(
    [string]$OllamaExe = "C:\\Users\\franc\\AppData\\Local\\Programs\\Ollama\\ollama.exe",
    [int]$OllamaPort = 11434,
    [string]$ProjectRoot = $PSScriptRoot,
    [string]$LlmModel = "mistral",
    [string]$EmbeddingModel = "nomic-embed-text",
    [int]$ContextLength = 4096,
    [switch]$AutoSetupVenv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-OllamaUp {
    param([int]$Port = 11434)
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri ("http://127.0.0.1:{0}/api/tags" -f $Port) -TimeoutSec 2
        return $r.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Start-OllamaAuto {
    param([string]$Exe, [int]$Port, [string]$LogDir)
    if (-not (Test-Path $Exe)) { throw "Ollama.exe introuvable: $Exe" }
    $ts = Get-Date -Format 'yyyyMMdd-HHmmss'
    $outLog = Join-Path $LogDir ("ollama-cuda-{0}.out.log" -f $ts)
    $errLog = Join-Path $LogDir ("ollama-cuda-{0}.err.log" -f $ts)

    # Laisser Ollama choisir automatiquement CPU/GPU (aucune variable forcee)
    Remove-Item Env:\OLLAMA_NO_GPU -ErrorAction SilentlyContinue | Out-Null
    Remove-Item Env:\OLLAMA_LLM_LIBRARY -ErrorAction SilentlyContinue | Out-Null
    Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue | Out-Null
    Remove-Item Env:\HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue | Out-Null
    Remove-Item Env:\ROCR_VISIBLE_DEVICES -ErrorAction SilentlyContinue | Out-Null

    Write-Host "[INFO] Demarrage d'Ollama (auto) avec $Exe"
    $null = Start-Process -FilePath $Exe -ArgumentList 'serve' -NoNewWindow -PassThru `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog

    # Attente disponibilite API
    for ($i=0; $i -lt 30; $i++) {
        if (Test-OllamaUp -Port $Port) { break }
        Start-Sleep -Seconds 1
    }
    if (-not (Test-OllamaUp -Port $Port)) {
        throw "Ollama ne repond pas sur 127.0.0.1:$Port apres demarrage. Voir logs:`n$outLog`n$errLog"
    }
    Write-Host "[OK] Ollama pret sur http://127.0.0.1:$Port"
}

# Utilitaires: inspection et pull des modeles Ollama
function Get-OllamaTags {
    param([int]$Port = 11434)
    try {
        $resp = Invoke-RestMethod -UseBasicParsing -Uri ("http://127.0.0.1:{0}/api/tags" -f $Port) -TimeoutSec 5
        return $resp.models
    } catch {
        return @()
    }
}

function Test-OllamaModelPresent {
    param(
        [string]$ModelName,
        [int]$Port = 11434
    )
    $models = Get-OllamaTags -Port $Port
    if (-not $models) { return $false }
    foreach ($m in $models) {
        if ($m.name -eq $ModelName) { return $true }
        if ($ModelName -notmatch ":" -and $m.name -eq ("{0}:latest" -f $ModelName)) { return $true }
    }
    return $false
}

function Ensure-OllamaModel {
    param(
        [string]$ModelName,
        [string]$OllamaExePath,
        [int]$Port = 11434
    )
    if (Test-OllamaModelPresent -ModelName $ModelName -Port $Port) { return }
    Write-Host "[INFO] Pull du modele: $ModelName (via ollama pull)"
    & $OllamaExePath pull $ModelName
}

Write-Host "[INFO] Dossier projet: $ProjectRoot"

# 1) S'assurer qu'Ollama est lance (auto start si necessaire)
if (-not (Test-OllamaUp -Port $OllamaPort)) {
    Start-OllamaAuto -Exe $OllamaExe -Port $OllamaPort -LogDir (Get-Location)
} else {
    Write-Host "[OK] Ollama operationnel sur http://127.0.0.1:$OllamaPort"
}
if ($ContextLength -gt 0) {
    Write-Host "[INFO] Contexte souhaite: $ContextLength (a definir cote serveur si different)."
}

# 2) Verifier/puller les modeles necessaires
try {
    Ensure-OllamaModel -ModelName $LlmModel -OllamaExePath $OllamaExe -Port $OllamaPort
    Ensure-OllamaModel -ModelName $EmbeddingModel -OllamaExePath $OllamaExe -Port $OllamaPort
} catch {
    Write-Warning "[WARN] Impossible de verifier ou de pull les modeles Ollama: $($_.Exception.Message)"
}

# 3) Creer/activer l'environnement virtuel et installer requirements si besoin (silencieux, sauf defauts)
$venvDir = Join-Path $ProjectRoot "env"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$reqFile = Join-Path $ProjectRoot "requirements.txt"
if (-not (Test-Path $venvPython)) {
    if ($AutoSetupVenv -or $true) {
        try { & python -m venv $venvDir *> $null 2>&1 } catch {}
        if (-not (Test-Path $venvPython)) { Write-Warning "[WARN] Echec de creation du venv. Utilisation du Python systeme." }
    }
}

if (Test-Path $venvPython) {
    try { & $venvPython -m pip install --upgrade pip --disable-pip-version-check -q *> $null 2>&1 } catch { Write-Warning "[WARN] Echec MAJ pip: $($_.Exception.Message)" }
    if (Test-Path $reqFile) {
        try { & $venvPython -m pip install -r $reqFile --disable-pip-version-check -q *> $null 2>&1 } catch { Write-Warning "[WARN] Echec installation requirements.txt: $($_.Exception.Message)" }
    } else {
        try { & $venvPython -m pip install streamlit --disable-pip-version-check -q *> $null 2>&1 } catch { Write-Warning "[WARN] Echec installation minimale Streamlit: $($_.Exception.Message)" }
    }
} else {
    Write-Host "[INFO] Pas d'environnement virtuel 'env' actif. On continue avec Python systeme."
}

# 4) Lancer Streamlit
Set-Location $ProjectRoot
Write-Host "[INFO] Lancement de Streamlit (app/main.py)"
${env:LLM_NAME} = $LlmModel
${env:EMBEDDING_NAME} = $EmbeddingModel
if (Test-Path $venvPython) {
    & $venvPython -m streamlit run app/main.py
} else {
    python -m streamlit run app/main.py
}
