<#
Start-Ollama-Streamlit.ps1  (v5 - ensure models + separate stdout/stderr logs)
- Vérifie si Ollama est up (process + API).
- Lance Ollama en arrière-plan si nécessaire (redirections vers 2 fichiers).
- Vérifie et pull les modèles requis (LLM + embedding) si absents.
- Active le venv 'env' si présent.
- Lance Streamlit.
Place ce script à la racine du projet.
#>

[CmdletBinding()]
param(
    [string]$OllamaExe = "C:\\Users\\franc\\AppData\\Local\\Programs\\Ollama\\ollama.exe",
    [int]$OllamaPort = 11434,
    [string]$ProjectRoot = $PSScriptRoot,
    [string]$LlmModel = "mistral",
    [string]$EmbeddingModel = "nomic-embed-text",
    [switch]$NoGPU,
    [int]$ContextLength = 4096
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

Write-Host "📁  Dossier projet : $ProjectRoot"

# Stoppe le service Windows 'Ollama' si nécessaire (ex. pour forcer CPU)
function Stop-OllamaServiceIfRunning {
    try {
        $svc = Get-Service -Name 'Ollama' -ErrorAction SilentlyContinue
        if ($null -ne $svc -and $svc.Status -eq 'Running') {
            Write-Host "🛑 Arrêt du service Windows 'Ollama' (pour appliquer NoGPU/Contexte)."
            Stop-Service -Name 'Ollama' -Force -ErrorAction Stop
            Start-Sleep -Seconds 1
        }
    } catch {
        Write-Verbose "Impossible d'inspecter/arrêter le service 'Ollama': $($_.Exception.Message)"
    }
}

# Utilitaires: inspection et pull des modèles Ollama
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
    if (Test-OllamaModelPresent -ModelName $ModelName -Port $Port) {
        Write-Host "✅ Modèle déjà présent: $ModelName"
        return
    }
    Write-Host "⬇️  Pull du modèle: $ModelName (via ollama pull)"
    & $OllamaExePath pull $ModelName
}

# 1) Lancer Ollama si pas déjà up

$ollamaRunning = Get-Process -Name "ollama" -ErrorAction SilentlyContinue

# Si on souhaite désactiver le GPU ou changer le contexte et qu'Ollama tourne déjà,
# on le redémarre pour appliquer les variables d'environnement.
if ($ollamaRunning -and ($NoGPU -or $ContextLength -ne 4096)) {
    Write-Host "♻️ Redémarrage d'Ollama pour appliquer la configuration (GPU/Contexte)."
    try { Stop-Process -Name "ollama" -Force -ErrorAction Stop } catch {}
    Stop-OllamaServiceIfRunning
    Start-Sleep -Seconds 1
}

if (-not (Test-OllamaUp -Port $OllamaPort)) {
    if (-not (Test-Path $OllamaExe)) {
        throw "Ollama.exe introuvable à l'emplacement : $OllamaExe"
    }
    Write-Host "🚀 Démarrage d'Ollama..."

    # Logs uniques (stdout/err séparés)
    $ts = Get-Date -Format "yyyyMMdd-HHmmss"
    $outLog = Join-Path $ProjectRoot ("ollama-{0}.out.log" -f $ts)
    $errLog = Join-Path $ProjectRoot ("ollama-{0}.err.log" -f $ts)
    Write-Host "📄 stdout => $outLog"
    Write-Host "📄 stderr => $errLog"

    # Variables d'environnement pour le processus Ollama (avant le démarrage)
    if ($NoGPU) {
        $env:OLLAMA_NO_GPU = "1"
        $env:CUDA_VISIBLE_DEVICES = "-1"
        $env:OLLAMA_LLM_LIBRARY = "cpu"
    }
    if ($ContextLength -gt 0) { $env:OLLAMA_CONTEXT_LENGTH = [string]$ContextLength }

    $proc = Start-Process -FilePath $OllamaExe -ArgumentList "serve" -NoNewWindow -PassThru `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog

    # Attendre que l'API réponde
    $maxWait = 30
    for ($i=0; $i -lt $maxWait; $i++) {
        if (Test-OllamaUp -Port $OllamaPort) { break }
        Start-Sleep -Seconds 1
    }
    if (-not (Test-OllamaUp -Port $OllamaPort)) {
        throw "Ollama ne répond pas sur le port $OllamaPort après $maxWait secondes. Voir logs: `n$outLog`n$errLog"
    } else {
        Write-Host "✅ Ollama est opérationnel sur http://127.0.0.1:$OllamaPort"
    }
} else {
    Write-Host "ℹ️ Ollama est déjà lancé."
}

# 2) Vérifier/puller les modèles nécessaires
try {
    Ensure-OllamaModel -ModelName $LlmModel -OllamaExePath $OllamaExe -Port $OllamaPort
    Ensure-OllamaModel -ModelName $EmbeddingModel -OllamaExePath $OllamaExe -Port $OllamaPort
} catch {
    Write-Warning "Impossible de vérifier ou de pull les modèles Ollama: $($_.Exception.Message)"
}

# 3) Activer l'environnement virtuel si présent
$activate = Join-Path $ProjectRoot "env\Scripts\Activate.ps1"
if (Test-Path $activate) {
    Write-Host "🐍 Activation de l'environnement virtuel (env)"
    . $activate
} else {
    Write-Host "ℹ️  Pas d'environnement virtuel 'env' détecté. On continue avec Python système."
}

# 4) Lancer Streamlit
Set-Location $ProjectRoot
Write-Host "▶️ Lancement de Streamlit (app/main.py)"
${env:LLM_NAME} = $LlmModel
${env:EMBEDDING_NAME} = $EmbeddingModel
python -m streamlit run app/main.py
