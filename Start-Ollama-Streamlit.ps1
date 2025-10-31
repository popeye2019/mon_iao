<# 
Start-Ollama-Streamlit.ps1  (v4 — separate stdout/stderr logs)
- Vérifie si Ollama est up (process + API).
- Lance Ollama en arrière-plan si nécessaire (redirections vers 2 fichiers).
- Active le venv 'env' si présent.
- Lance Streamlit.
Place ce script à la **racine du projet**.
#>

[CmdletBinding()]
param(
    [string]$OllamaExe = "C:\Users\franc\AppData\Local\Programs\Ollama\ollama.exe",
    [int]$OllamaPort = 11434,
    [string]$ProjectRoot = $PSScriptRoot
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

Write-Host "➡️  Dossier projet : $ProjectRoot"

# 1) Lancer Ollama si pas déjà up
$ollamaRunning = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaRunning -and -not (Test-OllamaUp -Port $OllamaPort)) {
    if (-not (Test-Path $OllamaExe)) {
        throw "Ollama.exe introuvable à l'emplacement : $OllamaExe"
    }
    Write-Host "🚀 Démarrage d'Ollama..."

    # Logs uniques (stdout/err séparés)
    $ts = Get-Date -Format "yyyyMMdd-HHmmss"
    $outLog = Join-Path $ProjectRoot ("ollama-{0}.out.log" -f $ts)
    $errLog = Join-Path $ProjectRoot ("ollama-{0}.err.log" -f $ts)
    Write-Host "📝 stdout => $outLog"
    Write-Host "📝 stderr => $errLog"

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
    Write-Host "✅ Ollama est déjà lancé."
}

# 2) Activer l'environnement virtuel si présent
$activate = Join-Path $ProjectRoot "env\Scripts\Activate.ps1"
if (Test-Path $activate) {
    Write-Host "🧪 Activation de l'environnement virtuel (env)"
    . $activate
} else {
    Write-Host "ℹ️  Pas d'environnement virtuel 'env' détecté. On continue avec Python système."
}

# 3) Lancer Streamlit
Set-Location $ProjectRoot
Write-Host "🚀 Lancement de Streamlit (app/main.py)"
python -m streamlit run app/main.py
