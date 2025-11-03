<#
Rebuild-Vectorstore.ps1

Reconstruit l'index vectoriel depuis les fichiers du dossier `data/`
et persiste dans `vectorstore/` en utilisant les utilitaires Python
du projet (app/loader.py, app/indexer.py).

Exemples:
  .\Rebuild-Vectorstore.ps1
  .\Rebuild-Vectorstore.ps1 -DataDir .\data -PersistDir .\vectorstore
#>

[CmdletBinding()]
param(
  [string]$ProjectRoot = $PSScriptRoot,
  [string]$DataDir = (Join-Path $PSScriptRoot 'data'),
  [string]$PersistDir = (Join-Path $PSScriptRoot 'vectorstore'),
  [string]$LlmModel = 'mistral',
  [string]$EmbeddingModel = 'nomic-embed-text',
  [int]$LlmNumCtx = 2048,
  [int]$OllamaPort = 11434,
  [bool]$UseGpuIndex = $true,
  [switch]$AutoStartOllama,
  [string]$OllamaExe = "C:\\Users\\franc\\AppData\\Local\\Programs\\Ollama\\ollama.exe",
  [string]$ChunksExport = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Test-OllamaUp {
  param([int]$Port = 11434)
  try {
    $r = Invoke-WebRequest -UseBasicParsing -Uri ("http://127.0.0.1:{0}/api/tags" -f $Port) -TimeoutSec 2
    return $r.StatusCode -eq 200
  } catch { return $false }
}

function Start-OllamaAuto {
  param([string]$Exe, [int]$Port, [string]$LogDir)
  if (-not (Test-Path $Exe)) { throw "Ollama.exe introuvable: $Exe" }
  $ts = Get-Date -Format 'yyyyMMdd-HHmmss'
  $outLog = Join-Path $LogDir ("ollama-cuda-{0}.out.log" -f $ts)
  $errLog = Join-Path $LogDir ("ollama-cuda-{0}.err.log" -f $ts)

  # Ne conserver que les 5 derniers jeux de logs (par suffixe)
  try {
    foreach ($suffix in @('out.log','err.log')) {
      $files = Get-ChildItem -Path $LogDir -Filter "ollama-*.${suffix}" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
      if ($files.Count -gt 5) {
        $files | Select-Object -Skip 5 | ForEach-Object { try { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue } catch {} }
      }
    }
  } catch {}
  Remove-Item Env:\OLLAMA_NO_GPU -ErrorAction SilentlyContinue | Out-Null
  Remove-Item Env:\OLLAMA_LLM_LIBRARY -ErrorAction SilentlyContinue | Out-Null
  Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue | Out-Null
  Remove-Item Env:\HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue | Out-Null
  Remove-Item Env:\ROCR_VISIBLE_DEVICES -ErrorAction SilentlyContinue | Out-Null
  Write-Host "[INFO] Demarrage d'Ollama (auto) avec $Exe"
  $null = Start-Process -FilePath $Exe -ArgumentList 'serve' -NoNewWindow -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog
  for ($i=0; $i -lt 30; $i++) {
    if (Test-OllamaUp -Port $Port) { break }
    Start-Sleep -Seconds 1
  }
  if (-not (Test-OllamaUp -Port $Port)) {
    throw "Ollama ne repond pas sur 127.0.0.1:$Port apres demarrage. Voir logs:`n$outLog`n$errLog"
  }
  Write-Host "[OK] Ollama pret sur http://127.0.0.1:$Port"
}

Write-Host "[INFO] ProjectRoot: $ProjectRoot"
Write-Host "[INFO] DataDir: $DataDir"
Write-Host "[INFO] PersistDir: $PersistDir"

# 1) venv local (env) + deps (silencieux, n'afficher qu'en cas d'erreur)
$venvDir = Join-Path $ProjectRoot 'env'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'
$reqFile = Join-Path $ProjectRoot 'requirements.txt'
if (-not (Test-Path $venvPython)) {
  try { & python -m venv $venvDir *> $null 2>&1 } catch {}
  if (-not (Test-Path $venvPython)) { Write-Warning "[WARN] Echec de creation du venv 'env'. Utilisation du Python systeme." }
}
if (Test-Path $venvPython) {
  try { & $venvPython -m pip install --upgrade pip --disable-pip-version-check -q *> $null 2>&1 } catch { Write-Warning "[WARN] Echec MAJ pip: $($_.Exception.Message)" }
  if (Test-Path $reqFile) {
    try { & $venvPython -m pip install -r $reqFile --disable-pip-version-check -q *> $null 2>&1 } catch { Write-Warning "[WARN] Echec installation requirements.txt: $($_.Exception.Message)" }
  }
}

# 2) Verifier/Demarrer Ollama si necessaire
if (-not (Test-OllamaUp -Port $OllamaPort)) {
  if ($AutoStartOllama) {
    Start-OllamaAuto -Exe $OllamaExe -Port $OllamaPort -LogDir (Get-Location)
  } else {
    Write-Warning "[WARN] Ollama ne repond pas sur 127.0.0.1:$OllamaPort. L'index peut echouer si les LLM/embeddings ne sont pas accessibles. (Utilisez -AutoStartOllama pour demarrer automatiquement)"
  }
}

# 2bis) Environnement GPU pour indexation (laisser Ollama en mode auto)
Remove-Item Env:\OLLAMA_NO_GPU -ErrorAction SilentlyContinue | Out-Null
Remove-Item Env:\OLLAMA_LLM_LIBRARY -ErrorAction SilentlyContinue | Out-Null
Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue | Out-Null

if (-not $ChunksExport -or $ChunksExport.Trim().Length -eq 0) {
  $ChunksExport = Join-Path $PersistDir 'chunks_export.jsonl'
}

# 3) Executer Python utilitaire: build index + export des chunks
$embeddingNumGpuLiteral = if ($UseGpuIndex) { 'None' } else { '0' }

$argsList = @(
  '-m','app.utils.build_index',
  '--data-dir', $DataDir,
  '--persist-dir', $PersistDir,
  '--llm-model', $LlmModel,
  '--embedding-model', $EmbeddingModel,
  '--llm-num-ctx', $LlmNumCtx,
  '--embedding-num-gpu', $embeddingNumGpuLiteral,
  '--export-chunks', $ChunksExport
)

Set-Location $ProjectRoot
if (Test-Path $venvPython) {
  & $venvPython @argsList
  $exit = $LASTEXITCODE
} else {
  python @argsList
  $exit = $LASTEXITCODE
}

if ($exit -ne 0) {
  Write-Error "[FAIL] Reconstruction echouee (code=$exit). Verifiez que Ollama repond sur le port $OllamaPort et que les modeles sont disponibles."
} else {
  Write-Host "[DONE] Reconstruction terminee."
}
