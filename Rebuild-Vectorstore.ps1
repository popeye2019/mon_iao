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
  [string]$OllamaExe = "C:\\Users\\franc\\AppData\\Local\\Programs\\Ollama\\ollama.exe"
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

# 1) venv local (env) + deps
$venvDir = Join-Path $ProjectRoot 'env'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'
$reqFile = Join-Path $ProjectRoot 'requirements.txt'
if (-not (Test-Path $venvPython)) {
  Write-Host "[INFO] Creation du venv local 'env'"
  & python -m venv $venvDir
}
if (Test-Path $venvPython) {
  Write-Host "[INFO] MAJ pip + installation requirements"
  & $venvPython -m pip install --upgrade pip | Write-Host
  if (Test-Path $reqFile) { & $venvPython -m pip install -r $reqFile | Write-Host }
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

# 3) Executer Python: charger docs et (re)construire l'index
$embeddingNumGpuLiteral = if ($UseGpuIndex) { 'None' } else { '0' }
$py = @"
import os
import sys
import logging
from app.loader import load_documents
from app.indexer import build_or_load_index, get_vector_count

data_dir = r"""$DataDir"""
persist_dir = r"""$PersistDir"""

os.makedirs(persist_dir, exist_ok=True)

# Rendre les logs verbeux muets pendant l'indexation (remplace la ligne HTTP Request)
for _name in [
    'httpx', 'ollama', 'chromadb', 'llama_index', 'llama_index.core',
    'llama_index.embeddings', 'urllib3'
]:
    try:
        logging.getLogger(_name).setLevel(logging.WARNING)
    except Exception:
        pass

docs = load_documents(data_dir)
total = len(docs)
if total == 0:
    print("Aucun document a indexer.")
else:
    processed = 0
    # Taille de lot: maximum 64 docs ou ~10 lots
    import math
    batch_size = max(1, min(64, math.ceil(total / 10)))
    for i in range(0, total, batch_size):
        batch = docs[i:i+batch_size]
        idx = build_or_load_index(
            data_documents=batch,
            persist_dir=persist_dir,
            llm_name=r"""$LlmModel""",
            embedding_name=r"""$EmbeddingModel""",
            llm_num_ctx=int($LlmNumCtx),
            embedding_num_gpu=$embeddingNumGpuLiteral,
        )
        processed += len(batch)
        pct = int(processed * 100 / total)
        print(f"[{pct:3d}%] Indexation {processed}/{total}")
count = get_vector_count(persist_dir)
print(f"OK - vecteurs: {count}")
"@

Set-Location $ProjectRoot
if (Test-Path $venvPython) {
  $py | & $venvPython -
  $exit = $LASTEXITCODE
} else {
  $py | python -
  $exit = $LASTEXITCODE
}

if ($exit -ne 0) {
  Write-Error "[FAIL] Reconstruction echouee (code=$exit). Verifiez que Ollama repond sur le port $OllamaPort et que les modeles sont disponibles."
} else {
  Write-Host "[DONE] Reconstruction terminee."
}
