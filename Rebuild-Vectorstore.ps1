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
  [bool]$UseGpuIndex = $true
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

# 2) Verifier Ollama
if (-not (Test-OllamaUp -Port $OllamaPort)) {
  Write-Warning "[WARN] Ollama ne repond pas sur 127.0.0.1:$OllamaPort. L'index peut echouer si les LLM/embeddings ne sont pas accessibles."
}

# 2bis) Environnement GPU pour indexation (laisser Ollama en mode auto)
Remove-Item Env:\OLLAMA_NO_GPU -ErrorAction SilentlyContinue | Out-Null
Remove-Item Env:\OLLAMA_LLM_LIBRARY -ErrorAction SilentlyContinue | Out-Null
Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue | Out-Null

# 3) Executer Python: charger docs et (re)construire l'index
$embeddingNumGpuLiteral = if ($UseGpuIndex) { 'None' } else { '0' }
$py = @"
import os
from app.loader import load_documents
from app.indexer import build_or_load_index, get_vector_count

data_dir = r"""$DataDir"""
persist_dir = r"""$PersistDir"""

os.makedirs(persist_dir, exist_ok=True)

docs = load_documents(data_dir)
idx = build_or_load_index(
    data_documents=docs,
    persist_dir=persist_dir,
    llm_name=r"""$LlmModel""",
    embedding_name=r"""$EmbeddingModel""",
    llm_num_ctx=int($LlmNumCtx),
    embedding_num_gpu=$embeddingNumGpuLiteral,
)
count = get_vector_count(persist_dir)
print(f"OK - vecteurs: {count}")
"@

Set-Location $ProjectRoot
if (Test-Path $venvPython) {
  $py | & $venvPython -
} else {
  $py | python -
}

Write-Host "[DONE] Reconstruction terminee."
