<#
Show-Chunks.ps1

Petit utilitaire pour afficher les chunks indexés (Chroma) depuis PowerShell.
S'appuie sur app\inspect_chunks.py et le venv local s'il existe.

Exemples:
  .\Show-Chunks.ps1
  .\Show-Chunks.ps1 -PersistDir .\vectorstore -Limit 20 -GroupByFile
#>

[CmdletBinding()]
param(
  [string]$ProjectRoot = $PSScriptRoot,
  [string]$PersistDir = (Join-Path $PSScriptRoot 'vectorstore'),
  [string]$Collection = 'eau_docs',
  [int]$Limit = 50,
  [int]$Offset = 0,
  [switch]$GroupByFile,
  [string]$SourceFilter,
  [string]$ExportCsv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$venvPy = Join-Path $ProjectRoot 'env\Scripts\python.exe'
$argsList = @('app/inspect_chunks.py', '--persist-dir', $PersistDir, '--collection', $Collection, '--limit', $Limit, '--offset', $Offset)
if ($GroupByFile) { $argsList += '--group-by-file' }
if ($SourceFilter) { $argsList += @('--source-filter', $SourceFilter) }
if ($ExportCsv) { $argsList += @('--export-csv', $ExportCsv) }

Set-Location $ProjectRoot
if (Test-Path $venvPy) {
  & $venvPy @argsList
} else {
  python @argsList
}

