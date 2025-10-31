<#
Purge-Vectorstore.ps1
Supprime les vecteurs/indices persistés dans le dossier de stockage (Chroma + LlamaIndex).

Par défaut, efface le CONTENU du dossier `vectorstore` (conserve le dossier).
Options:
  -VectorstoreDir  Chemin du dossier de persistance (défaut: ./vectorstore)
  -Hard           Supprime le dossier complet (et son contenu)
  -Backup         Sauvegarde le dossier dans un dossier horodaté avant suppression
  -WhatIf         Prévisualise sans effectuer d’actions

Exemples:
  .\Purge-Vectorstore.ps1
  .\Purge-Vectorstore.ps1 -Hard
  .\Purge-Vectorstore.ps1 -Backup
  .\Purge-Vectorstore.ps1 -VectorstoreDir .\data_store -Hard -Backup
#>

[CmdletBinding(SupportsShouldProcess=$true)]
param(
  [string]$VectorstoreDir = (Join-Path $PSScriptRoot 'vectorstore'),
  [switch]$Hard,
  [switch]$Backup
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "📁 Dossier vectorstore ciblé : $VectorstoreDir"

if (-not (Test-Path -Path $VectorstoreDir)) {
  Write-Host "ℹ️  Le dossier spécifié n’existe pas. Rien à purger." -ForegroundColor Yellow
  exit 0
}

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
if ($Backup) {
  $backupDir = Join-Path (Split-Path -Path $VectorstoreDir -Parent) ("vectorstore_backup_$timestamp")
  if ($PSCmdlet.ShouldProcess($VectorstoreDir, "Sauvegarde vers $backupDir")) {
    Write-Host "🗃️  Sauvegarde du contenu vers : $backupDir"
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    Copy-Item -Path (Join-Path $VectorstoreDir '*') -Destination $backupDir -Recurse -Force -ErrorAction SilentlyContinue
  }
}

if ($Hard) {
  if ($PSCmdlet.ShouldProcess($VectorstoreDir, 'Suppression complète du dossier')) {
    Write-Host "🧹 Suppression complète du dossier…"
    Remove-Item -LiteralPath $VectorstoreDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "✅ Purge terminée (mode Hard)."
  }
} else {
  if ($PSCmdlet.ShouldProcess($VectorstoreDir, 'Purge du contenu (conserver le dossier)')) {
    Write-Host "🧹 Purge du contenu (conservation du dossier)…"
    Get-ChildItem -LiteralPath $VectorstoreDir -Force | ForEach-Object {
      Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host "✅ Purge terminée (contenu vidé)."
  }
}

Write-Host "ℹ️  Au prochain clic sur \"📥 Charger & indexer\", l’index sera reconstruit depuis les documents."

