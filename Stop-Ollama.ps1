<# 
Stop-Ollama.ps1
Arrête proprement Ollama si il tourne.
#>

[CmdletBinding()]
param(
    [string]$ProcessName = "ollama"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$proc = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
if ($null -eq $proc) {
    Write-Host "ℹ️  Le processus '$ProcessName' n'est pas en cours d'exécution."
    exit 0
}

Write-Host "🛑 Arrêt du processus '$ProcessName' (PID: $($proc.Id))"
try {
    Stop-Process -Id $proc.Id -Force -ErrorAction Stop
    Write-Host "✅ Ollama arrêté."
} catch {
    Write-Host "❌ Impossible d'arrêter '$ProcessName' automatiquement : $($_.Exception.Message)"
    Write-Host "Essayez : taskkill /IM $ProcessName.exe /F"
    exit 1
}
