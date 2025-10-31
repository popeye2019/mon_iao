<# 
Who-Uses-11434.ps1
Affiche quel PID utilise le port 11434 (Ollama) et le nom du processus.
#>
$lines = netstat -ano | findstr ":11434"
if (-not $lines) {
    Write-Host "ℹ️  Aucun processus n'écoute sur :11434"
    exit 0
}
$lines | ForEach-Object {
    $parts = ($_ -replace "\s+", " ").Trim().Split(" ")
    $pid = $parts[-1]
    $p = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($p) {
        "{0} => PID {1} ({2})" -f $_, $pid, $p.ProcessName
    } else {
        "{0} => PID {1} (inconnu)" -f $_, $pid
    }
}
