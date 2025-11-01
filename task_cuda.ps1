param(
    [int]$Interval = 1,
    [string]$LogPath
)

# -----------------------------
# Helpers
# -----------------------------
function Test-NvidiaSmi {
    try {
        $null = & nvidia-smi --help 2>$null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

function Get-NvidiaGpuSummary {
    # Returns: @{ UtilGPU=..; UtilMem=..; MemUsed=..; MemTotal=..; Temp=..; Power=.. }
    $o = @{
        UtilGPU  = $null
        UtilMem  = $null
        MemUsed  = $null
        MemTotal = $null
        Temp     = $null
        Power    = $null
    }
    try {
        $line = & nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw `
            --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ($line) {
            $p = $line -split ',\s*'
            if ($p.Count -ge 6) {
                $o.UtilGPU  = [int]$p[0]
                $o.UtilMem  = [int]$p[1]
                $o.MemUsed  = [int]$p[2]
                $o.MemTotal = [int]$p[3]
                $o.Temp     = [int]$p[4]
                $o.Power    = [double]$p[5]
            }
        }
    } catch {}
    return $o
}

function Get-NvidiaPerProcess {
    # Returns objects: PID, ProcName, MemUsedMB
    $list = @()
    try {
        $lines = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null
        foreach ($l in $lines) {
            if (-not $l) { continue }
            $p = $l -split ',\s*'
            if ($p.Count -ge 3) {
                $list += [pscustomobject]@{
                    PID       = [int]$p[0]
                    ProcName  = $p[1]
                    MemUsedMB = [int]$p[2]
                    UtilPct   = $null  # rempli via counters si dispo
                }
            }
        }
    } catch {}
    return $list
}

function Get-GpuCountersAvailable {
    try {
        $ls1 = Get-Counter -ListSet 'GPU Engine' -ErrorAction Stop | Out-Null
        $ls2 = Get-Counter -ListSet 'GPU Adapter Memory' -ErrorAction Stop | Out-Null
        return $true
    } catch { return $false }
}

function Get-GpuCountersSnapshot {
    # Returns: @{ Engines=<samples>; Mem=<samples> }
    $result = @{
        Engines = $null
        Mem     = $null
    }
    try {
        $eng = Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction Stop
        $mem = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop
        $result.Engines = $eng
        $result.Mem     = $mem
    } catch {}
    return $result
}

function Parse-EngineUtilization {
    param(
        $engSamples
    )
    # Build per-PID utilization by summing all engine instances that include pid_XXXX
    $perPID = @{}
    if (-not $engSamples) { return $perPID }

    $samples = $engSamples.CounterSamples
    foreach ($s in $samples) {
        $inst = $s.Path
        $val  = [double]$s.CookedValue
        # Instance path example includes ...\GPU Engine(pid_1234_engtype_3D_...)\Utilization Percentage
        if ($inst -match 'pid_(\d+)') {
            $pid = [int]$Matches[1]
            if (-not $perPID.ContainsKey($pid)) { $perPID[$pid] = 0.0 }
            $perPID[$pid] += $val
        }
    }
    return $perPID
}

function Parse-TotalGPU {
    param($engSamples)
    if (-not $engSamples) { return $null }
    # Sum across all engines then clamp 0..100 (Windows may over-sum)
    $sum = ($engSamples.CounterSamples | Measure-Object -Property CookedValue -Sum).Sum
    if ($null -eq $sum) { return $null }
    return [math]::Min([math]::Max([int][math]::Round($sum,0),0),100)
}

function Parse-VramMB {
    param($memSamples)
    if (-not $memSamples) { return $null, $null }

    # Dedicated Usage is in bytes per adapter instance
    $vals = $memSamples.CounterSamples | ForEach-Object { [double]$_.CookedValue }
    if (-not $vals -or $vals.Count -eq 0) { return $null, $null }

    # If multiple adapters, sum usage; total is not directly exposed—fallback to NVIDIA for total if present.
    $usedMB = [int][math]::Round(($vals | Measure-Object -Sum).Sum / 1MB, 0)
    return $usedMB, $null
}

# Init
$hasNVSmi  = Test-NvidiaSmi
$hasCtrs   = Get-GpuCountersAvailable

if ($LogPath) {
    if (-not (Test-Path $LogPath)) {
        "Timestamp,UtilGPU,VRAM_UsedMB,TempC,PowerW" | Out-File -FilePath $LogPath -Encoding utf8
    }
}

Write-Host "GPU monitor en cours. Intervalle: $Interval s  |  Quitter: Ctrl+C"
if ($hasCtrs)   { Write-Host "Compteurs Windows: OK" } else { Write-Host "Compteurs Windows: indisponibles" }
if ($hasNVSmi)  { Write-Host "nvidia-smi: OK (temp/power/processus)" } else { Write-Host "nvidia-smi: non détecté ou indisponible" }
Start-Sleep -Seconds 1

# -----------------------------
# Main loop
# -----------------------------
while ($true) {
    $ts = Get-Date

    $ctrSnap = $null
    $perPidUtil = @{}
    $totalGPU = $null
    $vramUsedMB = $null
    $vramTotalMB = $null

    if ($hasCtrs) {
        $ctrSnap = Get-GpuCountersSnapshot
        if ($ctrSnap.Engines) {
            $perPidUtil = Parse-EngineUtilization -engSamples $ctrSnap.Engines
            $totalGPU   = Parse-TotalGPU -engSamples $ctrSnap.Engines
        }
        if ($ctrSnap.Mem) {
            $memRes = Parse-VramMB -memSamples $ctrSnap.Mem
            $vramUsedMB  = $memRes[0]
            $vramTotalMB = $memRes[1]  # souvent null côté Windows counters
        }
    }

    $nv = @{ UtilGPU=$null; UtilMem=$null; MemUsed=$null; MemTotal=$null; Temp=$null; Power=$null }
    $perProc = @()
    if ($hasNVSmi) {
        $nv = Get-NvidiaGpuSummary
        $perProc = Get-NvidiaPerProcess
        # Map utilisation (counters) vers la liste nvidia (par PID)
        foreach ($pp in $perProc) {
            if ($perPidUtil.ContainsKey($pp.PID)) {
                $pp.UtilPct = [int][math]::Round($perPidUtil[$pp.PID],0)
            }
        }
    }

    # Choix des valeurs globales “meilleures dispo”
    $displayUtil = $totalGPU
    if ($null -eq $displayUtil -and $nv.UtilGPU -ne $null) { $displayUtil = $nv.UtilGPU }

    $displayMemUsed = $vramUsedMB
    if ($null -eq $displayMemUsed -and $nv.MemUsed -ne $null) { $displayMemUsed = $nv.MemUsed }

    $displayMemTotal = $vramTotalMB
    if ($null -eq $displayMemTotal -and $nv.MemTotal -ne $null) { $displayMemTotal = $nv.MemTotal }

    $tempC = $nv.Temp
    $powerW = $nv.Power

    # Render
    Clear-Host
    Write-Host ("[{0}] GPU: {1}%  | VRAM: {2}{3}  | Temp: {4}°C  | Power: {5} W" -f `
        $ts.ToString("HH:mm:ss"),
        ($displayUtil -ne $null ? $displayUtil : "n/a"),
        ($displayMemUsed -ne $null ? ("{0} MB" -f $displayMemUsed) : "n/a"),
        ($displayMemTotal -ne $null ? ("/{0} MB" -f $displayMemTotal) : ""),
        ($tempC -ne $null ? $tempC : "n/a"),
        ($powerW -ne $null ? [int][math]::Round($powerW,0) : "n/a")
    )

    # Tableau par processus (Top 10)
    $rows = @()

    if ($perProc.Count -gt 0) {
        $rows = $perProc | Sort-Object @{Expression='UtilPct';Descending=$true}, @{Expression='MemUsedMB';Descending=$true} `
            | Select-Object -First 10 `
            | ForEach-Object {
                $name = $_.ProcName
                if (-not $name) {
                    try { $name = (Get-Process -Id $_.PID -ErrorAction SilentlyContinue).ProcessName } catch {}
                }
                [pscustomobject]@{
                    PID     = $_.PID
                    Name    = $name
                    "GPU %"<<# --> prevent syntax issues? In PS property names with space allowed in hashtable? We'll use GPUUtil instead
                }
            }
    } else {
        # Construire à partir des compteurs Windows si pas de nvidia-smi
        if ($perPidUtil.Keys.Count -gt 0) {
            $rows = $perPidUtil.GetEnumerator() | ForEach-Object {
                $pid = [int]$_.Key
                $util = [int][math]::Round([double]$_.Value,0)
                $pname = $null
                try { $pname = (Get-Process -Id $pid -ErrorAction SilentlyContinue).ProcessName } catch {}
                [pscustomobject]@{
                    PID     = $pid
                    Name    = $pname
                    GPUUtil = $util
                    MemMB   = $null
                }
            } | Sort-Object GPUUtil -Descending | Select-Object -First 10
        }
    }

    # Si on avait construit $rows via nvidia-smi, on le finalise maintenant (ajout champs standardisés)
    if ($perProc.Count -gt 0) {
        $rows = $perProc | Sort-Object @{Expression='UtilPct';Descending=$true}, @{Expression='MemUsedMB';Descending=$true} `
            | Select-Object -First 10 `
            | ForEach-Object {
                $name = $_.ProcName
                if (-not $name) {
                    try { $name = (Get-Process -Id $_.PID -ErrorAction SilentlyContinue).ProcessName } catch {}
                }
                [pscustomobject]@{
                    PID     = $_.PID
                    Name    = $name
                    GPUUtil = ($_.UtilPct -ne $null ? $_.UtilPct : 0)
                    MemMB   = $_.MemUsedMB
                }
            }
    }

    if ($rows -and $rows.Count -gt 0) {
        Write-Host ""
        $rows | Format-Table -AutoSize
    } else {
        Write-Host "`n(aucun processus GPU actif détecté via les sources disponibles)"
    }

    # Log CSV global (optionnel)
    if ($LogPath) {
        $line = ('{0},{1},{2},{3},{4}' -f `
            $ts.ToString('s'),
            ($displayUtil -ne $null ? $displayUtil : ''),
            ($displayMemUsed -ne $null ? $displayMemUsed : ''),
            ($tempC -ne $null ? $tempC : ''),
            ($powerW -ne $null ? [int][math]::Round($powerW,0) : '')
        )
        Add-Content -Path $LogPath -Value $line
    }

    Start-Sleep -Seconds $Interval
}
