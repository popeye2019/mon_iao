<#
Start-Ollama-CUDA-Test.ps1

But: Lancer Ollama (mode auto, sans forcer CPU/GPU),
verifier la consommation VRAM (nvidia-smi), et faire une generation de test.

Exemples:
  .\Start-Ollama-CUDA-Test.ps1
  .\Start-Ollama-CUDA-Test.ps1 -Model "mistral" -NumCtx 2048 -NumPredict 256 `
    -Temperature 0.7 -TopP 0.9 -TopK 40 -RepeatPenalty 1.1 -Seed 123 `
    -Stop "\n\nUtilisateur:","\nAssistant:"
#>

[CmdletBinding()]
param(
  [string]$OllamaExe = "C:\\Users\\franc\\AppData\\Local\\Programs\\Ollama\\ollama.exe",
  [int]$Port = 11434,
  [string]$Model = "mistral",
  [int]$CudaDevice = 0,
  [int]$NumCtx = 1536,
  [int]$NumPredict = 256,
  [int]$NumGpuLayers = 12,
  [string]$Prompt = "Decris brievement Paris en francais.",
  [int]$PollSeconds = 10,
  # Qualite / Style (tous optionnels, laissent les valeurs par defaut du modele si non fournis)
  [Nullable[Double]]$Temperature = $null,
  [Nullable[Double]]$TopP = $null,
  [Nullable[Int32]]$TopK = $null,
  [Nullable[Double]]$RepeatPenalty = $null,
  [string[]]$Stop = $null,
  [Nullable[Int32]]$Seed = $null
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Test-OllamaUp {
  param([int]$P)
  try {
    $r = Invoke-WebRequest -UseBasicParsing -Uri ("http://127.0.0.1:{0}/api/tags" -f $P) -TimeoutSec 2
    return $r.StatusCode -eq 200
  } catch { return $false }
}

function Ensure-OllamaStopped {
  # Non utilise ici (conserve pour reference)
  $svc = Get-Service -Name 'Ollama' -ErrorAction SilentlyContinue
  if ($svc -and $svc.Status -eq 'Running') {
    Write-Host "[INFO] Arret du service Windows 'Ollama'"
    Stop-Service -Name 'Ollama' -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
  }
  Get-Process -Name 'ollama' -ErrorAction SilentlyContinue | ForEach-Object {
    try { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue } catch {}
  }
}

function Start-OllamaCuda {
  param([string]$Exe, [int]$P, [string]$LogDir)
  if (-not (Test-Path $Exe)) { throw "Ollama.exe introuvable: $Exe" }

  $ts = Get-Date -Format 'yyyyMMdd-HHmmss'
  $outLog = Join-Path $LogDir ("ollama-cuda-{0}.out.log" -f $ts)
  $errLog = Join-Path $LogDir ("ollama-cuda-{0}.err.log" -f $ts)

  # Laisser Ollama choisir automatiquement CPU/GPU (aucun parametre force)
  Remove-Item Env:\OLLAMA_NO_GPU -ErrorAction SilentlyContinue | Out-Null
  Remove-Item Env:\OLLAMA_LLM_LIBRARY -ErrorAction SilentlyContinue | Out-Null
  Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue | Out-Null
  Remove-Item Env:\HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue | Out-Null
  Remove-Item Env:\ROCR_VISIBLE_DEVICES -ErrorAction SilentlyContinue | Out-Null

  Write-Host "[INFO] Demarrage d'Ollama (auto, exe $Exe)"
  $proc = Start-Process -FilePath $Exe -ArgumentList 'serve' -NoNewWindow -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog
  # Attente disponibilite API
  for ($i=0; $i -lt 30; $i++) {
    if (Test-OllamaUp -P $P) { break }
    Start-Sleep -Seconds 1
  }
  if (-not (Test-OllamaUp -P $P)) { throw "Ollama ne repond pas sur 127.0.0.1:$P. Voir logs: `n$outLog`n$errLog" }
  Write-Host "[OK] Ollama pret sur http://127.0.0.1:$P"
}

function Ensure-Model {
  param([string]$Exe, [string]$Name)
  try {
    $tags = (Invoke-RestMethod -UseBasicParsing -Uri "http://127.0.0.1:$Port/api/tags").models
    if ($tags -and ($tags.name -contains $Name -or $tags.name -contains ($Name+':latest'))) {
      Write-Host "[OK] Modele deja present: $Name"
      return
    }
  } catch {}
  Write-Host "[INFO] Pull du modele: $Name"
  & $Exe pull $Name | Write-Host
}

function Get-GpuMemUsedMb {
  param([int]$Dev)
  $nvsmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
  if (-not $nvsmi) { return $null }
  $val = & $nvsmi.Path --query-gpu=memory.used --format=csv,noheader,nounits -i $Dev 2>$null
  if ($LASTEXITCODE -ne 0) { return $null }
  return [int]$val
}

function Show-GpuUsageSample {
  param([int]$Dev, [int]$Seconds)
  $m = Get-GpuMemUsedMb -Dev $Dev
  if ($null -eq $m) {
    Write-Host "[WARN] nvidia-smi non trouve. Ouvrez le Gestionnaire des taches pour observer la VRAM."
    return
  }
  Write-Host ("[INFO] VRAM utilisee (GPU {0}) avant generation: {1} MiB" -f $Dev, $m)
  Write-Host "[INFO] Echantillonnage VRAM pendant $Seconds s"
  for ($i=0; $i -lt $Seconds; $i++) {
    $cur = Get-GpuMemUsedMb -Dev $Dev
    if ($cur -ne $null) { Write-Host ("  t+{0,2}s : {1} MiB" -f $i, $cur) }
    Start-Sleep -Seconds 1
  }
  $m2 = Get-GpuMemUsedMb -Dev $Dev
  if ($m2 -ne $null) { Write-Host ("[INFO] VRAM utilisee apres: {0} MiB" -f $m2) }
}

# 1) Lancer Ollama si necessaire (mode auto, sans forcer CPU/GPU)
if (-not (Test-OllamaUp -P $Port)) {
  Start-OllamaCuda -Exe $OllamaExe -P $Port -LogDir (Get-Location)
} else {
  Write-Host "[OK] Ollama deja en cours sur http://127.0.0.1:$Port"
}

# 2) Pull du modele si necessaire
Ensure-Model -Exe $OllamaExe -Name $Model

# 3) Generation de test (francais)
# Construire dynamiquement les options en n'ajoutant que les parametres fournis
$opts = @{ num_ctx = $NumCtx; num_predict = $NumPredict }
if ($PSBoundParameters.ContainsKey('Temperature')) { $opts.temperature = [double]$Temperature }
if ($PSBoundParameters.ContainsKey('TopP'))         { $opts.top_p       = [double]$TopP }
if ($PSBoundParameters.ContainsKey('TopK'))         { $opts.top_k       = [int]$TopK }
if ($PSBoundParameters.ContainsKey('RepeatPenalty')){ $opts.repeat_penalty = [double]$RepeatPenalty }
if ($PSBoundParameters.ContainsKey('Stop') -and $Stop) { $opts.stop = $Stop }
if ($PSBoundParameters.ContainsKey('Seed'))         { $opts.seed        = [int]$Seed }

$body = @{
  model = $Model
  system = "Reponds en francais, de maniere concise."
  prompt = $Prompt
  options = $opts
  stream = $false  # recuperer un seul objet reponse avec eval_count/duration
} | ConvertTo-Json -Depth 6

Write-Host "[INFO] Generation de test"; Write-Host "[INFO] Prompt: $Prompt"

# Echantillonnage VRAM en parallele
$job = Start-Job -ScriptBlock {
  param($dev, $secs)
  $nvsmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
  function Get-GpuMemUsedMb([int]$D){ if(-not $nvsmi){ return $null }; $v = & $nvsmi.Path --query-gpu=memory.used --format=csv,noheader,nounits -i $D 2>$null; if($LASTEXITCODE -ne 0){return $null}; return [int]$v }
  $m = Get-GpuMemUsedMb $dev; if($m -ne $null){ Write-Output ("VRAM start: {0} MiB" -f $m) }
  for($i=0; $i -lt $secs; $i++){ $cur = Get-GpuMemUsedMb $dev; if($cur -ne $null){ Write-Output ("t+{0}s: {1} MiB" -f $i,$cur) }; Start-Sleep -Seconds 1 }
} -ArgumentList $CudaDevice, $PollSeconds

try {
  $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/generate" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600
  Write-Host "[OK] Reponse: " ($resp.response | Out-String)
  # Stats: eval_count, eval_duration, token/s
  if ($resp -and $resp.eval_duration -ne $null -and $resp.eval_count -ne $null) {
    $evalSec = [double]$resp.eval_duration / 1e9
    if ($evalSec -gt 0) {
      $tps = [double]$resp.eval_count / $evalSec
      Write-Host ("[STATS] eval_count={0} eval_duration={1}ns => {2:N2} tok/s" -f $resp.eval_count, $resp.eval_duration, $tps)
    } else {
      Write-Host ("[STATS] eval_count={0} eval_duration={1}ns" -f $resp.eval_count, $resp.eval_duration)
    }
  }
} catch {
  Write-Warning "[WARN] Echec de generation: $($_.Exception.Message)"
}

Receive-Job -Job $job -Wait -AutoRemoveJob | ForEach-Object { Write-Host $_ }

Write-Host "[INFO] Test termine. Si la VRAM n'a pas varie, verifiez que 'nvidia-smi' est installe et que le device $CudaDevice correspond au GPU NVIDIA."
