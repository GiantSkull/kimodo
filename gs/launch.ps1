# Copyright Giant Skull: daniel@giantskull.com
# Starts GS Kimodo and opens it in the browser. Runs setup.ps1 first when the environment is missing.
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$Url = "http://127.0.0.1:7860"

if (-not (Test-Path $Py)) {
    & (Join-Path $PSScriptRoot "setup.ps1")
}

# torch derives a cache folder from the user name; set it so locked-down shells work too.
if (-not $env:USERNAME) { $env:USERNAME = "gskimodo" }
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $env:LOCALAPPDATA "GSKimodo\torchinductor"

# The text model needs ~17 GB of GPU memory; on smaller cards run it on the CPU (slower prompts, under 3 GB VRAM).
$vram = 0
$mem = & nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
if ($mem) { $vram = [int]$mem }
if ($vram -lt 20000 -and -not $env:TEXT_ENCODER_DEVICE) { $env:TEXT_ENCODER_DEVICE = "cpu" }

# Already running? Just open it.
try {
    Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2 | Out-Null
    Start-Process $Url
    exit 0
} catch {}

Write-Host "Starting GS Kimodo (the first start downloads the models and takes a few minutes)..."
$proc = Start-Process -FilePath $Py -ArgumentList "-m", "kimodo.demo" -WorkingDirectory $Repo -NoNewWindow -PassThru
for ($i = 0; $i -lt 900; $i++) {
    if ($proc.HasExited) { throw "GS Kimodo stopped while starting (see the messages above)." }
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2 | Out-Null
        Start-Process $Url
        break
    } catch { Start-Sleep -Seconds 1 }
}
Write-Host "GS Kimodo is running at $Url - close this window to stop it."
$proc.WaitForExit()
