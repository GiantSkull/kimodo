# Copyright Giant Skull: daniel@giantskull.com
# One-time setup of GS Kimodo on a Windows PC with an NVIDIA GPU: Python environment, Kimodo, Blender FBX module and
# the studio Hugging Face token. Safe to run again (it repairs / updates the environment).
param(
    [string]$HfToken = $env:GS_KIMODO_HF_TOKEN
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Repo ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }

Step "Checking the graphics card"
$gpu = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
if (-not $gpu) { throw "No NVIDIA graphics card found (nvidia-smi failed). GS Kimodo needs an NVIDIA GPU." }
Write-Host $gpu

Step "Getting uv (Python installer)"
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $uv = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter uv.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $uv) {
    winget install --id astral-sh.uv -e --silent --accept-package-agreements --accept-source-agreements | Out-Host
    $uv = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter uv.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $uv) { throw "Could not install uv." }

Step "Creating the Python environment"
if (-not (Test-Path $Py)) { & $uv venv --python 3.11 $Venv; if ($LASTEXITCODE) { throw "uv venv failed" } }

Step "Installing PyTorch (CUDA) - large download the first time"
& $uv pip install --python $Py torch --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE) { throw "PyTorch install failed" }

Step "Installing Kimodo"
& $uv pip install --python $Py cmake
$env:PATH = "$(Join-Path $Venv 'Scripts');$env:PATH"
& $uv pip install --python $Py -e "$Repo[all]"
if ($LASTEXITCODE) { throw "Kimodo install failed (it compiles a C++ module: Visual Studio C++ build tools are required)" }

Step "Installing the Blender FBX module"
& $uv pip install --python $Py "bpy==4.2.*"
if ($LASTEXITCODE) { throw "bpy install failed" }

Step "Hugging Face access (Kimodo's text model)"
$tokenFile = Join-Path $env:USERPROFILE ".cache\huggingface\token"
if (-not $HfToken -and -not (Test-Path $tokenFile)) {
    $HfToken = Read-Host "Paste the studio Hugging Face token (ask your lead)"
}
if ($HfToken) {
    New-Item -ItemType Directory -Force (Split-Path $tokenFile) | Out-Null
    [IO.File]::WriteAllText($tokenFile, $HfToken.Trim())
    Write-Host "Token saved."
} else {
    Write-Host "Using the existing Hugging Face token."
}

Step "Checking the install"
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $env:LOCALAPPDATA "GSKimodo\torchinductor"
& $Py -c "import torch, kimodo, bpy; assert torch.cuda.is_available(), 'CUDA not available'; print('OK:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE) { throw "Install check failed" }
Write-Host "`nGS Kimodo is ready. Start it with GSKimodo.bat." -ForegroundColor Green
