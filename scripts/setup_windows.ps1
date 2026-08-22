# Windows / CUDA setup for the world-model conda env.
#
# RTX 50-series (Blackwell, sm_120) needs a CUDA 12.8+ PyTorch wheel.
# The default `pip install torch` from PyPI is CPU-only on Windows and will
# silently skip the GPU.
#
# Usage (from repo root, in PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Find-Conda {
    $candidates = @(
        "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniforge3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) { return $path }
    }
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -and (Test-Path $cmd.Source)) {
        return $cmd.Source
    }
    return $null
}

Write-Host "==> nvidia-smi"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi not found. Install an NVIDIA Game Ready / Studio driver first."
}
nvidia-smi

$Conda = Find-Conda
if (-not $Conda) {
    throw "conda not found. Install Miniforge: winget install -e --id CondaForge.Miniforge3 --accept-package-agreements --accept-source-agreements"
}

# Make this process see conda even if PATH was not refreshed after install.
$CondaRoot = Split-Path (Split-Path $Conda)
$env:Path = "$CondaRoot;$CondaRoot\Scripts;$CondaRoot\Library\bin;" + $env:Path

Write-Host "`n==> conda ($Conda)"
& $Conda --version
if ($LASTEXITCODE -ne 0) { throw "conda --version failed (exit $LASTEXITCODE)" }

$EnvName = "worldmodel"
$EnvList = & $Conda env list | Out-String
if ($EnvList -notmatch "(?m)^\s*$EnvName\s") {
    Write-Host "`n==> creating conda env '$EnvName' (Python 3.11)"
    & $Conda env create -f environment.yml
    if ($LASTEXITCODE -ne 0) { throw "conda env create failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n==> conda env '$EnvName' already exists - skipping create"
}

Write-Host "`n==> installing CUDA 12.8 PyTorch (Blackwell / RTX 5080)"
& $Conda run -n $EnvName python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE)" }
& $Conda run -n $EnvName python -m pip uninstall -y torch torchvision torchaudio
& $Conda run -n $EnvName python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "CUDA torch install failed (exit $LASTEXITCODE)" }

Write-Host "`n==> installing project (editable) + dev extras"
& $Conda run -n $EnvName python -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "pip install -e failed (exit $LASTEXITCODE)" }

Write-Host "`n==> registering Jupyter kernel"
& $Conda run -n $EnvName python -m ipykernel install --user --name worldmodel --display-name "Python (worldmodel)"
if ($LASTEXITCODE -ne 0) { throw "ipykernel install failed (exit $LASTEXITCODE)" }

Write-Host "`n==> CUDA sanity"
& $Conda run -n $EnvName python scripts/check_cuda.py
if ($LASTEXITCODE -ne 0) { throw "CUDA sanity check failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "Done. In a NEW PowerShell (so PATH picks up conda):"
Write-Host "    conda activate worldmodel"
Write-Host "    python scripts/verify_m0.py"
Write-Host "    python scripts/smoke_cuda_step.py"
Write-Host "    jupyter notebook notebooks/05_train_world_model.ipynb"
