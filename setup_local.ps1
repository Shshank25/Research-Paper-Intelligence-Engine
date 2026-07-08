# ============================================================
# setup_local.ps1 — Local Setup Script
# Installs Miniconda on E: drive (bypasses C: disk full issue)
# and creates a dedicated conda environment for this project.
#
# Run with:  .\setup_local.ps1
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Research Paper Intelligence Engine" -ForegroundColor Cyan
Write-Host " Local Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── Configuration ────────────────────────────────────────────
$MINICONDA_DIR   = "E:\Miniconda3"
$CONDA_ENV_NAME  = "rag_env"
$PROJECT_DIR     = "E:\VT"
$INSTALLER_PATH  = "E:\miniconda_installer.exe"
$MINICONDA_URL   = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"

# ── Step 1: Check if conda already installed on E: ───────────
$conda_exe = "$MINICONDA_DIR\Scripts\conda.exe"

if (Test-Path $conda_exe) {
    Write-Host "Miniconda already installed at $MINICONDA_DIR" -ForegroundColor Green
} else {
    Write-Host "Downloading Miniconda to E: drive..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $MINICONDA_URL -OutFile $INSTALLER_PATH -UseBasicParsing
    
    Write-Host "Installing Miniconda to $MINICONDA_DIR ..." -ForegroundColor Yellow
    Start-Process -FilePath $INSTALLER_PATH -ArgumentList "/S", "/D=$MINICONDA_DIR" -Wait -NoNewWindow
    
    Write-Host "Miniconda installed." -ForegroundColor Green
    Remove-Item $INSTALLER_PATH -Force
}

# ── Step 2: Add conda to PATH for this session ───────────────
$env:PATH = "$MINICONDA_DIR\Scripts;$MINICONDA_DIR;$env:PATH"

# ── Step 3: Create conda environment ─────────────────────────
$env_exists = & "$conda_exe" env list | Select-String $CONDA_ENV_NAME

if ($env_exists) {
    Write-Host "Conda env '$CONDA_ENV_NAME' already exists." -ForegroundColor Green
} else {
    Write-Host "Creating conda env '$CONDA_ENV_NAME' with Python 3.10..." -ForegroundColor Yellow
    & "$conda_exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    & "$conda_exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    & "$conda_exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
    & "$conda_exe" create -n $CONDA_ENV_NAME python=3.10 -y
    Write-Host "Conda env created." -ForegroundColor Green
}

# ── Step 4: Install project dependencies ─────────────────────
$pip_exe = "$MINICONDA_DIR\envs\$CONDA_ENV_NAME\Scripts\pip.exe"

Write-Host ""
Write-Host "Installing project dependencies (this may take 5-10 mins)..." -ForegroundColor Yellow
$env:TMP = "E:\Temp"
$env:TEMP = "E:\Temp"
& "$pip_exe" install -r "$PROJECT_DIR\requirements.txt" --cache-dir "E:\pip_cache"

Write-Host ""
Write-Host "All dependencies installed!" -ForegroundColor Green

# ── Step 5: Create run script ─────────────────────────────────
$streamlit_exe = "$MINICONDA_DIR\envs\$CONDA_ENV_NAME\Scripts\streamlit.exe"

$run_script = @"
# Auto-generated launcher
`$env:PATH = "$MINICONDA_DIR\Scripts;$MINICONDA_DIR\envs\$CONDA_ENV_NAME\Scripts;`$env:PATH"
Set-Location "$PROJECT_DIR"
Write-Host "Launching Research Paper Intelligence Engine..." -ForegroundColor Cyan
Write-Host "   URL: http://localhost:8501" -ForegroundColor Green
& "$streamlit_exe" run "$PROJECT_DIR\app.py" --server.port 8501 --browser.gatherUsageStats false
"@

$run_script | Out-File -FilePath "$PROJECT_DIR\start_app.ps1" -Encoding utf8
Write-Host "Created start_app.ps1" -ForegroundColor Green

# ── Done ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the app:" -ForegroundColor White
Write-Host "   powershell -ExecutionPolicy Bypass -File E:\VT\start_app.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Or use Google Colab:" -ForegroundColor White
Write-Host "   Open: E:\VT\Research_Paper_Intelligence_Engine.ipynb" -ForegroundColor Yellow
Write-Host ""
