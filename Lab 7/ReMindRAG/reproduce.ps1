# ReMindRAG Reproduction Script
# This script automates the environment check and demo execution.

$VENV_DIR = "venv"
$EXAMPLE_SCRIPT = "example/example.py"

Write-Host "--- ReMindRAG Reproduction Start ---" -ForegroundColor Cyan

# 1. Check for Virtual Environment
if (-Not (Test-Path $VENV_DIR)) {
    Write-Host "[!] Virtual environment '$VENV_DIR' not found." -ForegroundColor Yellow
    Write-Host "[*] Please create it first using: python -m venv venv"
    exit 1
}

# 2. Check for .env file
if (-Not (Test-Path ".env")) {
    Write-Host "[!] .env file not found." -ForegroundColor Yellow
    if (Test-Path ".env.template") {
        Write-Host "[*] Copying .env.template to .env. Please update it with your API keys."
        Copy-Item ".env.template" ".env"
    } else {
        Write-Host "[*] Please create a .env file with OPENAI_API_KEY."
    }
    exit 1
}

# 3. Activate and Run
Write-Host "[*] Activating environment and running demo..." -ForegroundColor Green
& "$VENV_DIR/Scripts/python.exe" $EXAMPLE_SCRIPT

if ($LASTEXITCODE -eq 0) {
    Write-Host "--- Execution Successful ---" -ForegroundColor Green
} else {
    Write-Host "--- Execution Failed ---" -ForegroundColor Red
    exit $LASTEXITCODE
}
