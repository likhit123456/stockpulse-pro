# ════════════════════════════════════════════════════════════
#  StockPulse Pro v3.0 — Windows startup script (PowerShell)
#  Usage: .\run.ps1
#  If blocked by execution policy, run once:
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# ════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

function Info  { param($msg) Write-Host "[INFO]  $msg" -ForegroundColor Green }
function Warn  { param($msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Abort { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# ── Check Python ────────────────────────────────────────────
try { $pyver = python --version 2>&1 } catch { Abort "Python is not installed or not in PATH" }
Info "Detected: $pyver"

# ── Create / activate venv ──────────────────────────────────
if (-not (Test-Path "venv")) {
    Info "Creating virtual environment..."
    python -m venv venv
}
Info "Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"

# ── Install dependencies ────────────────────────────────────
Info "Installing / updating dependencies..."
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
Info "Dependencies installed"

# ── Check .env ──────────────────────────────────────────────
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Warn ".env not found — copying from .env.example"
        Copy-Item ".env.example" ".env"
    } else {
        Abort ".env not found and no .env.example to copy from. Create .env with your API keys."
    }
    Warn "Please edit .env and add your API keys, then re-run this script."
    exit 1
}

# Read and validate key env vars
Get-Content ".env" | ForEach-Object {
    $line = $_.Trim()
    # Skip blank lines and comments
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    $idx = $line.IndexOf("=")
    if ($idx -lt 1) { return }
    $key = $line.Substring(0, $idx).Trim()
    $val = $line.Substring($idx + 1).Trim().Trim('"').Trim("'")
    # Set at both Process AND Machine scope so uvicorn subprocess inherits them
    [System.Environment]::SetEnvironmentVariable($key, $val, "Process")
    Set-Item -Path "Env:\$key" -Value $val
}

if (-not $env:GROQ_API_KEY)    { Abort "GROQ_API_KEY is not set in .env" }
if (-not $env:ANAKIN_API_KEY)  { Abort "ANAKIN_API_KEY is not set in .env" }

$groqPreview   = $env:GROQ_API_KEY.Substring(0, [Math]::Min(8, $env:GROQ_API_KEY.Length)) + "…"
$anakinPreview = $env:ANAKIN_API_KEY.Substring(0, [Math]::Min(8, $env:ANAKIN_API_KEY.Length)) + "…"
Info "GROQ_API_KEY    loaded: $groqPreview"
Info "ANAKIN_API_KEY  loaded: $anakinPreview"

# ── Check static folder ─────────────────────────────────────
if (-not (Test-Path "static\index.html")) {
    Abort "static\index.html not found. Put your frontend at static\index.html"
}

# ── Start server ────────────────────────────────────────────
$BindHost = if ($env:HOST)   { $env:HOST }   else { "0.0.0.0" }
$BindPort = if ($env:PORT)   { $env:PORT }   else { "8000" }
$Reload   = if ($env:RELOAD) { $env:RELOAD } else { "true" }

Info "Starting StockPulse Pro on http://${BindHost}:${BindPort}"
Info "API docs: http://${BindHost}:${BindPort}/docs"
Info "Press Ctrl+C to stop"

if ($Reload -eq "true") {
    uvicorn main:app --host $BindHost --port $BindPort --reload --log-level info
} else {
    uvicorn main:app --host $BindHost --port $BindPort --log-level info
}