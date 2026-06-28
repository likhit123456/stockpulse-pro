#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════
#  StockPulse Pro v3.0 — startup script  (Linux / macOS)
#  Usage:  chmod +x run.sh && ./run.sh
#
#  Windows (PowerShell) — use run.ps1 instead:
#    python -m venv venv
#    .\venv\Scripts\Activate.ps1
#    pip install -r requirements.txt
#    uvicorn main:app --reload --port 8000
# ════════════════════════════════════════════════════════════
set -euo pipefail

# Colour helpers
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Check Python ────────────────────────────────────────────
python3 --version &>/dev/null || error "Python 3 is not installed"
PYTHON_MIN="3.10"
PYTHON_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Python $PYTHON_VER detected"

# ── Check / create venv ─────────────────────────────────────
if [ ! -d "venv" ]; then
    info "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
info "Virtual environment activated"

# ── Install dependencies ────────────────────────────────────
info "Installing / updating dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
info "Dependencies installed"

# ── Check .env ──────────────────────────────────────────────
if [ ! -f ".env" ]; then
    warn ".env not found — copying from .env.example"
    cp .env.example .env
    warn "Please edit .env and add your API keys, then re-run this script."
    exit 1
fi

# Validate critical keys are set
source .env 2>/dev/null || true
if [ -z "${GROQ_API_KEY:-}" ]; then
    error "GROQ_API_KEY is not set in .env — AI features will not work"
fi
if [ -z "${ANAKIN_API_KEY:-}" ] || [ -z "${ANAKIN_APP_ID:-}" ]; then
    warn "ANAKIN_API_KEY or ANAKIN_APP_ID not set — stock scraping will not work"
fi

# ── Check static folder ─────────────────────────────────────
if [ ! -f "static/index.html" ]; then
    error "static/index.html not found. Put your frontend file at static/index.html"
fi

# ── Start server ────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
RELOAD="${RELOAD:-true}"

info "Starting StockPulse Pro on http://$HOST:$PORT"
info "API docs: http://$HOST:$PORT/docs"
info "Press Ctrl+C to stop"
echo ""

if [ "$RELOAD" = "true" ]; then
    uvicorn main:app --host "$HOST" --port "$PORT" --reload --log-level info
else
    uvicorn main:app --host "$HOST" --port "$PORT" --workers "$WORKERS" --log-level info
fi