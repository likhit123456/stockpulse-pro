"""
StockPulse Pro — main.py  (v3.0 — Production-grade)
====================================================
Hardened FastAPI backend with:
  • Retry logic with exponential back-off on every external call
  • Per-source circuit breaker (stops hammering a dead endpoint)
  • Two-tier cache: hot (5 min) + warm (30 min stale-while-revalidate)
  • Rate limiting (per-IP via slowapi)
  • Structured JSON logging via structlog
  • Graceful degradation: analysis succeeds even if 3/6 sources fail
  • Input validation + sanitisation for every endpoint
  • Pydantic v2 models with strict field-level validation
  • Async background refresh of stale cache entries
  • /api/health returns full diagnostic with latency probes
  • /api/screener  — scan a basket of tickers and rank by signal strength
  • /api/technicals/{ticker} — pure technical analysis (no AI needed)
  • /api/news/{ticker} — latest news headlines only
"""

import asyncio
import logging
import os
import re
import time
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Auth import (after load_dotenv so env is ready)
from auth import register_user, login_user, get_current_user

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

ANAKIN_API_KEY   = os.getenv("ANAKIN_API_KEY", "")
ANAKIN_APP_ID    = os.getenv("ANAKIN_APP_ID", "")  # legacy/unused — anakin.io needs no app id
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
SUPABASE_URL     = os.getenv("SUPABASE_URL", "")
ENVIRONMENT      = os.getenv("ENVIRONMENT", "development")   # development | production
LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO")

_JWT_DEFAULT     = "stockpulse-secret-change-this"
JWT_SECRET       = os.getenv("JWT_SECRET", _JWT_DEFAULT)
if ENVIRONMENT == "production" and JWT_SECRET == _JWT_DEFAULT:
    raise RuntimeError(
        "JWT_SECRET must be set to a strong random secret in production. "
        "Run: python -c \"import secrets; print(secrets.token_hex(32))\" and add it to .env"
    )

CACHE_HOT_TTL    = int(os.getenv("CACHE_HOT_TTL",  "300"))   # 5 min   – fresh data
CACHE_WARM_TTL   = int(os.getenv("CACHE_WARM_TTL", "1800"))  # 30 min  – stale fallback
CACHE_MAX_ITEMS  = int(os.getenv("CACHE_MAX_ITEMS", "500"))

SCRAPE_TIMEOUT   = int(os.getenv("SCRAPE_TIMEOUT",  "30"))   # seconds per source
GROQ_TIMEOUT     = int(os.getenv("GROQ_TIMEOUT",    "45"))   # seconds for AI calls
MAX_RETRIES      = int(os.getenv("MAX_RETRIES",       "3"))

# Circuit breaker thresholds (failures → open circuit for N seconds)
CB_FAILURE_THRESHOLD = 5
CB_RECOVERY_SECONDS  = 60

# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURED LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger = structlog.get_logger("stockpulse")

# ═══════════════════════════════════════════════════════════════════════════════
# TWO-TIER CACHE
# ═══════════════════════════════════════════════════════════════════════════════

_hot_cache:  TTLCache = TTLCache(maxsize=CACHE_MAX_ITEMS, ttl=CACHE_HOT_TTL)
_warm_cache: TTLCache = TTLCache(maxsize=CACHE_MAX_ITEMS, ttl=CACHE_WARM_TTL)
_refresh_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

def cache_get(key: str) -> tuple:
    """Returns (data, is_fresh). Returns warm stale data with is_fresh=False."""
    hot = _hot_cache.get(key)
    if hot:
        return hot, True
    warm = _warm_cache.get(key)
    if warm:
        return warm, False
    return None, False

def cache_set(key: str, data: dict) -> None:
    _hot_cache[key]  = data
    _warm_cache[key] = data

# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """Per-source circuit breaker. Opens after N failures, retries after timeout."""
    def __init__(self, name: str):
        self.name       = name
        self.failures   = 0
        self.opened_at  = 0.0
        self.state      = "closed"   # closed | open | half-open

    def record_success(self) -> None:
        self.failures  = 0
        self.state     = "closed"

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= CB_FAILURE_THRESHOLD:
            self.state     = "open"
            self.opened_at = time.monotonic()
            logger.warning("circuit_breaker_opened", source=self.name, failures=self.failures)

    def is_available(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.monotonic() - self.opened_at > CB_RECOVERY_SECONDS:
                self.state = "half-open"
                return True
            return False
        return True  # half-open: allow one probe


_circuit_breakers: dict[str, CircuitBreaker] = {
    name: CircuitBreaker(name)
    for name in ["yahoo_quote", "yahoo_analysis", "finviz", "reddit", "investor_page", "seekingalpha", "wire", "groq"]
}

# ═══════════════════════════════════════════════════════════════════════════════
# RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════

limiter = Limiter(key_func=get_remote_address)

# ═══════════════════════════════════════════════════════════════════════════════
# STATIC DATA MAPS
# ═══════════════════════════════════════════════════════════════════════════════


IR_PAGES: dict[str, str] = {
    "AAPL":  "https://investor.apple.com/news-and-events/press-releases/default.aspx",
    "MSFT":  "https://www.microsoft.com/en-us/investor/earnings/",
    "NVDA":  "https://investor.nvidia.com/news/press-release-details/2024/",
    "TSLA":  "https://ir.tesla.com/",
    "META":  "https://investor.fb.com/investor-news/default.aspx",
    "GOOGL": "https://abc.xyz/investor/",
    "AMZN":  "https://ir.aboutamazon.com/news-releases/",
    "AMD":   "https://ir.amd.com/news-events/press-releases",
    "NFLX":  "https://ir.netflix.net/ir-overview/press-releases/default.aspx",
    "INTC":  "https://www.intc.com/news-releases/",
    "JPM":   "https://www.jpmorganchase.com/ir/news",
    "BAC":   "https://investor.bankofamerica.com/press-releases",
    "WMT":   "https://stock.walmart.com/news",
    "JNJ":   "https://investor.jnj.com/news-releases",
    "PG":    "https://pginvestor.com/news",
    "KO":    "https://investors.coca-colacompany.com/news-releases",
    "XOM":   "https://investor.exxonmobil.com/news-releases",
    "V":     "https://investor.visa.com/news/news-details",
    "MA":    "https://investor.mastercard.com/news-and-events/press-releases",
}

# ── Wire financial data source map ───────────────────────────────────────────
# Scrapes stockanalysis.com for PE ratio and valuation data.
# stockanalysis.com renders its financials in static HTML — no JavaScript
# rendering required, so Anakin's useBrowser=False works reliably.
# URL pattern: https://stockanalysis.com/stocks/{ticker}/financials/
# No slug map needed — ticker is sufficient for all stocks.
# This gives Wire a genuine valuation signal: PE, forward PE, PS ratio, PB ratio
# extracted from the income statement / ratios page.

# Screener universe — 50 major liquid US tickers
SCREENER_UNIVERSE = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AMD","INTC","NFLX",
    "JPM","BAC","V","MA","GS","MS","C","WFC","BRK-B","BLK",
    "JNJ","PFE","MRK","ABBV","UNH","LLY","CVS","MDT","ABT","AMGN",
    "XOM","CVX","COP","SLB","OXY","MPC","VLO","PSX","HAL","EOG",
    "WMT","TGT","COST","HD","LOW","MCD","SBUX","NKE","PG","KO",
]

# ═══════════════════════════════════════════════════════════════════════════════
# LIFESPAN
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("stockpulse_starting", env=ENVIRONMENT,
                groq_ok=bool(GROQ_API_KEY), anakin_ok=bool(ANAKIN_API_KEY))
    yield
    await _shared_http_client.aclose()
    logger.info("stockpulse_shutdown")

# ═══════════════════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="StockPulse Pro API",
    version="3.0.0",
    description="Production-grade stock intelligence backend with AI synthesis",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# In production, set CORS_ORIGINS=https://yourdomain.com in .env
_CORS_ORIGINS = (
    [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    if ENVIRONMENT == "production"
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Cache", "X-Data-Age", "X-Request-Id"],
)

# Compress responses over the wire — analysis/screener payloads are large
# JSON blobs and Render's free/starter network egress benefits noticeably
# from this on every request, not just static files.
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=500)

# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST ID MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════════

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    import uuid
    req_id = request.headers.get("X-Request-Id", str(uuid.uuid4())[:8])
    request.state.req_id = req_id
    start = time.monotonic()
    response: Response = await call_next(request)
    duration_ms = round((time.monotonic() - start) * 1000, 1)
    response.headers["X-Request-Id"] = req_id
    response.headers["X-Response-Time-Ms"] = str(duration_ms)
    logger.info("request", method=request.method, path=request.url.path,
                status=response.status_code, ms=duration_ms, req_id=req_id)
    return response

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_ticker(ticker: str) -> str:
    """Sanitise and validate a ticker symbol. Raises HTTPException on bad input."""
    t = ticker.strip().upper()
    # Allow 1–5 letters, optionally followed by a hyphen and 1 letter (BRK-B style)
    if not re.match(r'^[A-Z]{1,5}(-[A-Z]{1})?$', t):
        raise HTTPException(status_code=400,
            detail=f"Invalid ticker '{ticker}'. Use 1–5 letters, optionally with a suffix (e.g. AAPL, BRK-B).")
    return t

def _fmt_downloads(n: int) -> str:
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)

def strip_json_fences(raw: str) -> str:
    """Remove markdown code fences that LLMs sometimes add."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Drop first line (```json or ```) and last line (```) if present
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        raw = "\n".join(inner)
    return raw.strip()

# ═══════════════════════════════════════════════════════════════════════════════
# RETRY-WRAPPED HTTP HELPER
# ═══════════════════════════════════════════════════════════════════════════════

class ScrapingError(Exception):
    """Raised when a scrape fails after all retries."""

# ── Shared, connection-pooled HTTP client ──────────────────────────────────
# Creating a new httpx.AsyncClient per call means a fresh TCP+TLS handshake
# every time (to Anakin, Wire, Groq, Yahoo, etc.) — on Render this alone can
# add 100-300ms per outbound call. One shared client with keep-alive pooling
# reuses connections across requests and is closed cleanly on shutdown.
_shared_http_client: httpx.AsyncClient = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=30, keepalive_expiry=30),
)

async def _http_get_with_retry(
    url: str,
    headers: dict,
    cb_name: str,
    timeout: int = SCRAPE_TIMEOUT,
) -> dict:
    """GET request with retry + circuit breaker. Returns parsed JSON."""
    cb = _circuit_breakers.get(cb_name)
    if cb and not cb.is_available():
        raise ScrapingError(f"Circuit open for {cb_name}")

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
            reraise=True,
        ):
            with attempt:
                resp = await _shared_http_client.get(url, headers=headers, timeout=timeout)
                if resp.status_code == 429:
                    raise httpx.TimeoutException("rate limited", request=resp.request)
                resp.raise_for_status()
                if cb: cb.record_success()
                return resp.json()
    except RetryError as e:
        if cb: cb.record_failure()
        raise ScrapingError(f"{cb_name} GET failed after {MAX_RETRIES} attempts: {e}") from e
    except Exception as e:
        if cb: cb.record_failure()
        raise ScrapingError(f"{cb_name} GET error: {e}") from e

async def _http_post_with_retry(
    url: str,
    headers: dict,
    payload: dict,
    cb_name: str,
    timeout: int = SCRAPE_TIMEOUT,
) -> dict:
    """POST request with retry + circuit breaker. Returns parsed JSON."""
    cb = _circuit_breakers.get(cb_name)
    if cb and not cb.is_available():
        raise ScrapingError(f"Circuit open for {cb_name}")

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
            reraise=True,
        ):
            with attempt:
                resp = await _shared_http_client.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code == 429:
                    raise httpx.TimeoutException("rate limited", request=resp.request)
                if resp.status_code not in (200, 201, 202):
                    raise ScrapingError(
                        f"{cb_name} returned HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                if cb: cb.record_success()
                return resp.json()
    except RetryError as e:
        if cb: cb.record_failure()
        raise ScrapingError(f"{cb_name} POST failed after {MAX_RETRIES} attempts: {e}") from e
    except ScrapingError:
        if cb: cb.record_failure()
        raise
    except Exception as e:
        if cb: cb.record_failure()
        raise ScrapingError(f"{cb_name} POST error: {e}") from e

# ═══════════════════════════════════════════════════════════════════════════════
# ANAKIN SCRAPER
# ═══════════════════════════════════════════════════════════════════════════════

async def anakin_scrape(url: str, cb_name: str) -> str:
    """
    Scrape a URL via Anakin.io's URL Scraper API.
    Anakin.io is a job-based scraping API: submit a URL, get a jobId back,
    then poll GET /v1/url-scraper/{id} until status is "completed" or "failed".
    Requires only ANAKIN_API_KEY (no app/workspace ID involved).
    Returns the page's markdown/cleaned text, or raises ScrapingError.
    """
    if not ANAKIN_API_KEY:
        raise ScrapingError("Anakin not configured — set ANAKIN_API_KEY")

    headers = {
        "X-API-Key": ANAKIN_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "url": url,
        "country": "us",
        "useBrowser": False,
        "generateJson": False,
    }

    # ── Step 1: submit the scrape job ──
    submit_data = await _http_post_with_retry(
        "https://api.anakin.io/v1/url-scraper",
        headers=headers,
        payload=payload,
        cb_name=cb_name,
        timeout=SCRAPE_TIMEOUT,
    )

    job_id = submit_data.get("jobId")
    if not job_id:
        raise ScrapingError(f"{cb_name}: no jobId returned from Anakin submit: {submit_data}")

    # ── Step 2: poll for completion ──
    # Anakin jobs often finish in 1-3s; polling at a flat 2s wastes latency
    # on the common case. Start fast, back off, same total budget (~30s).
    poll_url = f"https://api.anakin.io/v1/url-scraper/{job_id}"
    poll_schedule = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]  # sums to ~32s
    max_polls = len(poll_schedule)

    cb = _circuit_breakers.get(cb_name)

    for poll_interval in poll_schedule:
        await asyncio.sleep(poll_interval)
        try:
            resp = await _shared_http_client.get(poll_url, headers=headers, timeout=SCRAPE_TIMEOUT)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            if cb: cb.record_failure()
            raise ScrapingError(f"{cb_name}: poll request failed: {e}") from e

        status = result.get("status")
        if status == "completed":
            if cb: cb.record_success()
            content = result.get("markdown") or result.get("cleanedHtml") or result.get("html") or ""
            if len(content) < 50:
                raise ScrapingError(f"Suspiciously short response ({len(content)} chars) from {url}")
            return content
        if status == "failed":
            if cb: cb.record_failure()
            raise ScrapingError(f"{cb_name}: job failed — {result.get('error', 'unknown error')}")
        # else: still pending/processing — keep polling

    if cb: cb.record_failure()
    raise ScrapingError(f"{cb_name}: job {job_id} did not complete within {sum(poll_schedule)}s")

# ═══════════════════════════════════════════════════════════════════════════════
# PER-SOURCE SCRAPERS  (each catches its own errors → empty string on failure)
# ═══════════════════════════════════════════════════════════════════════════════

async def _safe_scrape(coro, source_name: str, fallback: Any = "") -> Any:
    """Wraps any coroutine; returns fallback + logs on error instead of raising."""
    try:
        return await coro
    except ScrapingError as e:
        logger.warning("scrape_failed", source=source_name, reason=str(e))
        return fallback
    except Exception as e:
        logger.error("scrape_unexpected_error", source=source_name,
                     error=str(e), tb=traceback.format_exc())
        return fallback

async def scrape_yahoo_quote(ticker: str) -> str:
    return await _safe_scrape(
        anakin_scrape(f"https://finance.yahoo.com/quote/{ticker}/", "yahoo_quote"),
        "yahoo_quote",
    )

async def scrape_yahoo_analysis(ticker: str) -> str:
    return await _safe_scrape(
        anakin_scrape(f"https://finance.yahoo.com/quote/{ticker}/analysis/", "yahoo_analysis"),
        "yahoo_analysis",
    )

async def scrape_finviz(ticker: str) -> str:
    return await _safe_scrape(
        anakin_scrape(f"https://finviz.com/quote.ashx?t={ticker}", "finviz"),
        "finviz",
    )

async def scrape_reddit(ticker: str) -> str:
    return await _safe_scrape(
        anakin_scrape(
            f"https://www.reddit.com/search/?q={ticker}+stock&sort=hot&t=week",
            "reddit",
        ),
        "reddit",
    )

async def scrape_investor_page(ticker: str) -> str:
    url = IR_PAGES.get(ticker.upper(),
                       f"https://finance.yahoo.com/quote/{ticker}/press-releases/")
    return await _safe_scrape(
        anakin_scrape(url, "investor_page"),
        "investor_page",
    )

async def scrape_seekingalpha_news(ticker: str) -> str:
    """Extra source: Seeking Alpha news feed as additional signal."""
    url = f"https://seekingalpha.com/symbol/{ticker}/news"
    return await _safe_scrape(
        anakin_scrape(url, "seekingalpha"),   # own circuit breaker, independent of investor_page
        "seekingalpha",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# WIRE API  (Anakin's pre-built action layer)
# ═══════════════════════════════════════════════════════════════════════════════

async def get_wire_insight(ticker: str) -> dict:
    """
    Wire signal: scrapes stockanalysis.com for PE and valuation ratios via Anakin.

    stockanalysis.com renders its financials table in static HTML (no JavaScript
    rendering required), so Anakin's useBrowser=False scrapes it reliably.
    URL: https://stockanalysis.com/stocks/{ticker}/financials/?p=quarterly

    Extracts: PE ratio, forward PE, P/S ratio, P/B ratio from the ratios table.

    Scoring (0–5 wire_score):
      +1  PE data successfully extracted
      +1  PE < 30 (not in bubble territory)
      +1  Forward PE < trailing PE (earnings expected to grow)
      +1  P/S ratio < 5 (not wildly overpriced on revenue)
      +1  P/B ratio < 10 (reasonable book value multiple)
    """
    if not ANAKIN_API_KEY:
        return {"source": "wire", "summary": "Wire not configured", "wire_score": 0}

    cb = _circuit_breakers["wire"]
    if not cb.is_available():
        logger.warning("wire_circuit_open", ticker=ticker)
        return {"source": "wire", "summary": "Wire circuit open", "wire_score": 0}

    t = ticker.upper()
    url = f"https://stockanalysis.com/stocks/{t.lower()}/financials/?p=quarterly"

    try:
        content = await anakin_scrape(url, "wire")
        cb.record_success()

        # Extract PE and valuation ratios via regex on the scraped text
        pe_match      = re.search(r'P/E\s*Ratio[^\d]*(\d+\.?\d*)', content, re.IGNORECASE)
        fwd_pe_match  = re.search(r'Forward\s*P/E[^\d]*(\d+\.?\d*)', content, re.IGNORECASE)
        ps_match      = re.search(r'P/S\s*Ratio[^\d]*(\d+\.?\d*)', content, re.IGNORECASE)
        pb_match      = re.search(r'P/B\s*Ratio[^\d]*(\d+\.?\d*)', content, re.IGNORECASE)

        current_pe  = float(pe_match.group(1))     if pe_match     else None
        forward_pe  = float(fwd_pe_match.group(1)) if fwd_pe_match else None
        ps_ratio    = float(ps_match.group(1))     if ps_match     else None
        pb_ratio    = float(pb_match.group(1))     if pb_match     else None

        # Compute wire_score
        wire_score = 0
        signals = []

        if current_pe is not None:
            wire_score += 1
            if current_pe < 30:
                wire_score += 1
                signals.append(f"PE {current_pe:.1f} < 30")
            else:
                signals.append(f"PE {current_pe:.1f} (elevated)")

        if forward_pe is not None and current_pe is not None:
            if forward_pe < current_pe:
                wire_score += 1
                signals.append(f"fwd PE {forward_pe:.1f} < trailing (earnings growth expected)")

        if ps_ratio is not None and ps_ratio < 5:
            wire_score += 1
            signals.append(f"P/S {ps_ratio:.1f} < 5")

        if pb_ratio is not None and pb_ratio < 10:
            wire_score += 1
            signals.append(f"P/B {pb_ratio:.1f} < 10")

        wire_score = min(5, wire_score)

        summary = "; ".join(signals) if signals else "Valuation data unavailable"

        logger.info("wire_completed", ticker=t, source="stockanalysis",
                    pe=current_pe, fwd_pe=forward_pe, ps=ps_ratio, wire_score=wire_score)

        return {
            "source":       "wire",
            "data_source":  "stockanalysis.com",
            "url":          url,
            "pe_ratio":     current_pe,
            "forward_pe":   forward_pe,
            "ps_ratio":     ps_ratio,
            "pb_ratio":     pb_ratio,
            "wire_score":   wire_score,
            "summary":      summary,
            "raw_content":  content[:1500],
        }

    except ScrapingError as e:
        logger.warning("wire_scrape_failed", ticker=t, reason=str(e))
        return {"source": "wire", "summary": "Wire unavailable", "wire_score": 0}
    except Exception as e:
        cb.record_failure()
        logger.error("wire_unexpected_error", ticker=t, error=str(e))
        return {"source": "wire", "summary": "Wire error", "wire_score": 0}

# ═══════════════════════════════════════════════════════════════════════════════

async def run_pipeline(ticker: str) -> dict:
    """
    Full analysis pipeline:
      1. Check hot cache → return immediately if fresh
      2. Check warm cache → return stale + kick off background refresh
      3. Run 6 scrapers concurrently (each tolerates failure independently)
      4. AI synthesis via Groq
      5. Post-process: compute technical score, add metadata
      6. Persist to both cache tiers
    """
    ticker = validate_ticker(ticker)

    cached, is_fresh = cache_get(ticker)
    if cached and is_fresh:
        logger.info("cache_hit_hot", ticker=ticker)
        return {**cached, "cached": True, "cache_age": "fresh"}

    if cached and not is_fresh:
        logger.info("cache_hit_warm_stale", ticker=ticker)
        lock = _refresh_locks[ticker]
        if not lock.locked():
            asyncio.create_task(_background_refresh(ticker))
        return {**cached, "cached": True, "cache_age": "stale"}

    return await _fetch_and_cache(ticker)

async def _background_refresh(ticker: str) -> None:
    lock = _refresh_locks[ticker]
    if lock.locked():
        return
    async with lock:
        try:
            logger.info("background_refresh_start", ticker=ticker)
            await _fetch_and_cache(ticker)
            logger.info("background_refresh_done", ticker=ticker)
        except Exception as e:
            logger.warning("background_refresh_failed", ticker=ticker, error=str(e))

async def _fetch_and_cache(ticker: str) -> dict:
    t_start = time.monotonic()
    logger.info("pipeline_start", ticker=ticker)

    # ── Run all scrapers concurrently (failures are tolerated) ──
    (
        quote_raw, analysis_raw, finviz_raw,
        ir_raw, reddit_raw, sa_raw, wire_raw,
    ) = await asyncio.gather(
        scrape_yahoo_quote(ticker),
        scrape_yahoo_analysis(ticker),
        scrape_finviz(ticker),
        scrape_investor_page(ticker),
        scrape_reddit(ticker),
        scrape_seekingalpha_news(ticker),
        get_wire_insight(ticker),
    )

    # Count how many data sources actually came back with content
    sources_ok = sum([
        bool(quote_raw), bool(analysis_raw), bool(finviz_raw),
        bool(ir_raw), bool(reddit_raw), bool(sa_raw),
    ])
    logger.info("scrape_complete", ticker=ticker, sources_ok=sources_ok,
                wire_score=(wire_raw or {}).get("wire_score", 0))

    if sources_ok == 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "All data sources failed or Anakin is not configured. "
                "Set ANAKIN_API_KEY in your .env file."
            ),
        )

    # Include Wire's raw_content in the text pool so the AI synthesizer can
    # pick up historical PE context from Macrotrends alongside other sources.
    wire_text = ""
    if isinstance(wire_raw, dict) and wire_raw.get("raw_content"):
        wire_text = f"\n\n[Wire/StockAnalysis Valuation Data]\n{wire_raw['raw_content']}"

    scraped = {
        "yahoo_quote":    quote_raw    or "",
        "yahoo_analysis": analysis_raw or "",
        "finviz":         finviz_raw   or "",
        "investor_page":  (ir_raw or "") + "\n" + (sa_raw or "") + wire_text,
        "reddit":         reddit_raw   or "",
        "wire":           wire_raw if isinstance(wire_raw, dict) else {},
        "sources_ok":     sources_ok,
    }

    # Import here to avoid circular at module level
    from analyzer import synthesize_stock_data
    result = await synthesize_stock_data(ticker, scraped)

    # ── Attach Wire insight directly to result ──
    result["wire_insight"] = wire_raw if isinstance(wire_raw, dict) else {}

    # ── Enrich with computed fields ──
    result["_meta"] = {
        "fetched_at":  datetime.now(timezone.utc).isoformat(),
        "sources_ok":  sources_ok,
        "pipeline_ms": round((time.monotonic() - t_start) * 1000),
    }

    cache_set(ticker, result)
    logger.info("pipeline_done", ticker=ticker,
                ms=result["_meta"]["pipeline_ms"], sources=sources_ok)
    return {**result, "cached": False, "cache_age": "fresh"}

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class PortfolioRequest(BaseModel):
    amount:   float  = Field(..., gt=0, description="Investment amount in your currency")
    currency: str    = Field("INR", description="INR or USD")
    horizon:  str    = Field("medium", pattern="^(short|medium|long)$")
    risk:     str    = Field("low",    pattern="^(low|medium|high)$")
    sectors:  list[str] = Field(default_factory=list)
    spread:   int    = Field(5, ge=1, le=20)
    notes:    str    = Field("", max_length=500)

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be positive")
        return round(v, 2)

    @field_validator("currency")
    @classmethod
    def currency_upper(cls, v: str) -> str:
        return v.strip().upper()


class ChatMessage(BaseModel):
    role:    str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages:       list[ChatMessage] = Field(..., min_length=1, max_length=40)
    stock_context:  Optional[str]     = Field(None, max_length=200,
                        description="Optional: ticker being discussed, injected into system prompt")
    analysis_data:  Optional[dict]    = Field(None,
                        description="Optional: pass the full analysis JSON for grounded responses")


class ScreenerRequest(BaseModel):
    tickers:  list[str] = Field(default_factory=list,
                             description="Custom list. Empty = use built-in universe (50 stocks)")
    top_n:    int = Field(10, ge=1, le=50, description="Return top N by signal score")
    min_conf: float = Field(0.4, ge=0, le=1, description="Minimum AI confidence to include")


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT SYSTEM PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_chat_system_prompt(
    stock_context: Optional[str],
    analysis_data: Optional[dict],
) -> str:
    base = """You are PulseAdvisor, a senior AI investment advisor built into the StockPulse Pro terminal. You combine institutional-grade analysis with clear, plain-language advice.

CORE RULES:
1. Always give a direct BUY / HOLD / SELL verdict when asked. Never be vague about your recommendation.
2. Back every recommendation with 3–5 specific reasons: valuation, momentum, fundamentals, catalysts, risks.
3. Always state the key risk clearly — hiding risks is dishonest and harmful.
4. Use this format for stock questions:
   **Verdict:** BUY / HOLD / SELL  
   **Why:** (3–5 bullet points with concrete data)  
   **Key Risk:** (1–2 sentences on the biggest threat to the thesis)  
   **Entry Strategy:** (e.g. "Consider buying in 2 tranches: 50% now, 50% on any dip to $X")
   **Bottom Line:** (one bold sentence summarising the trade)
5. Use ✅ BUY, ⚠️ HOLD, ❌ SELL as signal markers.
6. Quote real metrics when you know them: P/E, RSI, analyst target, EPS growth, revenue growth.
7. For portfolio questions, give specific % allocations.
8. For "what should I do with X?", consider whether the user might already be holding — ask if unclear.
9. Keep responses 200–400 words. Do not pad.
10. Never say "I cannot provide financial advice." You ARE the financial advisor here. Give the best advice you can with a clear disclaimer at the bottom: "⚠️ This is AI-generated analysis, not certified financial advice. Always verify with multiple sources before trading."

IMPORTANT — LOSS PREVENTION RULES:
- If RSI > 75: always flag as potentially overbought — warn against chasing
- If short float > 25%: always flag high short-squeeze risk (both directions)
- If earnings are within 5 days: always flag binary event risk
- If a stock is down >15% in a month: require stronger justification before BUY
- Never recommend putting >40% of a portfolio in a single stock
- Always recommend stop-loss levels for aggressive positions
- For leveraged ETFs: always warn about decay and unsuitability for long holds"""

    if stock_context:
        base += f"\n\nCURRENT STOCK CONTEXT: The user is asking about {stock_context}."

    if analysis_data:
        ac    = analysis_data.get("analyst_consensus", {})
        fv    = analysis_data.get("finviz_signals", {})
        eps   = analysis_data.get("eps_estimate", {})
        base += f"""

LIVE DATA FOR {analysis_data.get('ticker', stock_context or 'this stock')}:
- Price: ${analysis_data.get('current_price', 'N/A')}  Change: {analysis_data.get('price_change_pct', 'N/A')}%
- Market Cap: {analysis_data.get('market_cap', 'N/A')}
- Analyst consensus: {ac.get('label', 'N/A')} ({ac.get('total_analysts', 0)} analysts)
- Price target: ${ac.get('mean_target', 'N/A')}  Upside: {ac.get('upside_pct', 'N/A')}%
- P/E: {fv.get('pe_ratio', 'N/A')}  RSI-14: {fv.get('rsi_14', 'N/A')}
- Short float: {fv.get('short_float_pct', 'N/A')}%  Inst. ownership: {fv.get('inst_own_pct', 'N/A')}%
- EPS this quarter: ${eps.get('current_quarter', 'N/A')}  Beat history: {eps.get('surprise_history', 'N/A')}
- 52W High: ${fv.get('52w_high', 'N/A')}  52W Low: ${fv.get('52w_low', 'N/A')}
- Risk flags: {', '.join(analysis_data.get('risk_flags', [])) or 'None'}
- AI verdict: {analysis_data.get('one_line_verdict', 'N/A')}

Use this live data to ground your response. Reference these numbers explicitly."""

    return base

# ═══════════════════════════════════════════════════════════════════════════════
# GROQ CALLER
# ═══════════════════════════════════════════════════════════════════════════════

async def call_groq(
    messages: list[dict],
    max_tokens: int = 800,
    temperature: float = 0.3,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
) -> str:
    """
    Call Groq API with retry + circuit breaker.
    Returns the text content of the first choice.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured in .env")

    cb = _circuit_breakers["groq"]
    if not cb.is_available():
        raise HTTPException(status_code=503, detail="Groq AI service temporarily unavailable")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }
    # gpt-oss / qwen3 models reason by default and can burn the whole token
    # budget on chain-of-thought before writing any real content. "low" keeps
    # reasoning brief. Other models (e.g. llama-4-scout) don't support this
    # param at all, so only send it for models that actually use it.
    if "gpt-oss" in model or "qwen3" in model:
        payload["reasoning_effort"] = "low"

    try:
        data = await _http_post_with_retry(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            payload=payload,
            cb_name="groq",
            timeout=GROQ_TIMEOUT,
        )
        cb.record_success()
        message = data["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if not content:
            # gpt-oss sometimes returns an empty content with the real text
            # left in the "reasoning" field if it ran out of budget mid-thought.
            content = (message.get("reasoning") or "").strip()
        if not content:
            raise HTTPException(
                status_code=502,
                detail="Groq returned an empty response (model likely ran out of "
                       "tokens during reasoning) — try again or raise max_tokens.",
            )
        return content
    except ScrapingError as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL SCORING  (used by screener + compare)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_signal_score(data: dict) -> dict:
    """
    Multi-factor signal score (0–100) combining:
      - Analyst consensus   (0–30)
      - Price target upside (0–20)
      - Technical (RSI, 52W position) (0–20)
      - AI confidence       (0–15)
      - Risk penalty        (0 to -20)
      - Momentum (price change) (0–10)
      - Short interest      (bonus/malus ±3)
      - Wire content signal (bonus 0–2) — via Anakin Wire action layer
    Returns score dict with breakdown for transparency.
    """
    ac    = data.get("analyst_consensus", {}) or {}
    fv    = data.get("finviz_signals",    {}) or {}
    conf  = float(data.get("confidence",   0.5) or 0.5)

    # ── Analyst consensus ──
    label = (ac.get("label") or "").lower()
    analyst_score = {
        "strong buy": 30, "buy": 22, "hold": 10,
        "sell": 3, "strong sell": 0,
    }.get(label, 10)

    # ── Upside ──
    upside = float(ac.get("upside_pct") or 0)
    upside_score = min(20, max(0, upside / 2))   # 40% upside → 20 pts

    # ── RSI ──
    rsi = float(fv.get("rsi_14") or 50)
    if 40 <= rsi <= 60:
        rsi_score = 12   # neutral / healthy
    elif 30 <= rsi < 40 or 60 < rsi <= 70:
        rsi_score = 8    # mildly extended either way
    elif rsi < 30:
        rsi_score = 16   # oversold = potential entry
    else:
        rsi_score = 2    # overbought (>70) = caution

    # ── 52W position ──
    hi   = float(fv.get("52w_high") or 0)
    lo   = float(fv.get("52w_low")  or 0)
    cur  = float(data.get("current_price") or 0)
    if hi and lo and cur and hi != lo:
        pct_from_low = (cur - lo) / (hi - lo)
        # Sweet spot: 20–70% from low = not at high, well off lows
        if 0.2 <= pct_from_low <= 0.7:
            range_score = 8
        elif pct_from_low < 0.2:
            range_score = 4   # very close to 52W low — extra caution
        else:
            range_score = 3   # near 52W high — less room to run
    else:
        range_score = 5

    # ── AI confidence ──
    conf_score = round(conf * 15)

    # ── Risk penalty ──
    risk_count = len(data.get("risk_flags") or [])
    risk_penalty = min(20, risk_count * 4)

    # ── Momentum ──
    chg = float(data.get("price_change_pct") or 0)
    momentum_score = min(10, max(0, chg))   # positive daily change up to +10%

    # ── Short interest ──
    short_float = float(fv.get("short_float_pct") or 0)
    short_bonus = -3 if short_float > 20 else (2 if short_float < 5 else 0)

    # ── Wire content-trend signal ──
    # wire_score (0–5) comes from get_wire_insight(): richness of the Wire
    # action response (word_count, tags, recency). Richer content → higher
    # score → small positive adj. Missing/failed Wire → 0, no penalty.
    wire = data.get("wire_insight") or {}
    wire_score = int(wire.get("wire_score") or 0) if isinstance(wire, dict) else 0
    wire_bonus = 2 if wire_score >= 4 else (1 if wire_score >= 2 else 0)

    total = (
        analyst_score + upside_score + rsi_score + range_score +
        conf_score + momentum_score + short_bonus + wire_bonus - risk_penalty
    )
    total = max(0, min(100, round(total)))

    # Derive label
    if total >= 75:  signal = "STRONG BUY"
    elif total >= 58: signal = "BUY"
    elif total >= 42: signal = "HOLD"
    elif total >= 28: signal = "SELL"
    else:             signal = "STRONG SELL"

    return {
        "score":          total,
        "signal":         signal,
        "breakdown": {
            "analyst":    analyst_score,
            "upside":     round(upside_score, 1),
            "rsi":        rsi_score,
            "range":      range_score,
            "confidence": conf_score,
            "momentum":   momentum_score,
            "short_adj":  short_bonus,
            "wire_adj":   wire_bonus,
            "risk_pen":   -risk_penalty,
        },
    }

# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Single stock analysis ─────────────────────────────────────────────────────
@app.get("/api/analyze/{ticker}", summary="Full AI analysis for one ticker")
@limiter.limit("20/minute")
async def analyze(ticker: str, request: Request):
    t = validate_ticker(ticker)
    result = await run_pipeline(t)
    # Attach computed signal score
    result["signal_score"] = compute_signal_score(result)
    response = JSONResponse(content=result)
    response.headers["X-Cache"] = "HIT" if result.get("cached") else "MISS"
    return response


# ── Compare two tickers ───────────────────────────────────────────────────────
@app.get("/api/compare", summary="Side-by-side comparison of two tickers")
@limiter.limit("10/minute")
async def compare(t1: str, t2: str, request: Request):
    t1 = validate_ticker(t1)
    t2 = validate_ticker(t2)
    if t1 == t2:
        raise HTTPException(status_code=400, detail="Cannot compare a ticker with itself")

    r1, r2 = await asyncio.gather(run_pipeline(t1), run_pipeline(t2))
    r1["signal_score"] = compute_signal_score(r1)
    r2["signal_score"] = compute_signal_score(r2)

    from analyzer import build_compare_response
    result = build_compare_response(r1, r2)
    return result


# ── Portfolio advisor ─────────────────────────────────────────────────────────
@app.post("/api/portfolio", summary="AI-generated portfolio recommendations")
@limiter.limit("10/minute")
async def portfolio_advisor(req: PortfolioRequest, request: Request):
    from analyzer import generate_portfolio_reco
    result = await generate_portfolio_reco(
        amount=req.amount,
        currency=req.currency,
        horizon=req.horizon,
        risk=req.risk,
        sectors=req.sectors,
        spread=req.spread,
        notes=req.notes,
    )
    return result


# ── Chatbot ───────────────────────────────────────────────────────────────────
@app.post("/api/chat", summary="AI stock advisor chatbot")
@limiter.limit("30/minute")
async def chat_advisor(req: ChatRequest, request: Request):
    system_prompt = build_chat_system_prompt(
        stock_context=req.stock_context,
        analysis_data=req.analysis_data,
    )
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m.role, "content": m.content} for m in req.messages
    ]
    reply = await call_groq(messages, max_tokens=800, temperature=0.5)
    return {"reply": reply, "model": "openai/gpt-oss-20b"}


# ── Screener ──────────────────────────────────────────────────────────────────
@app.post("/api/screener", summary="Scan a basket of tickers and rank by signal score")
@limiter.limit("5/minute")
async def screener(req: ScreenerRequest, request: Request):
    """
    Runs the analysis pipeline on a basket of tickers concurrently,
    then ranks them by computed signal score.
    Returns top_n results above min_conf.
    """
    tickers = [validate_ticker(t) for t in req.tickers] if req.tickers else SCREENER_UNIVERSE[:30]
    if len(tickers) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 tickers per screener call")

    logger.info("screener_start", tickers=len(tickers))

    # Run all analyses in parallel — tolerate individual failures
    raw_results = await asyncio.gather(
        *[run_pipeline(t) for t in tickers],
        return_exceptions=True,
    )

    ranked = []
    for ticker, result in zip(tickers, raw_results):
        if isinstance(result, Exception):
            logger.warning("screener_ticker_failed", ticker=ticker, error=str(result))
            continue
        if float(result.get("confidence", 0)) < req.min_conf:
            continue
        score_data = compute_signal_score(result)
        ranked.append({
            "ticker":        result.get("ticker", ticker),
            "company_name":  result.get("company_name", ticker),
            "sector":        result.get("sector"),
            "current_price": result.get("current_price"),
            "price_change_pct": result.get("price_change_pct"),
            "market_cap":    result.get("market_cap"),
            "signal_score":  score_data,
            "analyst_label": (result.get("analyst_consensus") or {}).get("label"),
            "upside_pct":    (result.get("analyst_consensus") or {}).get("upside_pct"),
            "rsi":           (result.get("finviz_signals") or {}).get("rsi_14"),
            "confidence":    result.get("confidence"),
            "risk_count":    len(result.get("risk_flags") or []),
            "one_line_verdict": result.get("one_line_verdict"),
            "earnings_date": result.get("earnings_date"),
        })

    ranked.sort(key=lambda x: x["signal_score"]["score"], reverse=True)
    top = ranked[:req.top_n]

    return {
        "screened": len(tickers),
        "returned": len(top),
        "min_conf": req.min_conf,
        "results":  top,
    }


# ── Watchlist batch analysis ──────────────────────────────────────────────────
@app.get("/api/watchlist/analyze", summary="Batch-analyze comma-separated tickers")
@limiter.limit("10/minute")
async def analyze_watchlist(tickers: str, request: Request):
    raw_list = [t.strip() for t in tickers.split(",") if t.strip()]
    if not raw_list:
        raise HTTPException(status_code=400, detail="Provide at least one ticker")
    if len(raw_list) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 tickers per batch call")

    validated = []
    for t in raw_list:
        try:
            validated.append(validate_ticker(t))
        except HTTPException:
            pass   # skip bad tickers silently in batch mode

    results = await asyncio.gather(
        *[run_pipeline(t) for t in validated],
        return_exceptions=True,
    )
    out = {}
    for t, r in zip(validated, results):
        if isinstance(r, Exception):
            out[t] = {"error": str(r)}
        else:
            r["signal_score"] = compute_signal_score(r)
            out[t] = r
    return out


# ── Technicals-only endpoint ──────────────────────────────────────────────────
@app.get("/api/technicals/{ticker}", summary="Pure technical analysis (uses cached data)")
@limiter.limit("30/minute")
async def technicals(ticker: str, request: Request):
    """
    Returns the technical scoring breakdown without re-running the full pipeline.
    If data is cached, this is near-instant. If not cached, runs a light scrape.
    """
    t = validate_ticker(ticker)
    cached, _ = cache_get(t)
    if cached:
        fv   = cached.get("finviz_signals", {}) or {}
        score = compute_signal_score(cached)
        hi   = float(fv.get("52w_high") or 0)
        lo   = float(fv.get("52w_low")  or 0)
        cur  = float(cached.get("current_price") or 0)
        pct  = round((cur - lo) / (hi - lo) * 100, 1) if (hi and lo and hi != lo) else None

        return {
            "ticker":             t,
            "current_price":      cur,
            "rsi_14":             fv.get("rsi_14"),
            "pe_ratio":           fv.get("pe_ratio"),
            "short_float_pct":    fv.get("short_float_pct"),
            "inst_own_pct":       fv.get("inst_own_pct"),
            "insider_own_pct":    fv.get("insider_own_pct"),
            "52w_high":           hi,
            "52w_low":            lo,
            "pct_from_52w_low":   pct,
            "avg_volume":         fv.get("avg_volume"),
            "signal_score":       score,
            "cached":             True,
        }
    # No cache — run full pipeline and return technicals subset
    result = await run_pipeline(t)
    return await technicals(ticker, request)   # recurse once, now cached


# ── News-only endpoint ────────────────────────────────────────────────────────
@app.get("/api/news/{ticker}", summary="Latest news headlines for a ticker")
@limiter.limit("30/minute")
async def news(ticker: str, request: Request):
    t = validate_ticker(ticker)
    cached, _ = cache_get(t)
    if cached:
        return {
            "ticker":     t,
            "headlines":  cached.get("news_headlines", []),
            "company_signals": cached.get("company_signals", []),
            "fetched_at": (cached.get("_meta") or {}).get("fetched_at"),
            "cached":     True,
        }
    result = await run_pipeline(t)
    return {
        "ticker":    t,
        "headlines": result.get("news_headlines", []),
        "company_signals": result.get("company_signals", []),
        "fetched_at": (result.get("_meta") or {}).get("fetched_at"),
        "cached":    False,
    }


# ── Cache management ──────────────────────────────────────────────────────────
@app.delete("/api/cache/{ticker}", summary="Invalidate cache for a ticker (force fresh fetch)")
@limiter.limit("5/minute")
async def invalidate_cache(ticker: str, request: Request):
    t = validate_ticker(ticker)
    removed = False
    if t in _hot_cache:
        del _hot_cache[t]
        removed = True
    if t in _warm_cache:
        del _warm_cache[t]
        removed = True
    return {"ticker": t, "invalidated": removed}


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/health", summary="Diagnostic health check")
async def health():
    """Returns full system status including circuit breaker states."""
    cb_states = {
        name: {
            "state":    cb.state,
            "failures": cb.failures,
            "available": cb.is_available(),
        }
        for name, cb in _circuit_breakers.items()
    }
    return {
        "status": "ok",
        "version": "3.0.0",
        "environment": ENVIRONMENT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "groq_configured":   bool(GROQ_API_KEY),
            "groq_key_preview":  (GROQ_API_KEY[:8] + "…") if GROQ_API_KEY else "NOT SET",
            "anakin_configured": bool(ANAKIN_API_KEY),
            "anakin_key_preview": (ANAKIN_API_KEY[:8] + "…") if ANAKIN_API_KEY else "NOT SET",
            "wire_configured":   bool(ANAKIN_API_KEY),  # Wire uses Anakin to scrape Macrotrends PE data
        },
        "cache": {
            "hot_entries":  len(_hot_cache),
            "warm_entries": len(_warm_cache),
            "hot_ttl_sec":  CACHE_HOT_TTL,
            "warm_ttl_sec": CACHE_WARM_TTL,
        },
        "circuit_breakers": cb_states,
        "rate_limits": {
            "analyze":   "20/minute",
            "compare":   "10/minute",
            "portfolio": "10/minute",
            "chat":      "30/minute",
            "screener":  "5/minute",
        },
    }


# ── Auth routes ──────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    email:     str = Field(..., min_length=3, max_length=200)
    password:  str = Field(..., min_length=1, max_length=200)
    full_name: str = Field("", max_length=100)

@app.post("/auth/register", summary="Register a new user")
@limiter.limit("10/minute")
async def register(req: AuthRequest, request: Request):
    return await register_user(req.email, req.password, req.full_name)

@app.post("/auth/login", summary="Login with email and password")
@limiter.limit("20/minute")
async def login(req: AuthRequest, request: Request):
    return await login_user(req.email, req.password)

@app.get("/auth/me", summary="Get current logged-in user")
async def me(user: dict = Depends(get_current_user)):
    return {"email": user.get("email"), "full_name": user.get("full_name")}

# ── Serve frontend ────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/login", include_in_schema=False)
async def login_page():
    return FileResponse("static/login.html")

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")

@app.get("/{path:path}", include_in_schema=False)
async def spa_fallback(path: str):
    """Catch-all for SPA routing."""
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    return FileResponse("static/index.html")