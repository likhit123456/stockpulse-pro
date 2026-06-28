# StockPulse Pro v3.0 — Investment Intelligence Terminal

A production-grade stock analysis platform for real investment decisions. Bloomberg-style UI with AI-powered analysis, chatbot advisor, smart portfolio builder, and screener.

---

## Quick Start

```bash
chmod +x run.sh && ./run.sh
```

Then open **http://localhost:8000**

---

## What's New in v3.0

### Backend (main.py)
| Feature | Detail |
|---|---|
| **Retry + exponential back-off** | Every external call retries up to 3× with 1→8s back-off |
| **Circuit breakers** | Per-source breaker opens after 5 failures, auto-recovers in 60s |
| **Two-tier cache** | Hot (5 min fresh) + Warm (30 min stale-while-revalidate) with background refresh |
| **Rate limiting** | Per-IP limits on every endpoint via `slowapi` |
| **Structured logging** | JSON logs via `structlog` — every request, pipeline, and error logged with context |
| **Graceful degradation** | Analysis succeeds even if 3 of 6 scraping sources fail |
| **Request IDs** | Every request gets a UUID for tracing across logs |
| **Signal scoring** | 8-factor model: analyst + upside + RSI + 52W range + confidence + momentum + short int + risk |
| **New endpoints** | `/api/screener`, `/api/technicals/{ticker}`, `/api/news/{ticker}`, `DELETE /api/cache/{ticker}` |
| **SPA fallback** | Catch-all route for frontend routing |

### Analyzer (analyzer.py)
| Feature | Detail |
|---|---|
| **Richer synthesis prompt** | Extracts 25+ fields including forward P/E, PEG, P/B, P/S, beta, ROE, profit margin, dividend yield |
| **JSON repair** | 4-strategy parser — recovers from markdown fences, trailing commas, partial truncation |
| **Full field validation** | Every field clamped, type-checked, and defaulted after parse |
| **RSI sentiment normalisation** | Reddit bullish/bearish/neutral pcts forced to sum to 100% |
| **Upside recomputation** | Re-derives upside_pct from price + target if LLM gets it wrong |
| **Portfolio hard rules** | Max 35% per position, min 5%, allocation sum enforced to exactly 100% |
| **Stop-loss on every position** | Portfolio prompt requires a specific stop-loss price for each pick |
| **Risk/reward ratio** | Computed automatically from target + stop-loss + current price |
| **8-factor compare** | Analyst label, upside, RSI, valuation, confidence, momentum, risk flags, short interest |
| **Position typing** | Core / Growth / Speculative labels on portfolio picks |

---

## Architecture

```
Request
  │
  ▼
FastAPI (main.py)
  │  Rate limiter (slowapi)
  │  Request ID middleware
  │
  ├── Two-tier TTL cache (cachetools)
  │     Hot  (5 min)  → serve immediately
  │     Warm (30 min) → serve stale + async background refresh
  │
  ├── run_pipeline(ticker)
  │     │
  │     ├── asyncio.gather — 7 concurrent scrapers
  │     │     yahoo_quote        → Anakin → Yahoo Finance quote page
  │     │     yahoo_analysis     → Anakin → Yahoo Finance analyst page
  │     │     finviz             → Anakin → Finviz quote page
  │     │     investor_page      → Anakin → company IR page
  │     │     reddit             → Anakin → Reddit search
  │     │     seekingalpha_news  → Anakin → Seeking Alpha
  │     │     wire               → Wire API (npm / PyPI download stats)
  │     │
  │     │     Each scraper: retry 3× with back-off + circuit breaker
  │     │     Failure of any source → returns "" (graceful degradation)
  │     │
  │     └── synthesize_stock_data() (analyzer.py)
  │           Groq LLaMA 3.1 8B → JSON extraction + synthesis
  │           JSON repair (4 strategies)
  │           Field validation + normalisation
  │           Signal score computation
  │
  ├── /api/portfolio → generate_portfolio_reco() (analyzer.py)
  │     Groq LLaMA → structured portfolio JSON
  │     Allocation normalisation, risk/reward computation
  │
  ├── /api/chat → call_groq() with grounded system prompt
  │     Optional: inject live analysis data for grounded responses
  │
  └── /api/screener → parallel run_pipeline on N tickers, rank by score
```

---

## API Reference

### Single Stock Analysis
```
GET /api/analyze/{TICKER}
Rate limit: 20/minute
Returns: Full analysis with 25+ fields + signal_score breakdown
```

### Compare Two Stocks
```
GET /api/compare?t1=AAPL&t2=NVDA
Rate limit: 10/minute
Returns: Both analyses + 8-factor winner verdict
```

### Portfolio Advisor
```
POST /api/portfolio
Rate limit: 10/minute
Body:
{
  "amount": 500000,
  "currency": "INR",        // INR or USD
  "horizon": "medium",      // short | medium | long
  "risk": "medium",         // low | medium | high
  "sectors": ["Technology", "Healthcare"],  // optional, empty = diversified
  "spread": 5,              // 1–20 stocks
  "notes": "avoid Chinese ADRs"  // optional freetext
}
```

### AI Advisor Chat
```
POST /api/chat
Rate limit: 30/minute
Body:
{
  "messages": [
    {"role": "user", "content": "Should I buy NVDA right now?"}
  ],
  "stock_context": "NVDA",        // optional — focuses system prompt
  "analysis_data": { ... }        // optional — paste full /api/analyze response
                                  // for grounded, data-backed answers
}
```

### Screener
```
POST /api/screener
Rate limit: 5/minute
Body:
{
  "tickers": [],         // empty = scan built-in 30-stock universe
  "top_n": 10,           // return top N results
  "min_conf": 0.4        // minimum AI data confidence
}
Returns: Ranked list by signal score
```

### Technical Analysis
```
GET /api/technicals/{TICKER}
Returns: RSI, P/E, 52W range, short float, signal score breakdown
Instant if ticker is cached, else runs pipeline first
```

### News Headlines
```
GET /api/news/{TICKER}
Returns: Latest headlines + company signals from cached or fresh data
```

### Batch Watchlist
```
GET /api/watchlist/analyze?tickers=AAPL,NVDA,TSLA
Max 20 tickers. Returns map of ticker → full analysis
```

### Cache Invalidation
```
DELETE /api/cache/{TICKER}
Forces a fresh fetch on next request for that ticker
```

### Health Check
```
GET /api/health
Returns: Config status, cache sizes, circuit breaker states, rate limits
```

---

## Setup

```bash
# 1. Clone / unzip
cd stockpulse-pro

# 2. Configure
cp .env.example .env
# Edit .env — minimum required: GROQ_API_KEY + ANAKIN_API_KEY + ANAKIN_APP_ID

# 3. Run (creates venv + installs deps automatically)
chmod +x run.sh && ./run.sh
```

Manual setup if preferred:
```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

---

## API Keys

| Key | Required | Where to get |
|---|---|---|
| `GROQ_API_KEY` | **Yes** | https://console.groq.com → API Keys → Create (free) |
| `ANAKIN_API_KEY` | **Yes** | https://anakin.ai → Settings → API |
| `ANAKIN_APP_ID` | **Yes** | Same Anakin dashboard — Universal Scraper app ID |
| `WIRE_API_KEY` | No | https://wire.dev (adds developer ecosystem signals) |

---

## Configuration Tuning

All tuneable via `.env`:

| Variable | Default | Effect |
|---|---|---|
| `CACHE_HOT_TTL` | 300 | Seconds before data is considered stale |
| `CACHE_WARM_TTL` | 1800 | Seconds before stale data is fully evicted |
| `CACHE_MAX_ITEMS` | 500 | Max tickers in memory |
| `SCRAPE_TIMEOUT` | 30 | Per-source HTTP timeout |
| `GROQ_TIMEOUT` | 45 | AI synthesis timeout |
| `MAX_RETRIES` | 3 | Retry attempts per source before circuit opens |
| `ENVIRONMENT` | development | Set to `production` for stricter logging |
| `LOG_LEVEL` | INFO | DEBUG for verbose, WARNING for quiet |

---

## Circuit Breaker Behaviour

Each data source has its own circuit breaker:

- **Closed** (normal): requests flow through, failures counted
- **Open** (tripped): after 5 failures, source is skipped for 60 seconds
- **Half-open** (probing): one request is allowed through to test recovery

You can see all circuit breaker states at `/api/health`.
To manually reset: restart the server, or call `DELETE /api/cache/{ticker}` to force a fresh attempt.

---

## Disclaimer
**This tool provides AI-generated analysis for informational purposes only. It is NOT certified financial advice. Stock markets carry risk and you can lose money. Always:**
- Verify data from multiple independent sources before trading
- Consult a licensed financial advisor for significant investments
- Never invest more than you can afford to lose
- Use stop-losses on every position

The signal scores, buy/sell verdicts, and portfolio recommendations are generated by an AI model and may be incorrect, outdated, or based on incomplete data.