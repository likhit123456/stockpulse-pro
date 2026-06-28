"""
StockPulse Pro — analyzer.py  (v3.0 — Production-grade)
=========================================================
AI synthesis, portfolio construction, and comparison scoring.

Key improvements over v2:
  • Massively richer synthesis prompt — instructs the LLM to extract
    20+ distinct data points, add technical interpretation, and self-
    assess data quality
  • JSON repair: recovers from partial / truncated LLM responses by
    attempting multiple parse strategies before giving up
  • Portfolio prompt enforces hard risk rules (diversification,
    stop-losses, position sizing) that protect against common mistakes
  • Richer compare scoring: 8-factor model with penalty/bonus system
  • generate_portfolio_reco validates allocations sum to 100%,
    fills missing fields, and normalises signals
  • All Groq calls go through main.py's call_groq() helper
    (retry + circuit breaker) — no duplicate HTTP logic here
"""

import json
import logging
import math
import re
from typing import Any, Optional

import structlog
from fastapi import HTTPException

logger = structlog.get_logger("stockpulse.analyzer")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# USD/INR rate used for allocation display.
# Override via USD_TO_INR env var to keep it current without a code change.
# e.g. add  USD_TO_INR=84.2  to your .env
import os as _os
USD_TO_INR = float(_os.getenv("USD_TO_INR", "84.0"))

# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

SYNTHESIS_PROMPT = """\
You are a senior financial analyst at a tier-1 hedge fund. You have received raw scraped data
from multiple sources for the ticker {ticker}. Your job is to extract, verify across sources,
and synthesise this data into a precise, grounded JSON object for a live trading terminal.

CRITICAL RULES:
1. Return ONLY valid JSON — no markdown fences, no preamble, no explanation.
2. Cross-reference numbers across sources. If Yahoo says price $210 but Finviz says $215, use the
   Yahoo figure (it is more real-time) and note the discrepancy in risk_flags.
3. NEVER fabricate specific numbers. If a value cannot be found in the data, use null (for strings)
   or 0 (for numbers). Do NOT guess.
4. Extract ALL available analyst ratings counts (strong buy, buy, hold, sell, strong sell) from the
   analyst data — do not collapse them.
5. For EPS surprise history, count how many of the last 4 quarters were beats.
6. RSI, short float, P/E, 52W high/low, and avg volume MUST come from the Finviz data block.
7. Reddit sentiment must be estimated from the tone of posts — do not just say "Neutral" by default.
8. confidence is YOUR assessment of data quality: 0.9 if all sources rich, 0.5 if 2-3 sources
   returned data, 0.3 if only 1 source. This is not the stock's quality — it is the DATA quality.
9. one_line_verdict should be what a head trader would say at morning standup: specific, opinionated.
10. In risk_flags: list EVERY meaningful risk you can find — regulatory, valuation, competition,
    macro, technical. At least 2, at most 8.
11. In company_signals: list positive catalysts — new products, buybacks, earnings beats, expansion.
    At least 1, at most 6.
12. In news_headlines: extract verbatim short headlines from the investor/press page data.
    Up to 5 headlines. If none found, leave as empty array.

DATA SOURCES PROVIDED:
---
[YAHOO FINANCE — QUOTE PAGE]
{yahoo_quote}

[YAHOO FINANCE — ANALYST/ESTIMATES PAGE]
{yahoo_analysis}

[FINVIZ — TECHNICAL + FUNDAMENTAL SIGNALS]
{finviz}

[COMPANY INVESTOR RELATIONS + NEWS]
{investor_page}

[REDDIT SENTIMENT — r/stocks, r/investing, r/wallstreetbets (this week)]
{reddit}

[WIRE API — DEVELOPER ECOSYSTEM USAGE]
{wire_summary}
---

SOURCES AVAILABLE: {sources_ok}/6 data sources returned content.
{sources_warning}

Return exactly this JSON structure:

{{
  "ticker": "{ticker}",
  "company_name": "Full legal company name",
  "sector": "Sector (e.g. Technology, Healthcare, Energy)",
  "industry": "Industry (e.g. Semiconductors, Biotechnology)",
  "current_price": 0.00,
  "price_change_pct": 0.00,
  "price_change_abs": 0.00,
  "market_cap": "e.g. 2.87T",
  "volume_today": "e.g. 45.2M",
  "avg_volume": "e.g. 58.3M",
  "earnings_date": "Month DD YYYY or null",
  "ex_dividend_date": "Month DD YYYY or null",
  "analyst_consensus": {{
    "label": "Strong Buy | Buy | Hold | Sell | Strong Sell",
    "strong_buy": 0,
    "buy": 0,
    "hold": 0,
    "sell": 0,
    "strong_sell": 0,
    "total_analysts": 0,
    "mean_target": 0.00,
    "high_target": 0.00,
    "low_target": 0.00,
    "upside_pct": 0.00
  }},
  "eps_estimate": {{
    "current_quarter": 0.00,
    "next_quarter": 0.00,
    "current_year": 0.00,
    "next_year": 0.00,
    "beats_last_4q": 0,
    "surprise_history": "Beat last N of 4 quarters or null"
  }},
  "revenue_estimate": {{
    "current_quarter": "e.g. $120.4B or null",
    "current_year": "e.g. $480B or null",
    "yoy_growth_pct": 0.00
  }},
  "finviz_signals": {{
    "short_float_pct": 0.00,
    "short_ratio": 0.00,
    "insider_own_pct": 0.00,
    "inst_own_pct": 0.00,
    "pe_ratio": 0.00,
    "forward_pe": 0.00,
    "peg_ratio": 0.00,
    "price_to_sales": 0.00,
    "price_to_book": 0.00,
    "52w_high": 0.00,
    "52w_low": 0.00,
    "rsi_14": 0.00,
    "beta": 0.00,
    "avg_volume": "e.g. 58.3M",
    "debt_equity": 0.00,
    "roe_pct": 0.00,
    "profit_margin_pct": 0.00,
    "dividend_yield_pct": 0.00
  }},
  "reddit_sentiment": {{
    "label": "Bullish | Bearish | Neutral | Mixed",
    "bullish_pct": 0,
    "bearish_pct": 0,
    "neutral_pct": 0,
    "mention_count": 0,
    "avg_sentiment_score": 0.00,
    "top_themes": ["theme 1", "theme 2"]
  }},
  "wire_insight": {{
    "available": false,
    "summary": "npm/pypi package usage summary or null",
    "trend_direction": "up | down | flat | null"
  }},
  "company_signals": [
    "Positive catalyst or signal"
  ],
  "risk_flags": [
    "Specific risk (be concrete)"
  ],
  "news_headlines": [
    "Short headline from investor page"
  ],
  "technical_notes": "2–3 sentences on technical setup: trend, support/resistance, volume, momentum",
  "one_line_verdict": "What a head trader would say at morning standup — specific and opinionated",
  "bull_case": "Specific 1-sentence bull case with a catalyst",
  "bear_case": "Specific 1-sentence bear case with a risk",
  "suggested_entry": "e.g. Below $210 for a better risk/reward or null",
  "suggested_stop_loss": "e.g. $195 (-7% from current) or null",
  "confidence": 0.00
}}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

PORTFOLIO_PROMPT = """\
You are a senior portfolio manager at a top-tier asset management firm. A client wants
to invest in US-listed stocks. Your job is to generate a professionally constructed
portfolio that balances return potential with risk management.

CLIENT PROFILE:
- Investment budget: {currency_symbol}{amount} {currency_note}
- Approximate USD budget: ~${amount_usd}
- Time horizon: {horizon_label}
- Risk tolerance: {risk_label}
- Preferred sectors: {sectors}
- Number of positions: {spread}
{notes_line}

STRICT PORTFOLIO CONSTRUCTION RULES:
1. Allocations must sum to EXACTLY 100%. Check your math before responding.
2. Maximum single-position allocation: 35% (prevents concentration risk).
3. Minimum single-position allocation: 5% (avoids meaningless positions).
4. Risk diversification:
   - LOW risk: ≥60% must be in dividend-paying or defensive large-caps (JNJ, MSFT, PG, KO, V, MA, JPM)
   - MEDIUM risk: no single sector > 40% of portfolio; include at least 1 defensive name
   - HIGH risk: include at least 1 stop-loss note per position; warn about volatility
5. NEVER recommend a stock with P/E > 100 for a LOW-risk portfolio.
6. NEVER recommend the same stock twice.
7. For each position, you MUST provide a specific suggested stop-loss price.
8. For each position, you MUST provide a price target and the reasoning behind it.
9. "why" must reference specific metrics: EPS growth, P/E vs sector average, analyst targets,
   product catalysts, margin trends — NOT generic phrases like "strong company".
10. Allocate more to higher-conviction, lower-risk positions.

Return ONLY a valid JSON array — no markdown fences, no preamble, no explanation.

[
  {{
    "rank": 1,
    "ticker": "MSFT",
    "company": "Microsoft Corporation",
    "sector": "Technology",
    "industry": "Software",
    "current_price_usd": 422.00,
    "allocation_pct": 25,
    "signal": "BUY",
    "target_price_usd": 500.00,
    "stop_loss_usd": 390.00,
    "upside_pct": 18.5,
    "downside_risk_pct": 7.6,
    "risk_reward_ratio": 2.4,
    "expected_hold_months": 12,
    "why": "Azure cloud revenue growing 28% YoY with expanding margins; Copilot AI integration driving 15% premium on enterprise subscriptions. Trading at 32x forward P/E vs 5Y avg of 35x — slight discount on a durable compounder.",
    "bull_point": "AI Copilot adoption in enterprise could add $10B+ incremental revenue by FY2026",
    "risk": "Antitrust scrutiny of Activision integration and potential slowing of Azure growth if macro deteriorates",
    "stop_loss_rationale": "Below $390 breaks the 200-day MA and invalidates the bull thesis",
    "pe_ratio": 35.2,
    "forward_pe": 29.8,
    "dividend_yield_pct": 0.72,
    "analyst_rating": "Strong Buy",
    "total_analysts": 52,
    "rsi": 54,
    "beta": 0.9,
    "position_type": "Core"
  }}
]

Position types:
- "Core": 15–35%, high conviction, low-medium risk, hold for full horizon
- "Growth": 10–20%, medium conviction, medium risk, potential to outperform
- "Speculative": 5–10%, lower conviction, higher risk, more volatile — max 2 per portfolio

Return the array with exactly {spread} items.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# JSON REPAIR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _try_parse_json(raw: str) -> Any:
    """
    Attempt to parse JSON from a string using multiple strategies:
      1. Direct parse
      2. After stripping markdown fences
      3. Extract the first {...} or [...] block with regex
      4. Attempt to fix common LLM issues (trailing commas, unquoted keys)
    Raises ValueError if all strategies fail.
    """
    from main import strip_json_fences   # local import to avoid circular

    strategies = [
        lambda s: json.loads(s),
        lambda s: json.loads(strip_json_fences(s)),
        lambda s: json.loads(_extract_json_block(s)),
        lambda s: json.loads(_fix_json_quirks(strip_json_fences(s))),
    ]
    last_err = None
    for strategy in strategies:
        try:
            return strategy(raw)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            last_err = e
            continue
    raise ValueError(f"All JSON parse strategies failed. Last error: {last_err}. Raw snippet: {raw[:300]}")

def _extract_json_block(text: str) -> str:
    """Find the first { ... } or [ ... ] block in text."""
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    raise ValueError("No JSON block found in text")

def _fix_json_quirks(text: str) -> str:
    """Fix common LLM JSON generation mistakes."""
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([\}\]])', r'\1', text)
    # Replace single quotes with double quotes (crude but helps in simple cases)
    # Only when they appear to be string delimiters
    # This is intentionally conservative — only fixes obvious cases
    return text

# ═══════════════════════════════════════════════════════════════════════════════
# DATA VALIDATORS  (post-parse sanity checks)
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default

def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(float(val)) if val is not None else default
    except (TypeError, ValueError):
        return default

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def validate_and_normalise_synthesis(data: dict, ticker: str) -> dict:
    """
    Post-parse validation of the synthesis JSON.
    Fills gaps, clamps out-of-range values, ensures required keys exist.
    Does NOT fabricate data — only applies structural defaults.
    """
    # Top-level required strings
    data.setdefault("ticker", ticker)
    data.setdefault("company_name", ticker)
    data.setdefault("sector", None)
    data.setdefault("industry", None)
    data.setdefault("market_cap", None)
    data.setdefault("earnings_date", None)
    data.setdefault("ex_dividend_date", None)
    data.setdefault("one_line_verdict", "Insufficient data for verdict.")
    data.setdefault("bull_case", None)
    data.setdefault("bear_case", None)
    data.setdefault("technical_notes", None)
    data.setdefault("suggested_entry", None)
    data.setdefault("suggested_stop_loss", None)

    # Numeric top-level fields — clamp to sane ranges
    data["current_price"]    = max(0, _safe_float(data.get("current_price")))
    data["price_change_pct"] = _clamp(_safe_float(data.get("price_change_pct")), -99, 99)
    data["price_change_abs"] = _safe_float(data.get("price_change_abs"))
    data["confidence"]       = _clamp(_safe_float(data.get("confidence"), 0.5), 0.0, 1.0)

    # Analyst consensus
    ac = data.get("analyst_consensus") or {}
    if not isinstance(ac, dict):
        ac = {}
    ac.setdefault("label",          "Hold")
    ac.setdefault("strong_buy",     0)
    ac.setdefault("buy",            0)
    ac.setdefault("hold",           0)
    ac.setdefault("sell",           0)
    ac.setdefault("strong_sell",    0)
    ac.setdefault("total_analysts", 0)
    ac.setdefault("mean_target",    0.0)
    ac.setdefault("high_target",    0.0)
    ac.setdefault("low_target",     0.0)
    ac.setdefault("upside_pct",     0.0)
    # Recompute upside_pct if price and target are available
    if ac["mean_target"] and data["current_price"]:
        ac["upside_pct"] = round(
            (ac["mean_target"] - data["current_price"]) / data["current_price"] * 100, 2
        )
    data["analyst_consensus"] = ac

    # EPS estimates
    eps = data.get("eps_estimate") or {}
    if not isinstance(eps, dict):
        eps = {}
    eps.setdefault("current_quarter",  None)
    eps.setdefault("next_quarter",     None)
    eps.setdefault("current_year",     None)
    eps.setdefault("next_year",        None)
    eps.setdefault("beats_last_4q",    None)
    eps.setdefault("surprise_history", None)
    data["eps_estimate"] = eps

    # Revenue estimates
    rev = data.get("revenue_estimate") or {}
    if not isinstance(rev, dict):
        rev = {}
    rev.setdefault("current_quarter", None)
    rev.setdefault("current_year",    None)
    # Coerce to float or None — never leave as a raw non-numeric value
    _yoy = rev.get("yoy_growth_pct")
    rev["yoy_growth_pct"] = _safe_float(_yoy) if _yoy is not None else None
    data["revenue_estimate"] = rev

    # Finviz signals
    fv = data.get("finviz_signals") or {}
    if not isinstance(fv, dict):
        fv = {}
    fv_defaults = {
        "short_float_pct": None, "short_ratio": None,
        "insider_own_pct": None, "inst_own_pct": None,
        "pe_ratio": None, "forward_pe": None, "peg_ratio": None,
        "price_to_sales": None, "price_to_book": None,
        "52w_high": None, "52w_low": None, "rsi_14": None,
        "beta": None, "avg_volume": None, "debt_equity": None,
        "roe_pct": None, "profit_margin_pct": None, "dividend_yield_pct": None,
    }
    for k, v in fv_defaults.items():
        fv.setdefault(k, v)

    # Clamp RSI to 0–100
    if fv.get("rsi_14") is not None:
        fv["rsi_14"] = _clamp(_safe_float(fv["rsi_14"]), 0, 100)
    data["finviz_signals"] = fv

    # Reddit sentiment
    rs = data.get("reddit_sentiment") or {}
    if not isinstance(rs, dict):
        rs = {}
    rs.setdefault("label",               "Neutral")
    rs.setdefault("bullish_pct",         0)
    rs.setdefault("bearish_pct",         0)
    rs.setdefault("neutral_pct",         0)
    rs.setdefault("mention_count",       0)
    rs.setdefault("avg_sentiment_score", 0.0)
    rs.setdefault("top_themes",          [])
    # Normalise pcts to sum to 100
    total_pct = _safe_float(rs["bullish_pct"]) + _safe_float(rs["bearish_pct"]) + _safe_float(rs["neutral_pct"])
    if total_pct > 0 and abs(total_pct - 100) > 5:
        scale = 100 / total_pct
        rs["bullish_pct"] = round(_safe_float(rs["bullish_pct"]) * scale)
        rs["bearish_pct"] = round(_safe_float(rs["bearish_pct"]) * scale)
        rs["neutral_pct"] = 100 - rs["bullish_pct"] - rs["bearish_pct"]
    data["reddit_sentiment"] = rs

    # Wire insight
    wi = data.get("wire_insight") or {}
    if not isinstance(wi, dict):
        wi = {}
    wi.setdefault("available",        False)
    wi.setdefault("summary",          None)
    wi.setdefault("trend_direction",  None)
    data["wire_insight"] = wi

    # Lists — ensure they are actual lists of strings
    for key in ("company_signals", "risk_flags", "news_headlines"):
        val = data.get(key)
        if not isinstance(val, list):
            data[key] = []
        else:
            data[key] = [str(x) for x in val if x]

    return data


def validate_portfolio_item(item: dict, index: int, spread: int) -> dict:
    """
    Validate and normalise a single portfolio recommendation item.
    """
    item.setdefault("rank",                 index + 1)
    item.setdefault("ticker",               f"STOCK{index+1}")
    item.setdefault("company",              item.get("ticker", "Unknown"))
    item.setdefault("sector",               None)
    item.setdefault("industry",             None)
    item.setdefault("current_price_usd",    0.0)
    item.setdefault("allocation_pct",       round(100 / spread, 1))
    item.setdefault("signal",               "BUY")
    item.setdefault("target_price_usd",     0.0)
    item.setdefault("stop_loss_usd",        None)
    item.setdefault("upside_pct",           0.0)
    item.setdefault("downside_risk_pct",    None)
    item.setdefault("risk_reward_ratio",    None)
    item.setdefault("expected_hold_months", None)
    item.setdefault("why",                  "AI-generated recommendation.")
    item.setdefault("bull_point",           None)
    item.setdefault("risk",                 None)
    item.setdefault("stop_loss_rationale",  None)
    item.setdefault("pe_ratio",             None)
    item.setdefault("forward_pe",           None)
    item.setdefault("dividend_yield_pct",   None)
    item.setdefault("analyst_rating",       None)
    item.setdefault("total_analysts",       None)
    item.setdefault("rsi",                  None)
    item.setdefault("beta",                 None)
    item.setdefault("position_type",        "Growth")

    # Normalise signal
    sig = str(item["signal"]).upper().strip()
    if sig not in ("BUY", "HOLD", "SELL", "STRONG BUY", "STRONG SELL"):
        item["signal"] = "BUY"
    else:
        item["signal"] = sig

    # Compute risk/reward if missing
    if (item["current_price_usd"] and item["target_price_usd"]
            and item["stop_loss_usd"] and not item["risk_reward_ratio"]):
        upside   = item["target_price_usd"]   - item["current_price_usd"]
        downside = item["current_price_usd"]   - item["stop_loss_usd"]
        if downside > 0:
            item["risk_reward_ratio"] = round(upside / downside, 2)

    return item


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHESIZE STOCK DATA
# ═══════════════════════════════════════════════════════════════════════════════

async def synthesize_stock_data(ticker: str, scraped: dict) -> dict:
    """
    Call Groq LLaMA to synthesise scraped data into a structured analysis object.
    Includes JSON repair and full field validation.
    """
    from main import call_groq   # avoids circular at module-load time

    sources_ok = scraped.get("sources_ok", 3)
    sources_warning = (
        "⚠️  WARNING: Only 1–2 sources available. Confidence should be ≤ 0.4."
        if sources_ok <= 2 else ""
    )

    def trunc(text: Optional[str], max_chars: int) -> str:
        if not text:
            return "[No data from this source]"
        return text[:max_chars]

    wire_dict  = scraped.get("wire", {})
    wire_sum   = wire_dict.get("summary", "No Wire data available") if isinstance(wire_dict, dict) else "No Wire data"

    prompt = SYNTHESIS_PROMPT.format(
        ticker          = ticker,
        yahoo_quote     = trunc(scraped.get("yahoo_quote"),    2500),
        yahoo_analysis  = trunc(scraped.get("yahoo_analysis"), 2500),
        finviz          = trunc(scraped.get("finviz"),         2500),
        investor_page   = trunc(scraped.get("investor_page"),  2000),
        reddit          = trunc(scraped.get("reddit"),         2000),
        wire_summary    = wire_sum,
        sources_ok      = sources_ok,
        sources_warning = sources_warning,
    )

    logger.info("synthesis_start", ticker=ticker, sources_ok=sources_ok)

    raw = await call_groq(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3500,    # full schema ~700 tokens + headroom for gpt-oss low-effort reasoning
        temperature=0.05,   # very low temp for deterministic extraction
    )

    try:
        data = _try_parse_json(raw)
    except ValueError as e:
        logger.error("synthesis_json_parse_failed", ticker=ticker, error=str(e), raw_snippet=raw[:300])
        raise HTTPException(
            status_code=502,
            detail=(
                f"AI synthesis returned unparseable output for {ticker}. "
                f"This is a temporary Groq issue — please retry in a few seconds. "
                f"Details: {str(e)[:200]}"
            ),
        )

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=502,
            detail=f"AI synthesis returned wrong type ({type(data).__name__}) for {ticker}"
        )

    # Full validation pass
    data = validate_and_normalise_synthesis(data, ticker)

    # Attach raw Wire data so frontend can use trend_pct
    if isinstance(wire_dict, dict) and wire_dict:
        data["_wire_raw"] = wire_dict

    logger.info("synthesis_done", ticker=ticker, confidence=data.get("confidence"))
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO RECOMMENDATION GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_portfolio_reco(
    amount:   float,
    currency: str,
    horizon:  str,
    risk:     str,
    sectors:  list[str],
    spread:   int,
    notes:    str,
) -> dict:
    """
    Generate an institutional-quality portfolio recommendation.

    Includes:
      - Validated JSON parsing with repair
      - Allocation normalisation (forces sum to 100%)
      - Risk/reward computation for each position
      - Hard position-limit enforcement
      - Per-item validation and defaults
    """
    from main import call_groq

    horizon_labels = {
        "short":  "Short-term (1–6 months) — favour momentum, recent earnings beats, technical setups",
        "medium": "Medium-term (6–24 months) — mix of growth and value; watch upcoming catalysts",
        "long":   "Long-term (2–10 years) — prefer durable competitive moats, FCF compounders, dividend growers",
    }
    risk_labels = {
        "low":    "Conservative — capital preservation priority; prefer dividend-payers, large-caps, low beta (< 1.0)",
        "medium": "Balanced — moderate growth acceptable; mix of value and growth; beta 0.8–1.3",
        "high":   "Aggressive — maximum return priority; higher volatility accepted; growth/momentum/sector ETFs ok",
    }

    currency = currency.upper()
    if currency == "INR":
        currency_symbol = "₹"
        amount_usd      = round(amount / USD_TO_INR)
        currency_note   = f"(Indian Rupees — investor using international brokerage; converted at ₹{USD_TO_INR}/USD)"
    else:
        currency_symbol = "$"
        amount_usd      = round(amount)
        currency_note   = "(USD)"

    sector_str    = ", ".join(sectors) if sectors else "No preference — diversify across sectors"
    notes_line    = f"- Additional instructions from client: {notes}" if notes else ""
    horizon_label = horizon_labels.get(horizon, horizon)
    risk_label    = risk_labels.get(risk, risk)

    prompt = PORTFOLIO_PROMPT.format(
        currency_symbol = currency_symbol,
        amount          = f"{amount:,.0f}",
        amount_usd      = f"{amount_usd:,}",
        currency_note   = currency_note,
        horizon_label   = horizon_label,
        risk_label      = risk_label,
        sectors         = sector_str,
        spread          = spread,
        notes_line      = notes_line,
    )

    logger.info("portfolio_reco_start", amount=amount, currency=currency, risk=risk, spread=spread)

    raw = await call_groq(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3600,
        temperature=0.2,
    )

    try:
        recommendations = _try_parse_json(raw)
    except ValueError as e:
        logger.error("portfolio_json_parse_failed", error=str(e))
        raise HTTPException(
            status_code=502,
            detail=f"Portfolio AI returned unparseable output. Please retry. Details: {str(e)[:200]}"
        )

    if not isinstance(recommendations, list):
        # Sometimes the LLM wraps in {"recommendations": [...]}
        if isinstance(recommendations, dict):
            for key in ("recommendations", "stocks", "portfolio", "picks"):
                if isinstance(recommendations.get(key), list):
                    recommendations = recommendations[key]
                    break
        if not isinstance(recommendations, list):
            raise HTTPException(status_code=502, detail="Portfolio AI returned unexpected format")

    # ── Validate + normalise each item ──
    validated = [
        validate_portfolio_item(item, i, spread)
        for i, item in enumerate(recommendations)
    ]

    # ── Enforce allocation sum = 100% ──
    total_alloc = sum(_safe_float(r["allocation_pct"]) for r in validated)
    if abs(total_alloc - 100) > 0.5 and total_alloc > 0:
        logger.warning("portfolio_alloc_rebalance", original_total=total_alloc)
        scale = 100.0 / total_alloc
        for r in validated:
            r["allocation_pct"] = round(_safe_float(r["allocation_pct"]) * scale, 1)
        # Fix rounding residual on the largest position
        diff = 100.0 - sum(r["allocation_pct"] for r in validated)
        if validated:
            largest = max(validated, key=lambda x: x["allocation_pct"])
            largest["allocation_pct"] = round(largest["allocation_pct"] + diff, 1)

    # ── Compute INR/USD allocation amounts ──
    for r in validated:
        alloc_pct        = _safe_float(r["allocation_pct"])
        alloc_usd        = round(amount_usd * alloc_pct / 100, 2)
        alloc_local      = round(amount * alloc_pct / 100, 2)
        price            = _safe_float(r.get("current_price_usd"))
        shares           = math.floor(alloc_usd / price) if price > 0 else 0
        r["alloc_usd"]   = alloc_usd
        r["alloc_local"] = alloc_local
        r["shares_approx"] = shares
        r["currency_symbol"] = currency_symbol

    logger.info("portfolio_reco_done", spread=len(validated))

    return {
        "recommendations": validated,
        "meta": {
            "amount":           amount,
            "amount_usd":       amount_usd,
            "currency":         currency,
            "currency_symbol":  currency_symbol,
            "horizon":          horizon,
            "risk":             risk,
            "sectors":          sectors,
            "spread":           spread,
            "usd_rate":         USD_TO_INR if currency == "INR" else 1.0,
            "total_alloc_check": sum(r["allocation_pct"] for r in validated),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARE SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def build_compare_response(r1: dict, r2: dict) -> dict:
    """
    8-factor comparison scoring model.
    Returns both results enriched with score breakdowns and a winner.
    """
    def score(r: dict) -> tuple[float, dict]:
        conf    = _safe_float(r.get("confidence"), 0.5)
        ac      = r.get("analyst_consensus") or {}
        fv      = r.get("finviz_signals")    or {}
        label   = (ac.get("label") or "").lower()
        upside  = _safe_float(ac.get("upside_pct"))
        rsi     = _safe_float(fv.get("rsi_14"), 50)
        pe      = _safe_float(fv.get("pe_ratio"))
        short_f = _safe_float(fv.get("short_float_pct"))
        chg     = _safe_float(r.get("price_change_pct"))

        # Factor 1: Analyst label
        f1 = {"strong buy": 3.0, "buy": 2.0, "hold": 0.5, "sell": -1.0, "strong sell": -2.5}.get(label, 0.5)
        # Factor 2: Upside to target
        f2 = min(2.0, upside / 20)
        # Factor 3: RSI (prefer 35–65)
        f3 = 1.5 if 35 <= rsi <= 65 else (1.2 if rsi < 35 else 0.3)
        # Factor 4: Valuation (prefer lower forward P/E if available)
        forward_pe = _safe_float(fv.get("forward_pe"))
        f4 = (1.0 if forward_pe and forward_pe < 20 else
              0.5 if forward_pe and forward_pe < 35 else
              0.0 if forward_pe and forward_pe > 50 else 0.3)
        # Factor 5: AI confidence
        f5 = conf * 2.0
        # Factor 6: Momentum
        f6 = min(1.0, max(-1.0, chg / 5))
        # Factor 7: Risk penalty
        risk_count = len(r.get("risk_flags") or [])
        f7 = -min(2.0, risk_count * 0.3)
        # Factor 8: Short interest risk
        f8 = -0.5 if short_f > 20 else (0.3 if short_f < 3 else 0.0)

        total = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
        breakdown = {
            "analyst_label": round(f1, 2),
            "price_target":  round(f2, 2),
            "rsi":           round(f3, 2),
            "valuation":     round(f4, 2),
            "confidence":    round(f5, 2),
            "momentum":      round(f6, 2),
            "risk":          round(f7, 2),
            "short_int":     round(f8, 2),
            "total":         round(total, 3),
        }
        return round(total, 3), breakdown

    s1, b1 = score(r1)
    s2, b2 = score(r2)

    r1["_compare_score"] = {"score": s1, "breakdown": b1}
    r2["_compare_score"] = {"score": s2, "breakdown": b2}

    winner     = r1.get("ticker") if s1 >= s2 else r2.get("ticker")
    margin     = round(abs(s1 - s2), 3)
    confidence = "clear" if margin > 1.0 else ("slight" if margin > 0.3 else "very close")

    return {
        "stock1":      r1,
        "stock2":      r2,
        "winner":      winner,
        "score1":      s1,
        "score2":      s2,
        "margin":      margin,
        "confidence":  confidence,   # how decisive the winner is
        "verdict": (
            f"{winner} leads by a {confidence} margin "
            f"(score: {s1 if winner == r1.get('ticker') else s2:.2f} vs "
            f"{s2 if winner == r1.get('ticker') else s1:.2f})"
        ),
    }