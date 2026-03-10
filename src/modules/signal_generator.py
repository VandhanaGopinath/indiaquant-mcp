"""
AI Trade Signal Generator
Computes RSI, MACD, Bollinger Bands using pandas-ta / manual implementation.
Detects chart patterns: Head & Shoulders, Double Top/Bottom.
Integrates news sentiment from NewsAPI.
Outputs BUY/SELL/HOLD with confidence score 0-100.
"""

import asyncio
import logging
import os
import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import httpx

from src.modules.market_data import fetch_ohlc, normalize_symbol

logger = logging.getLogger(__name__)

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")


# ─────────────────────────────────────────────
# TECHNICAL INDICATORS (manual, no TA-Lib dep)
# ─────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD Line, Signal Line, Histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }


def compute_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
    """Bollinger Bands: upper, middle (SMA), lower."""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pct_b = (close - lower) / (upper - lower)  # %B indicator
    bandwidth = (upper - lower) / sma * 100     # Bandwidth %
    return {
        "upper": upper,
        "middle": sma,
        "lower": lower,
        "pct_b": pct_b,
        "bandwidth": bandwidth
    }


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ─────────────────────────────────────────────
# CHART PATTERN DETECTION
# ─────────────────────────────────────────────

def detect_patterns(df: pd.DataFrame) -> list[dict]:
    """Detect common chart patterns."""
    patterns = []
    close = df["Close"].values
    n = len(close)
    if n < 30:
        return patterns

    # Local peaks and troughs (for pattern matching)
    def local_peaks(arr, window=5):
        peaks = []
        for i in range(window, len(arr) - window):
            if arr[i] == max(arr[i-window:i+window+1]):
                peaks.append((i, arr[i]))
        return peaks

    def local_troughs(arr, window=5):
        troughs = []
        for i in range(window, len(arr) - window):
            if arr[i] == min(arr[i-window:i+window+1]):
                troughs.append((i, arr[i]))
        return troughs

    peaks = local_peaks(close)
    troughs = local_troughs(close)

    # Double Top detection
    if len(peaks) >= 2:
        p1_idx, p1_val = peaks[-2]
        p2_idx, p2_val = peaks[-1]
        if abs(p1_val - p2_val) / p1_val < 0.03 and (p2_idx - p1_idx) >= 10:
            patterns.append({
                "pattern": "DOUBLE_TOP",
                "signal": "BEARISH",
                "confidence": 70,
                "description": f"Double top at ~{p1_val:.0f}. Bearish reversal signal.",
            })

    # Double Bottom detection
    if len(troughs) >= 2:
        t1_idx, t1_val = troughs[-2]
        t2_idx, t2_val = troughs[-1]
        if abs(t1_val - t2_val) / t1_val < 0.03 and (t2_idx - t1_idx) >= 10:
            patterns.append({
                "pattern": "DOUBLE_BOTTOM",
                "signal": "BULLISH",
                "confidence": 70,
                "description": f"Double bottom at ~{t1_val:.0f}. Bullish reversal signal.",
            })

    # Head and Shoulders (3 peaks, middle highest)
    if len(peaks) >= 3:
        left_idx, left_val = peaks[-3]
        head_idx, head_val = peaks[-2]
        right_idx, right_val = peaks[-1]
        if (head_val > left_val * 1.02 and head_val > right_val * 1.02
                and abs(left_val - right_val) / left_val < 0.05):
            patterns.append({
                "pattern": "HEAD_AND_SHOULDERS",
                "signal": "BEARISH",
                "confidence": 75,
                "description": f"H&S: head at {head_val:.0f}, shoulders at ~{left_val:.0f}. Strong bearish reversal.",
            })

    # Inverse Head and Shoulders
    if len(troughs) >= 3:
        l_idx, l_val = troughs[-3]
        h_idx, h_val = troughs[-2]
        r_idx, r_val = troughs[-1]
        if (h_val < l_val * 0.98 and h_val < r_val * 0.98
                and abs(l_val - r_val) / l_val < 0.05):
            patterns.append({
                "pattern": "INVERSE_HEAD_AND_SHOULDERS",
                "signal": "BULLISH",
                "confidence": 75,
                "description": f"Inverse H&S: head at {h_val:.0f}. Bullish reversal signal.",
            })

    # Golden Cross / Death Cross (EMA 50 vs EMA 200)
    if n >= 60:
        ema50 = close[-50:].mean()  # Simplified for pattern check
        ema20 = close[-20:].mean()
        if ema20 > ema50 * 1.005:
            patterns.append({
                "pattern": "GOLDEN_CROSS",
                "signal": "BULLISH",
                "confidence": 65,
                "description": "20-day EMA > 50-day EMA. Bullish trend."
            })
        elif ema20 < ema50 * 0.995:
            patterns.append({
                "pattern": "DEATH_CROSS",
                "signal": "BEARISH",
                "confidence": 65,
                "description": "20-day EMA < 50-day EMA. Bearish trend."
            })

    return patterns


# ─────────────────────────────────────────────
# SENTIMENT ANALYSIS
# ─────────────────────────────────────────────

async def fetch_news_sentiment(symbol: str) -> dict:
    """Fetch news headlines and compute sentiment score using NewsAPI."""
    company_name = _symbol_to_company(symbol)
    query = f"{company_name} stock NSE"

    if not NEWSAPI_KEY:
        return {
            "sentiment_score": 50,
            "signal": "NEUTRAL",
            "headlines": [],
            "source": "no_api_key",
            "message": "Set NEWSAPI_KEY environment variable for live sentiment."
        }

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": NEWSAPI_KEY,
        "from": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = data.get("articles", [])
        if not articles:
            return {"sentiment_score": 50, "signal": "NEUTRAL", "headlines": [], "source": "newsapi"}

        headlines = [
            {
                "title": a["title"],
                "source": a["source"]["name"],
                "publishedAt": a["publishedAt"][:10],
                "sentiment": _classify_headline(a["title"])
            }
            for a in articles[:8]
        ]

        scores = [h["sentiment"]["score"] for h in headlines]
        avg_score = sum(scores) / len(scores) if scores else 50

        return {
            "sentiment_score": round(avg_score, 1),
            "signal": "BULLISH" if avg_score > 60 else "BEARISH" if avg_score < 40 else "NEUTRAL",
            "headlines": headlines,
            "article_count": len(headlines),
            "source": "newsapi"
        }

    except Exception as e:
        logger.warning(f"NewsAPI error: {e}")
        return {"sentiment_score": 50, "signal": "NEUTRAL", "headlines": [], "error": str(e)}


def _classify_headline(title: str) -> dict:
    """Simple keyword-based headline sentiment."""
    title_lower = title.lower()

    bullish_words = [
        "surge", "rally", "gains", "jumps", "soars", "beats", "strong",
        "record", "high", "growth", "profit", "buy", "upgrade", "positive",
        "outperform", "bullish", "target", "rise", "up", "boost", "winner"
    ]
    bearish_words = [
        "fall", "drop", "decline", "loss", "slump", "misses", "weak",
        "sell", "downgrade", "negative", "bearish", "crash", "down", "cut",
        "underperform", "concern", "risk", "warning", "reduce", "disappoints"
    ]

    bull_count = sum(1 for w in bullish_words if w in title_lower)
    bear_count = sum(1 for w in bearish_words if w in title_lower)

    if bull_count > bear_count:
        score = min(50 + (bull_count - bear_count) * 15, 95)
        label = "BULLISH"
    elif bear_count > bull_count:
        score = max(50 - (bear_count - bull_count) * 15, 5)
        label = "BEARISH"
    else:
        score = 50
        label = "NEUTRAL"

    return {"score": score, "label": label}


def _symbol_to_company(symbol: str) -> str:
    """Map NSE symbol to common company name for news search."""
    mapping = {
        "TCS": "Tata Consultancy Services",
        "INFY": "Infosys",
        "RELIANCE": "Reliance Industries",
        "HDFCBANK": "HDFC Bank",
        "ICICIBANK": "ICICI Bank",
        "WIPRO": "Wipro",
        "SBIN": "State Bank India",
        "MARUTI": "Maruti Suzuki",
        "SUNPHARMA": "Sun Pharma",
        "TATAMOTORS": "Tata Motors",
        "BAJFINANCE": "Bajaj Finance",
        "KOTAKBANK": "Kotak Bank",
        "NIFTY": "Nifty 50 index",
        "BANKNIFTY": "Bank Nifty index"
    }
    return mapping.get(symbol.upper(), symbol.upper())


# ─────────────────────────────────────────────
# SIGNAL GENERATOR (combines all signals)
# ─────────────────────────────────────────────

async def generate_trade_signal(symbol: str, timeframe: str = "1d") -> dict:
    """
    Generate composite BUY/SELL/HOLD signal.
    Weights: RSI(25%) + MACD(25%) + BB(20%) + Patterns(15%) + Sentiment(15%)
    """
    # Fetch historical data
    period_map = {"1d": "6mo", "1wk": "2y", "1mo": "5y"}
    period = period_map.get(timeframe, "6mo")
    df = await fetch_ohlc(symbol, period=period, interval=timeframe)

    if len(df) < 30:
        raise ValueError(f"Insufficient historical data for {symbol}")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ── RSI Analysis (25 points) ──
    rsi = compute_rsi(close)
    rsi_val = float(rsi.iloc[-1])
    rsi_prev = float(rsi.iloc[-2]) if len(rsi) > 1 else rsi_val

    if rsi_val < 30:
        rsi_score = 80  # Strongly oversold = bullish
        rsi_signal = "BULLISH"
        rsi_reason = f"RSI {rsi_val:.1f} is oversold (<30)"
    elif rsi_val < 40:
        rsi_score = 65
        rsi_signal = "BULLISH"
        rsi_reason = f"RSI {rsi_val:.1f} approaching oversold"
    elif rsi_val > 70:
        rsi_score = 20  # Overbought = bearish
        rsi_signal = "BEARISH"
        rsi_reason = f"RSI {rsi_val:.1f} is overbought (>70)"
    elif rsi_val > 60:
        rsi_score = 35
        rsi_signal = "BEARISH"
        rsi_reason = f"RSI {rsi_val:.1f} approaching overbought"
    else:
        rsi_score = 50
        rsi_signal = "NEUTRAL"
        rsi_reason = f"RSI {rsi_val:.1f} in neutral zone"

    # RSI divergence bonus
    if rsi_prev < rsi_val and rsi_signal == "BULLISH":
        rsi_score = min(rsi_score + 5, 95)

    # ── MACD Analysis (25 points) ──
    macd_data = compute_macd(close)
    macd_val = float(macd_data["macd"].iloc[-1])
    signal_val = float(macd_data["signal"].iloc[-1])
    hist_val = float(macd_data["histogram"].iloc[-1])
    hist_prev = float(macd_data["histogram"].iloc[-2]) if len(macd_data["histogram"]) > 1 else 0

    if macd_val > signal_val and hist_val > 0:
        macd_score = 75 if hist_val > hist_prev else 65
        macd_signal = "BULLISH"
        macd_reason = "MACD above signal line (bullish crossover)"
    elif macd_val < signal_val and hist_val < 0:
        macd_score = 25 if hist_val < hist_prev else 35
        macd_signal = "BEARISH"
        macd_reason = "MACD below signal line (bearish crossover)"
    else:
        macd_score = 50
        macd_signal = "NEUTRAL"
        macd_reason = "MACD at crossover point"

    # ── Bollinger Bands (20 points) ──
    bb = compute_bollinger_bands(close)
    pct_b = float(bb["pct_b"].iloc[-1]) if not math.isnan(float(bb["pct_b"].iloc[-1])) else 0.5
    bandwidth = float(bb["bandwidth"].iloc[-1]) if not math.isnan(float(bb["bandwidth"].iloc[-1])) else 10

    current_price = float(close.iloc[-1])
    bb_upper = float(bb["upper"].iloc[-1])
    bb_lower = float(bb["lower"].iloc[-1])
    bb_mid = float(bb["middle"].iloc[-1])

    if pct_b < 0.05:
        bb_score = 80
        bb_signal = "BULLISH"
        bb_reason = f"Price near lower BB ({bb_lower:.0f}). Bounce likely."
    elif pct_b > 0.95:
        bb_score = 20
        bb_signal = "BEARISH"
        bb_reason = f"Price near upper BB ({bb_upper:.0f}). Pullback likely."
    elif pct_b > 0.5 and current_price > bb_mid:
        bb_score = 60
        bb_signal = "BULLISH"
        bb_reason = f"Price above BB midline. Uptrend intact."
    else:
        bb_score = 40
        bb_signal = "BEARISH"
        bb_reason = f"Price below BB midline. Downtrend."

    # ── Chart Patterns (15 points) ──
    patterns = detect_patterns(df)
    if patterns:
        bullish_patterns = [p for p in patterns if p["signal"] == "BULLISH"]
        bearish_patterns = [p for p in patterns if p["signal"] == "BEARISH"]
        if len(bullish_patterns) > len(bearish_patterns):
            pattern_score = 75
            pattern_signal = "BULLISH"
        elif len(bearish_patterns) > len(bullish_patterns):
            pattern_score = 25
            pattern_signal = "BEARISH"
        else:
            pattern_score = 50
            pattern_signal = "NEUTRAL"
    else:
        pattern_score = 50
        pattern_signal = "NEUTRAL"

    # ── Sentiment Analysis (15 points) ──
    sentiment = await fetch_news_sentiment(symbol)
    sentiment_raw = sentiment.get("sentiment_score", 50)
    sentiment_signal = sentiment.get("signal", "NEUTRAL")

    # ── Volume Confirmation ──
    vol_avg = float(volume.tail(20).mean())
    vol_today = float(volume.iloc[-1])
    vol_ratio = vol_today / vol_avg if vol_avg > 0 else 1.0

    # ── Composite Score ──
    composite = (
        rsi_score * 0.25 +
        macd_score * 0.25 +
        bb_score * 0.20 +
        pattern_score * 0.15 +
        sentiment_raw * 0.15
    )

    # Volume multiplier: high volume confirms signal
    if vol_ratio > 1.5:
        composite = composite * 1.05 if composite > 50 else composite * 0.95

    composite = max(0, min(100, composite))

    # ── Final Signal ──
    if composite >= 65:
        final_signal = "BUY"
    elif composite <= 35:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    # ── Support / Resistance ──
    support = float(df["Low"].tail(20).min())
    resistance = float(df["High"].tail(20).max())

    # ── 52-week context ──
    high_52w = float(df["High"].tail(252).max()) if len(df) >= 252 else float(df["High"].max())
    low_52w = float(df["Low"].tail(252).min()) if len(df) >= 252 else float(df["Low"].min())

    return {
        "symbol": symbol.upper(),
        "signal": final_signal,
        "confidence": round(composite, 1),
        "timeframe": timeframe,
        "current_price": round(current_price, 2),
        "components": {
            "rsi": {
                "value": round(rsi_val, 1),
                "signal": rsi_signal,
                "score": round(rsi_score, 1),
                "reason": rsi_reason,
                "weight": "25%"
            },
            "macd": {
                "macd_line": round(macd_val, 4),
                "signal_line": round(signal_val, 4),
                "histogram": round(hist_val, 4),
                "signal": macd_signal,
                "score": round(macd_score, 1),
                "reason": macd_reason,
                "weight": "25%"
            },
            "bollinger_bands": {
                "upper": round(bb_upper, 2),
                "middle": round(bb_mid, 2),
                "lower": round(bb_lower, 2),
                "pct_b": round(pct_b, 3),
                "bandwidth": round(bandwidth, 2),
                "signal": bb_signal,
                "score": round(bb_score, 1),
                "reason": bb_reason,
                "weight": "20%"
            },
            "chart_patterns": {
                "detected": patterns,
                "signal": pattern_signal,
                "score": round(pattern_score, 1),
                "weight": "15%"
            },
            "sentiment": {
                "score": sentiment_raw,
                "signal": sentiment_signal,
                "headlines_analyzed": len(sentiment.get("headlines", [])),
                "weight": "15%"
            }
        },
        "volume": {
            "today": int(vol_today),
            "avg_20d": int(vol_avg),
            "ratio": round(vol_ratio, 2),
            "surge": vol_ratio > 1.5
        },
        "key_levels": {
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "52w_high": round(high_52w, 2),
            "52w_low": round(low_52w, 2)
        },
        "timestamp": datetime.now().isoformat()
    }
