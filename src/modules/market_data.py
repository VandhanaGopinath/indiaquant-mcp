"""
Market Data Engine
Fetches live NSE/BSE prices, OHLC history, and sector data via yfinance.
Handles symbol normalization, caching, and edge cases.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional
import time

import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# NSE suffix for yfinance
NSE_SUFFIX = ".NS"
BSE_SUFFIX = ".BO"

# Cache: symbol -> (timestamp, data)
_price_cache: dict = {}
CACHE_TTL_SECONDS = 60  # 1 minute for live prices

# Nifty 50 constituents
NIFTY50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
    "TITAN", "BAJFINANCE", "WIPRO", "ULTRACEMCO", "ONGC",
    "NTPC", "POWERGRID", "M&M", "NESTLEIND", "TECHM",
    "HCLTECH", "BAJAJFINSV", "JSWSTEEL", "TATASTEEL", "GRASIM",
    "INDUSINDBK", "ADANIENT", "ADANIPORTS", "COALINDIA", "DIVISLAB",
    "CIPLA", "APOLLOHOSP", "EICHERMOT", "HEROMOTOCO", "DRREDDY",
    "BPCL", "TATACONSUM", "BRITANNIA", "HINDALCO", "SBILIFE",
    "HDFCLIFE", "BAJAJ-AUTO", "UPL", "TATAMOTORS", "LTF"
]

# Sector mapping
SECTOR_MAP = {
    "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK"],
    "Pharma": ["SUNPHARMA", "DIVISLAB", "CIPLA", "DRREDDY", "APOLLOHOSP"],
    "Auto": ["MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM"],
    "Energy": ["RELIANCE", "ONGC", "BPCL", "NTPC", "POWERGRID", "COALINDIA"],
    "Metal": ["JSWSTEEL", "TATASTEEL", "HINDALCO", "ADANIENT"],
    "Finance": ["BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "LTF"],
    "Infra": ["LT", "ADANIPORTS", "GRASIM", "ULTRACEMCO"],
    "Consumer": ["ASIANPAINT", "TITAN", "UPL"]
}

# Index symbols
INDEX_MAP = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "NIFTYMIDCAP": "^NSEMDCP50",
    "NIFTYIT": "^CNXIT",
}


def normalize_symbol(symbol: str) -> str:
    """Convert raw symbol to yfinance format."""
    s = symbol.upper().strip()
    # Check index map first
    if s in INDEX_MAP:
        return INDEX_MAP[s]
    # Already has suffix
    if s.startswith("^") or s.endswith(".NS") or s.endswith(".BO"):
        return s
    # Default: NSE suffix
    return s + NSE_SUFFIX


def _is_cache_valid(symbol: str) -> bool:
    if symbol not in _price_cache:
        return False
    ts, _ = _price_cache[symbol]
    return (time.time() - ts) < CACHE_TTL_SECONDS


def get_cached_or_fetch(symbol: str):
    if _is_cache_valid(symbol):
        return _price_cache[symbol][1]
    return None


def set_cache(symbol: str, data):
    _price_cache[symbol] = (time.time(), data)


async def fetch_live_price(symbol: str) -> dict:
    """Fetch current market price for a symbol."""
    yf_symbol = normalize_symbol(symbol)

    cached = get_cached_or_fetch(yf_symbol)
    if cached:
        cached["cached"] = True
        return cached

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _fetch_price_sync, yf_symbol, symbol)
    set_cache(yf_symbol, data)
    return data


def _fetch_price_sync(yf_symbol: str, raw_symbol: str) -> dict:
    """Synchronous yfinance price fetch."""
    ticker = yf.Ticker(yf_symbol)
    info = ticker.fast_info

    # fast_info is more reliable and faster
    try:
        price = float(info.last_price)
        prev_close = float(info.previous_close) if info.previous_close else price
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0
        volume = int(info.last_volume) if info.last_volume else 0
        market_cap = info.market_cap if hasattr(info, 'market_cap') and info.market_cap else None

        # Get 52w high/low from history if not in fast_info
        try:
            year_high = float(info.year_high)
            year_low = float(info.year_low)
        except Exception:
            year_high = year_low = None

        result = {
            "symbol": raw_symbol.upper(),
            "yf_symbol": yf_symbol,
            "price": round(price, 2),
            "prev_close": round(prev_close, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": volume,
            "market_cap": market_cap,
            "52w_high": year_high,
            "52w_low": year_low,
            "currency": "INR",
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }
        return result
    except Exception as e:
        # Fallback: use history
        hist = ticker.history(period="2d", interval="1d")
        if hist.empty:
            raise ValueError(f"No data found for symbol '{raw_symbol}'. Check if symbol is correct.")
        row = hist.iloc[-1]
        prev_row = hist.iloc[-2] if len(hist) > 1 else row
        price = float(row["Close"])
        prev_close = float(prev_row["Close"])
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0
        return {
            "symbol": raw_symbol.upper(),
            "yf_symbol": yf_symbol,
            "price": round(price, 2),
            "prev_close": round(prev_close, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": int(row.get("Volume", 0)),
            "market_cap": None,
            "52w_high": None,
            "52w_low": None,
            "currency": "INR",
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }


async def fetch_ohlc(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLC historical data."""
    yf_symbol = normalize_symbol(symbol)
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(None, _fetch_ohlc_sync, yf_symbol, period, interval)
    return df


def _fetch_ohlc_sync(yf_symbol: str, period: str, interval: str) -> pd.DataFrame:
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No historical data for {yf_symbol}")
    df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=["Close"])
    return df


async def fetch_sector_performance() -> dict:
    """Fetch sector-wise % change using sector ETF proxies and constituent averages."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_sector_sync)


def _fetch_sector_sync() -> dict:
    results = {}
    for sector, symbols in SECTOR_MAP.items():
        changes = []
        for sym in symbols:
            try:
                yf_sym = sym + NSE_SUFFIX
                ticker = yf.Ticker(yf_sym)
                info = ticker.fast_info
                price = float(info.last_price)
                prev = float(info.previous_close) if info.previous_close else price
                if prev:
                    chg = (price - prev) / prev * 100
                    changes.append(round(chg, 2))
            except Exception:
                continue
        if changes:
            avg_chg = round(sum(changes) / len(changes), 2)
            best = max(changes)
            worst = min(changes)
            results[sector] = {
                "avg_change_pct": avg_chg,
                "best_stock_change": best,
                "worst_stock_change": worst,
                "stocks_tracked": len(changes),
                "signal": "BULLISH" if avg_chg > 0.5 else "BEARISH" if avg_chg < -0.5 else "NEUTRAL"
            }
    return results


async def fetch_nifty50_snapshot() -> list[dict]:
    """Batch fetch all Nifty 50 stocks for scanning."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _batch_fetch_sync, NIFTY50_SYMBOLS)


def _batch_fetch_sync(symbols: list) -> list[dict]:
    results = []
    # Use yfinance download for batch efficiency
    yf_symbols = [s + NSE_SUFFIX for s in symbols]
    try:
        data = yf.download(
            yf_symbols,
            period="25d",
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True
        )
        for sym, yf_sym in zip(symbols, yf_symbols):
            try:
                if len(yf_symbols) > 1:
                    df = data[yf_sym] if yf_sym in data.columns.get_level_values(0) else None
                else:
                    df = data
                if df is None or df.empty:
                    continue
                df = df.dropna(subset=["Close"])
                if len(df) < 2:
                    continue
                price = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                vol_today = float(df["Volume"].iloc[-1])
                vol_avg = float(df["Volume"].tail(20).mean())
                vol_ratio = vol_today / vol_avg if vol_avg > 0 else 1.0
                change_pct = (price - prev) / prev * 100 if prev else 0
                results.append({
                    "symbol": sym,
                    "price": round(price, 2),
                    "change_pct": round(change_pct, 2),
                    "volume_ratio": round(vol_ratio, 2),
                    "close_series": df["Close"].values.tolist(),
                    "volume": int(vol_today)
                })
            except Exception as e:
                logger.debug(f"Skipping {sym}: {e}")
                continue
    except Exception as e:
        logger.error(f"Batch fetch error: {e}")
    return results
