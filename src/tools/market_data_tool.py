"""
MCP Tool Handlers: Market Data Tools
- get_live_price
- scan_market
- get_sector_heatmap
"""

import asyncio
import logging
from datetime import datetime

from src.modules.market_data import (
    fetch_live_price,
    fetch_nifty50_snapshot,
    fetch_sector_performance,
    SECTOR_MAP
)
from src.modules.signal_generator import compute_rsi

logger = logging.getLogger(__name__)


async def get_live_price_handler(args: dict) -> dict:
    """Handler for get_live_price tool."""
    symbol = args.get("symbol", "").strip()
    if not symbol:
        return {"error": True, "message": "symbol is required"}

    data = await fetch_live_price(symbol)

    # Add human-readable price movement
    movement = "▲" if data["change"] > 0 else "▼" if data["change"] < 0 else "─"
    data["movement"] = movement
    data["formatted"] = (
        f"{data['symbol']} {movement} ₹{data['price']:,.2f} "
        f"({data['change']:+.2f}, {data['change_pct']:+.2f}%) "
        f"Vol: {data['volume']:,}"
    )

    return data


async def scan_market_handler(args: dict) -> dict:
    """Handler for scan_market tool. Filters Nifty 50 by criteria."""
    criteria = args.get("filter_criteria", {})
    if not criteria:
        return {"error": True, "message": "filter_criteria object is required"}

    rsi_min = criteria.get("rsi_min", 0)
    rsi_max = criteria.get("rsi_max", 100)
    sector_filter = criteria.get("sector", "").strip()
    min_change = criteria.get("min_change_pct", -100)
    max_change = criteria.get("max_change_pct", 100)
    min_vol_ratio = criteria.get("min_volume_ratio", 0)

    # Get sector-filtered symbols if needed
    if sector_filter:
        sector_upper = sector_filter.upper()
        matching_sector = None
        for s in SECTOR_MAP:
            if s.upper() == sector_upper or sector_upper in s.upper():
                matching_sector = s
                break

        if matching_sector:
            symbols_to_scan = SECTOR_MAP[matching_sector]
        else:
            return {
                "error": True,
                "message": f"Sector '{sector_filter}' not found. Available: {list(SECTOR_MAP.keys())}"
            }
    else:
        symbols_to_scan = None  # Will use all Nifty 50

    # Fetch snapshot
    snapshot = await fetch_nifty50_snapshot()

    # Filter by requested symbols
    if symbols_to_scan:
        snapshot = [s for s in snapshot if s["symbol"] in symbols_to_scan]

    # Compute RSI for each stock and apply filters
    import pandas as pd
    results = []

    for stock in snapshot:
        close_vals = stock.get("close_series", [])
        if len(close_vals) < 14:
            continue

        # Compute RSI
        close_series = pd.Series(close_vals)
        rsi_series = compute_rsi(close_series)
        rsi_val = float(rsi_series.iloc[-1])

        # Apply filters
        if not (rsi_min <= rsi_val <= rsi_max):
            continue
        if not (min_change <= stock["change_pct"] <= max_change):
            continue
        if stock["volume_ratio"] < min_vol_ratio:
            continue

        # Determine signal from RSI
        if rsi_val < 30:
            rsi_signal = "OVERSOLD (BUY)"
        elif rsi_val > 70:
            rsi_signal = "OVERBOUGHT (SELL)"
        else:
            rsi_signal = "NEUTRAL"

        # Find sector
        symbol_sector = "Unknown"
        for sec, syms in SECTOR_MAP.items():
            if stock["symbol"] in syms:
                symbol_sector = sec
                break

        results.append({
            "symbol": stock["symbol"],
            "price": stock["price"],
            "change_pct": stock["change_pct"],
            "rsi": round(rsi_val, 1),
            "rsi_signal": rsi_signal,
            "volume_ratio": stock["volume_ratio"],
            "sector": symbol_sector,
            "volume": stock["volume"]
        })

    # Sort by most extreme RSI (oversold/overbought)
    results.sort(key=lambda x: abs(x["rsi"] - 50), reverse=True)

    return {
        "scan_results": results,
        "total_matches": len(results),
        "filter_applied": criteria,
        "scanned_symbols": len(snapshot),
        "timestamp": datetime.now().isoformat(),
        "summary": _summarize_scan(results)
    }


def _summarize_scan(results: list) -> str:
    if not results:
        return "No stocks matched the filter criteria."
    oversold = [r for r in results if r["rsi"] < 30]
    overbought = [r for r in results if r["rsi"] > 70]
    summary_parts = [f"Found {len(results)} matching stocks."]
    if oversold:
        syms = ", ".join(r["symbol"] for r in oversold[:3])
        summary_parts.append(f"{len(oversold)} oversold (RSI<30): {syms}")
    if overbought:
        syms = ", ".join(r["symbol"] for r in overbought[:3])
        summary_parts.append(f"{len(overbought)} overbought (RSI>70): {syms}")
    return " ".join(summary_parts)


async def get_sector_heatmap_handler(args: dict) -> dict:
    """Handler for get_sector_heatmap tool."""
    sector_data = await fetch_sector_performance()

    # Sort by change
    sorted_sectors = sorted(
        sector_data.items(),
        key=lambda x: x[1]["avg_change_pct"],
        reverse=True
    )

    # Overall market bias
    all_changes = [v["avg_change_pct"] for v in sector_data.values()]
    market_avg = sum(all_changes) / len(all_changes) if all_changes else 0

    heatmap = []
    for sector, data in sorted_sectors:
        emoji = "🟢" if data["avg_change_pct"] > 0.5 else "🔴" if data["avg_change_pct"] < -0.5 else "🟡"
        heatmap.append({
            "sector": sector,
            "avg_change_pct": data["avg_change_pct"],
            "signal": data["signal"],
            "best_stock_change": data["best_stock_change"],
            "worst_stock_change": data["worst_stock_change"],
            "stocks_tracked": data["stocks_tracked"],
            "indicator": emoji
        })

    return {
        "heatmap": heatmap,
        "market_avg_change": round(market_avg, 2),
        "market_bias": "BULLISH" if market_avg > 0.3 else "BEARISH" if market_avg < -0.3 else "NEUTRAL",
        "top_sector": sorted_sectors[0][0] if sorted_sectors else None,
        "bottom_sector": sorted_sectors[-1][0] if sorted_sectors else None,
        "timestamp": datetime.now().isoformat()
    }
