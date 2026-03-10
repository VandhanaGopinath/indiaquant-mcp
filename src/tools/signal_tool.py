"""
MCP Tool Handlers: Signal & Sentiment Tools
- generate_signal
- analyze_sentiment
"""

import logging
from src.modules.signal_generator import generate_trade_signal, fetch_news_sentiment

logger = logging.getLogger(__name__)


async def generate_signal_handler(args: dict) -> dict:
    """Handler for generate_signal tool."""
    symbol = args.get("symbol", "").strip()
    timeframe = args.get("timeframe", "1d")

    if not symbol:
        return {"error": True, "message": "symbol is required"}

    if timeframe not in ["1d", "1wk", "1mo"]:
        timeframe = "1d"

    result = await generate_trade_signal(symbol, timeframe)

    # Add human-readable recommendation
    signal = result["signal"]
    confidence = result["confidence"]
    price = result["current_price"]

    if signal == "BUY":
        recommendation = (
            f"📈 BUY {symbol.upper()} at ₹{price}. "
            f"Confidence: {confidence:.0f}%. "
            f"Stop loss: ₹{result['key_levels']['support']:.0f}, "
            f"Target: ₹{result['key_levels']['resistance']:.0f}"
        )
    elif signal == "SELL":
        recommendation = (
            f"📉 SELL/AVOID {symbol.upper()} at ₹{price}. "
            f"Confidence: {confidence:.0f}%. "
            f"Multiple bearish indicators aligned."
        )
    else:
        recommendation = (
            f"⏸️ HOLD/WAIT on {symbol.upper()} at ₹{price}. "
            f"Mixed signals, wait for clearer direction."
        )

    result["recommendation"] = recommendation
    return result


async def analyze_sentiment_handler(args: dict) -> dict:
    """Handler for analyze_sentiment tool."""
    symbol = args.get("symbol", "").strip()
    if not symbol:
        return {"error": True, "message": "symbol is required"}

    sentiment = await fetch_news_sentiment(symbol)

    score = sentiment.get("sentiment_score", 50)
    signal = sentiment.get("signal", "NEUTRAL")

    # Add interpretation
    if score >= 70:
        interpretation = f"Strongly bullish sentiment ({score:.0f}/100). Recent news is overwhelmingly positive."
    elif score >= 60:
        interpretation = f"Mildly bullish sentiment ({score:.0f}/100). More positive than negative coverage."
    elif score <= 30:
        interpretation = f"Strongly bearish sentiment ({score:.0f}/100). Negative news flow. Exercise caution."
    elif score <= 40:
        interpretation = f"Mildly bearish sentiment ({score:.0f}/100). Some concerns in recent coverage."
    else:
        interpretation = f"Neutral sentiment ({score:.0f}/100). Mixed or limited news flow."

    return {
        "symbol": symbol.upper(),
        "sentiment_score": score,
        "signal": signal,
        "interpretation": interpretation,
        "headlines": sentiment.get("headlines", []),
        "article_count": sentiment.get("article_count", 0),
        "data_source": sentiment.get("source", "newsapi"),
        "note": sentiment.get("message", None)
    }
