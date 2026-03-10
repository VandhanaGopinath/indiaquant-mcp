#!/usr/bin/env python3
"""
IndiaQuant MCP Server
Real-time Indian stock market AI assistant via Model Context Protocol.
"""

import asyncio
import json
import logging
import sys
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions

from src.tools.market_data_tool import get_live_price_handler, scan_market_handler, get_sector_heatmap_handler
from src.tools.options_tool import get_options_chain_handler, calculate_greeks_handler, detect_unusual_activity_handler
from src.tools.signal_tool import generate_signal_handler, analyze_sentiment_handler
from src.tools.portfolio_tool import get_portfolio_pnl_handler, place_virtual_trade_handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("indiaquant-mcp")

server = Server("indiaquant-mcp")

TOOLS = [
    types.Tool(
        name="get_live_price",
        description="Fetch live NSE/BSE price for a stock or index. Returns current price, change%, volume, and market cap.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE/BSE ticker symbol (e.g. RELIANCE, HDFCBANK, NIFTY50, ^NSEI for Nifty index)"
                }
            },
            "required": ["symbol"]
        }
    ),
    types.Tool(
        name="get_options_chain",
        description="Fetch live options chain for a symbol with CE/PE open interest, volume, IV, and Greeks.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE symbol (e.g. NIFTY, BANKNIFTY, RELIANCE)"
                },
                "expiry": {
                    "type": "string",
                    "description": "Expiry date in YYYY-MM-DD format. Leave empty for nearest expiry."
                }
            },
            "required": ["symbol"]
        }
    ),
    types.Tool(
        name="analyze_sentiment",
        description="Analyze market sentiment for a stock using recent news headlines. Returns sentiment score, key headlines, and a buy/sell/hold signal.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol or company name (e.g. INFY, TCS, Reliance)"
                }
            },
            "required": ["symbol"]
        }
    ),
    types.Tool(
        name="generate_signal",
        description="Generate a technical trade signal (BUY/SELL/HOLD) with confidence score using RSI, MACD, Bollinger Bands, and sentiment.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE ticker symbol (e.g. HDFCBANK, WIPRO)"
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["1d", "1wk", "1mo"],
                    "description": "Candle timeframe for analysis. Default: 1d"
                }
            },
            "required": ["symbol"]
        }
    ),
    types.Tool(
        name="get_portfolio_pnl",
        description="Show current virtual portfolio with live P&L, position-level breakdown, and risk scores.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    types.Tool(
        name="place_virtual_trade",
        description="Place a virtual buy or sell trade in the paper trading portfolio.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE ticker symbol"
                },
                "qty": {
                    "type": "integer",
                    "description": "Number of shares to trade"
                },
                "side": {
                    "type": "string",
                    "enum": ["BUY", "SELL"],
                    "description": "Trade direction"
                }
            },
            "required": ["symbol", "qty", "side"]
        }
    ),
    types.Tool(
        name="calculate_greeks",
        description="Calculate Black-Scholes option Greeks (Delta, Gamma, Theta, Vega, Rho) for any option contract.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Underlying symbol (e.g. NIFTY, RELIANCE)"
                },
                "strike": {
                    "type": "number",
                    "description": "Option strike price"
                },
                "expiry": {
                    "type": "string",
                    "description": "Expiry date in YYYY-MM-DD format"
                },
                "option_type": {
                    "type": "string",
                    "enum": ["CE", "PE"],
                    "description": "Call (CE) or Put (PE)"
                },
                "option_price": {
                    "type": "number",
                    "description": "Current market price of the option (for IV calculation)"
                }
            },
            "required": ["symbol", "strike", "expiry", "option_type"]
        }
    ),
    types.Tool(
        name="detect_unusual_activity",
        description="Detect unusual options activity for a symbol: OI spikes, PCR anomalies, volume surges, and max pain.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE symbol (e.g. NIFTY, BANKNIFTY, INFY)"
                }
            },
            "required": ["symbol"]
        }
    ),
    types.Tool(
        name="scan_market",
        description="Scan Nifty 50 / Nifty 100 stocks by filter criteria: RSI range, sector, price change, volume surge, etc.",
        inputSchema={
            "type": "object",
            "properties": {
                "filter_criteria": {
                    "type": "object",
                    "description": "Filter object with optional keys: rsi_max, rsi_min, sector, min_change_pct, max_change_pct, min_volume_ratio",
                    "properties": {
                        "rsi_min": {"type": "number", "description": "Minimum RSI (e.g. 0)"},
                        "rsi_max": {"type": "number", "description": "Maximum RSI (e.g. 30 for oversold)"},
                        "sector": {"type": "string", "description": "Sector filter (IT, Banking, Pharma, Auto, FMCG, Energy, Metal)"},
                        "min_change_pct": {"type": "number", "description": "Minimum daily % change"},
                        "max_change_pct": {"type": "number", "description": "Maximum daily % change"},
                        "min_volume_ratio": {"type": "number", "description": "Min volume vs 20-day avg (e.g. 2.0 for 2x surge)"}
                    }
                }
            },
            "required": ["filter_criteria"]
        }
    ),
    types.Tool(
        name="get_sector_heatmap",
        description="Get sector-wise % change heatmap for Indian markets: IT, Banking, Pharma, Auto, FMCG, Metal, Energy, etc.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    )
]

HANDLERS = {
    "get_live_price": get_live_price_handler,
    "get_options_chain": get_options_chain_handler,
    "analyze_sentiment": analyze_sentiment_handler,
    "generate_signal": generate_signal_handler,
    "get_portfolio_pnl": get_portfolio_pnl_handler,
    "place_virtual_trade": place_virtual_trade_handler,
    "calculate_greeks": calculate_greeks_handler,
    "detect_unusual_activity": detect_unusual_activity_handler,
    "scan_market": scan_market_handler,
    "get_sector_heatmap": get_sector_heatmap_handler,
}


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    logger.info(f"Tool call: {name} with args: {arguments}")
    handler = HANDLERS.get(name)
    if not handler:
        raise ValueError(f"Unknown tool: {name}")
    try:
        result = await handler(arguments)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}", exc_info=True)
        error_result = {
            "error": True,
            "tool": name,
            "message": str(e),
            "hint": "Check symbol format (e.g. RELIANCE.NS for yfinance) or try again."
        }
        return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def main():
    logger.info("Starting IndiaQuant MCP Server...")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="indiaquant-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
