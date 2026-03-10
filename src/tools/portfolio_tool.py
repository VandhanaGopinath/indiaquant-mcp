"""
MCP Tool Handlers: Portfolio Tools
- get_portfolio_pnl
- place_virtual_trade
"""

import logging
from src.modules.portfolio_manager import get_portfolio_pnl, place_trade

logger = logging.getLogger(__name__)


async def get_portfolio_pnl_handler(args: dict) -> dict:
    """Handler for get_portfolio_pnl tool."""
    return await get_portfolio_pnl()


async def place_virtual_trade_handler(args: dict) -> dict:
    """Handler for place_virtual_trade tool."""
    symbol = args.get("symbol", "").strip()
    qty = args.get("qty")
    side = args.get("side", "").upper()

    if not symbol:
        return {"error": True, "message": "symbol is required"}
    if not qty or int(qty) <= 0:
        return {"error": True, "message": "qty must be a positive integer"}
    if side not in ["BUY", "SELL"]:
        return {"error": True, "message": "side must be BUY or SELL"}

    return await place_trade(symbol, int(qty), side)
