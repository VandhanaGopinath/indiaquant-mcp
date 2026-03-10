"""
MCP Tool Handlers: Options Tools
- get_options_chain
- calculate_greeks
- detect_unusual_activity
"""

import logging
from datetime import datetime, date

from src.modules.options_engine import (
    fetch_options_chain,
    calculate_greeks as _bs_greeks,
    black_scholes_greeks,
    implied_volatility_bisection,
    detect_unusual_options_activity
)
from src.modules.market_data import fetch_live_price

logger = logging.getLogger(__name__)

# India risk-free rate (RBI repo rate)
RISK_FREE_RATE = 0.065


async def get_options_chain_handler(args: dict) -> dict:
    """Handler for get_options_chain tool."""
    symbol = args.get("symbol", "").strip().upper()
    expiry = args.get("expiry", None)

    if not symbol:
        return {"error": True, "message": "symbol is required"}

    chain = await fetch_options_chain(symbol, expiry)

    # Limit returned strikes to ATM ± 10 strikes to avoid huge payloads
    spot = chain["spot_price"]

    def atm_filter(opts, n=12):
        """Return n strikes closest to spot on each side."""
        sorted_opts = sorted(opts, key=lambda x: abs(x["strike"] - spot))
        return sorted_opts[:n * 2]

    chain["calls"] = atm_filter(chain["calls"])
    chain["puts"] = atm_filter(chain["puts"])

    # Sort by strike
    chain["calls"].sort(key=lambda x: x["strike"])
    chain["puts"].sort(key=lambda x: x["strike"])

    chain["note"] = "Showing ~24 closest strikes to spot. Greeks computed using Black-Scholes from scratch."
    return chain


async def calculate_greeks_handler(args: dict) -> dict:
    """Handler for calculate_greeks tool. Full Black-Scholes implementation."""
    symbol = args.get("symbol", "").strip().upper()
    strike = args.get("strike")
    expiry_str = args.get("expiry")
    option_type = args.get("option_type", "CE").upper()
    option_price = args.get("option_price", None)

    if not all([symbol, strike, expiry_str]):
        return {"error": True, "message": "symbol, strike, and expiry are required"}

    # Fetch live spot price
    price_data = await fetch_live_price(symbol)
    spot = price_data["price"]

    # Time to expiry
    try:
        expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    except ValueError:
        return {"error": True, "message": "expiry must be in YYYY-MM-DD format"}

    today = date.today()
    T = max((expiry_dt - today).days / 365.0, 0.0)

    if T <= 0:
        T = 0.001  # Treat as nearly expired

    # Implied volatility
    if option_price and option_price > 0:
        sigma = implied_volatility_bisection(
            market_price=float(option_price),
            S=spot,
            K=float(strike),
            T=T,
            r=RISK_FREE_RATE,
            option_type=option_type
        )
        iv_source = "computed_from_market_price"
    else:
        # Use default historical volatility estimate
        sigma = 0.25
        iv_source = "default_25pct_assumed"

    greeks = black_scholes_greeks(
        S=spot,
        K=float(strike),
        T=T,
        r=RISK_FREE_RATE,
        sigma=sigma,
        option_type=option_type
    )

    # Interpret greeks for humans
    delta_interp = _interpret_delta(greeks["delta"], option_type)
    theta_interp = f"Option loses ₹{abs(greeks['theta']):.2f} per day due to time decay"
    vega_interp = f"Option gains/loses ₹{abs(greeks['vega']):.2f} for each 1% change in IV"

    return {
        "symbol": symbol,
        "option_contract": f"{symbol} {strike} {option_type} {expiry_str}",
        "spot_price": spot,
        "strike": float(strike),
        "expiry": expiry_str,
        "days_to_expiry": int(T * 365),
        "option_type": option_type,
        "risk_free_rate": RISK_FREE_RATE,
        "implied_volatility_pct": greeks["iv"],
        "iv_source": iv_source,
        "greeks": {
            "delta": greeks["delta"],
            "gamma": greeks["gamma"],
            "theta": greeks["theta"],
            "vega": greeks["vega"],
            "rho": greeks["rho"]
        },
        "option_pricing": {
            "theoretical_price": greeks["price"],
            "intrinsic_value": greeks["intrinsic_value"],
            "time_value": greeks["time_value"],
            "d1": greeks["d1"],
            "d2": greeks["d2"]
        },
        "interpretations": {
            "delta": delta_interp,
            "theta": theta_interp,
            "vega": vega_interp,
            "moneyness": "ITM" if greeks["intrinsic_value"] > 0 else "OTM"
        },
        "methodology": "Black-Scholes (implemented from scratch, no libraries)",
        "timestamp": datetime.now().isoformat()
    }


def _interpret_delta(delta: float, option_type: str) -> str:
    abs_delta = abs(delta)
    if abs_delta > 0.7:
        return f"Deep ITM. For every ₹1 move in spot, option moves ₹{abs_delta:.2f}"
    elif abs_delta > 0.5:
        return f"ITM. Moves ₹{abs_delta:.2f} per ₹1 spot move"
    elif abs_delta > 0.3:
        return f"Near ATM. Moves ₹{abs_delta:.2f} per ₹1 spot move"
    else:
        return f"OTM. Moves ₹{abs_delta:.2f} per ₹1 spot move. High theta decay risk."


async def detect_unusual_activity_handler(args: dict) -> dict:
    """Handler for detect_unusual_activity tool."""
    symbol = args.get("symbol", "").strip()
    if not symbol:
        return {"error": True, "message": "symbol is required"}

    return await detect_unusual_options_activity(symbol)
