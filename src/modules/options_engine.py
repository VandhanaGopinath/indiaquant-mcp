"""
Options Chain Analyzer
- Fetches live options chain via yfinance
- Implements Black-Scholes from scratch: Delta, Gamma, Theta, Vega, Rho
- Max pain calculation
- Unusual activity detection: OI spikes, PCR anomalies, volume surges
"""

import asyncio
import logging
import math
from datetime import datetime, date
from typing import Optional

import yfinance as yf
import pandas as pd
import numpy as np

from src.modules.market_data import normalize_symbol, fetch_live_price

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# BLACK-SCHOLES FROM SCRATCH
# ─────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc for precision."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def black_scholes_greeks(
    S: float,        # Spot price
    K: float,        # Strike price
    T: float,        # Time to expiry in years
    r: float,        # Risk-free rate (annualized)
    sigma: float,    # Implied volatility (annualized)
    option_type: str # "CE" or "PE"
) -> dict:
    """
    Full Black-Scholes Greeks calculation from scratch.
    Returns: delta, gamma, theta, vega, rho, price, d1, d2
    """
    if T <= 0:
        # Expired option
        if option_type == "CE":
            intrinsic = max(S - K, 0)
        else:
            intrinsic = max(K - S, 0)
        return {
            "delta": 1.0 if S > K and option_type == "CE" else (0.0 if option_type == "CE" else -1.0 if S < K else 0.0),
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
            "price": intrinsic,
            "iv": sigma * 100,
            "d1": None,
            "d2": None,
            "intrinsic_value": intrinsic,
            "time_value": 0.0
        }

    if sigma <= 0:
        sigma = 0.0001  # Avoid division by zero

    # Core Black-Scholes formula
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    N_d1 = _norm_cdf(d1)
    N_d2 = _norm_cdf(d2)
    N_neg_d1 = _norm_cdf(-d1)
    N_neg_d2 = _norm_cdf(-d2)
    pdf_d1 = _norm_pdf(d1)

    discount = math.exp(-r * T)

    if option_type == "CE":
        price = S * N_d1 - K * discount * N_d2
        delta = N_d1
        rho = K * T * discount * N_d2 / 100
        intrinsic = max(S - K, 0)
    else:  # PE
        price = K * discount * N_neg_d2 - S * N_neg_d1
        delta = N_d1 - 1  # negative for puts
        rho = -K * T * discount * N_neg_d2 / 100
        intrinsic = max(K - S, 0)

    # Gamma (same for CE and PE)
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))

    # Theta (per calendar day, not per year)
    theta_annual = (
        -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
        - r * K * discount * (N_d2 if option_type == "CE" else N_neg_d2)
    )
    theta = theta_annual / 365  # daily theta

    # Vega (for 1% move in IV)
    vega = S * pdf_d1 * math.sqrt(T) / 100

    time_value = max(price - intrinsic, 0)

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
        "rho": round(rho, 4),
        "price": round(max(price, 0), 2),
        "iv": round(sigma * 100, 2),
        "d1": round(d1, 4),
        "d2": round(d2, 4),
        "intrinsic_value": round(intrinsic, 2),
        "time_value": round(time_value, 2)
    }


def implied_volatility_bisection(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    tolerance: float = 1e-5,
    max_iter: int = 200
) -> float:
    """
    Compute implied volatility using bisection method.
    Robust for edge cases where Newton-Raphson fails.
    """
    if T <= 0 or market_price <= 0:
        return 0.0

    low_vol, high_vol = 0.001, 5.0  # 0.1% to 500% IV range

    for _ in range(max_iter):
        mid_vol = (low_vol + high_vol) / 2
        g = black_scholes_greeks(S, K, T, r, mid_vol, option_type)
        model_price = g["price"]
        diff = model_price - market_price

        if abs(diff) < tolerance:
            return mid_vol

        if diff > 0:
            high_vol = mid_vol
        else:
            low_vol = mid_vol

        if high_vol - low_vol < tolerance:
            break

    return (low_vol + high_vol) / 2


# ─────────────────────────────────────────────
# OPTIONS CHAIN FETCHER
# ─────────────────────────────────────────────

def _map_option_symbol(symbol: str) -> str:
    """Map NSE option symbol to yfinance format."""
    symbol_map = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "^CNXFIN",
    }
    s = symbol.upper()
    if s in symbol_map:
        return symbol_map[s]
    return s + ".NS"


async def fetch_options_chain(symbol: str, expiry: Optional[str] = None) -> dict:
    """Fetch live options chain with Greeks for all strikes."""
    yf_symbol = _map_option_symbol(symbol)
    spot_data = await fetch_live_price(symbol)
    spot_price = spot_data["price"]

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _fetch_options_sync, yf_symbol, symbol, spot_price, expiry
    )
    return result


def _fetch_options_sync(yf_symbol: str, raw_symbol: str, spot_price: float, requested_expiry: Optional[str]) -> dict:
    ticker = yf.Ticker(yf_symbol)
    expirations = ticker.options

    if not expirations:
        raise ValueError(f"No options data available for {raw_symbol}. May not be a derivative-eligible stock.")

    # Select expiry
    if requested_expiry:
        if requested_expiry in expirations:
            selected_expiry = requested_expiry
        else:
            # Find closest expiry
            req_dt = datetime.strptime(requested_expiry, "%Y-%m-%d").date()
            closest = min(expirations, key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d").date() - req_dt).days))
            selected_expiry = closest
    else:
        selected_expiry = expirations[0]  # Nearest expiry

    chain = ticker.option_chain(selected_expiry)
    calls = chain.calls.copy()
    puts = chain.puts.copy()

    # Time to expiry in years
    expiry_dt = datetime.strptime(selected_expiry, "%Y-%m-%d").date()
    today = date.today()
    T = max((expiry_dt - today).days / 365.0, 0.001)

    # India risk-free rate (RBI repo rate proxy)
    r = 0.065

    def enrich_with_greeks(row, option_type: str) -> dict:
        K = float(row.get("strike", 0))
        last_price = float(row.get("lastPrice", 0)) if not pd.isna(row.get("lastPrice", 0)) else 0
        market_iv = float(row.get("impliedVolatility", 0.3)) if not pd.isna(row.get("impliedVolatility", 0.3)) else 0.3

        # Use market IV if available, else compute from price
        if last_price > 0.5 and market_iv < 0.01:
            sigma = implied_volatility_bisection(last_price, spot_price, K, T, r, option_type)
        else:
            sigma = market_iv if market_iv > 0.01 else 0.25

        greeks = black_scholes_greeks(spot_price, K, T, r, sigma, option_type)

        return {
            "strike": K,
            "lastPrice": round(last_price, 2),
            "bid": round(float(row.get("bid", 0) or 0), 2),
            "ask": round(float(row.get("ask", 0) or 0), 2),
            "volume": int(row.get("volume", 0) or 0),
            "openInterest": int(row.get("openInterest", 0) or 0),
            "iv": round(sigma * 100, 2),
            "delta": greeks["delta"],
            "gamma": greeks["gamma"],
            "theta": greeks["theta"],
            "vega": greeks["vega"],
            "intrinsic": greeks["intrinsic_value"],
            "time_value": greeks["time_value"],
            "itm": (spot_price > K if option_type == "CE" else spot_price < K)
        }

    ce_data = [enrich_with_greeks(row, "CE") for _, row in calls.iterrows()]
    pe_data = [enrich_with_greeks(row, "PE") for _, row in puts.iterrows()]

    # Sort by strike
    ce_data.sort(key=lambda x: x["strike"])
    pe_data.sort(key=lambda x: x["strike"])

    # PCR (Put-Call Ratio)
    total_ce_oi = sum(x["openInterest"] for x in ce_data)
    total_pe_oi = sum(x["openInterest"] for x in pe_data)
    pcr = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else 0

    # Max pain calculation
    max_pain = _calculate_max_pain(ce_data, pe_data)

    return {
        "symbol": raw_symbol.upper(),
        "spot_price": spot_price,
        "expiry": selected_expiry,
        "days_to_expiry": int(T * 365),
        "available_expiries": list(expirations[:6]),
        "pcr": pcr,
        "pcr_signal": "BULLISH" if pcr < 0.7 else "BEARISH" if pcr > 1.2 else "NEUTRAL",
        "max_pain": max_pain,
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "calls": ce_data,
        "puts": pe_data,
        "timestamp": datetime.now().isoformat()
    }


def _calculate_max_pain(calls: list, puts: list) -> float:
    """
    Max pain = strike price where total option buyer losses are maximized.
    Algorithm: for each possible strike, compute total payout to holders;
    max pain is the strike minimizing total payout.
    """
    all_strikes = sorted(set(c["strike"] for c in calls) | set(p["strike"] for p in puts))
    if not all_strikes:
        return 0.0

    min_pain = float("inf")
    max_pain_strike = all_strikes[0]

    for expiry_price in all_strikes:
        total_pain = 0.0

        # Pain for call holders (they lose when expiry < strike)
        for c in calls:
            if expiry_price < c["strike"]:
                total_pain += (c["strike"] - expiry_price) * c["openInterest"]

        # Pain for put holders (they lose when expiry > strike)
        for p in puts:
            if expiry_price > p["strike"]:
                total_pain += (expiry_price - p["strike"]) * p["openInterest"]

        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = expiry_price

    return max_pain_strike


async def detect_unusual_options_activity(symbol: str) -> dict:
    """Detect OI spikes, PCR anomalies, volume surges, and max pain divergence."""
    chain_data = await fetch_options_chain(symbol)

    spot = chain_data["spot_price"]
    max_pain = chain_data["max_pain"]
    pcr = chain_data["pcr"]
    calls = chain_data["calls"]
    puts = chain_data["puts"]

    alerts = []

    # 1. Max pain divergence
    pain_divergence_pct = abs(spot - max_pain) / spot * 100 if spot > 0 else 0
    if pain_divergence_pct > 3:
        direction = "BELOW" if max_pain < spot else "ABOVE"
        alerts.append({
            "type": "MAX_PAIN_DIVERGENCE",
            "severity": "HIGH" if pain_divergence_pct > 5 else "MEDIUM",
            "message": f"Spot {spot} is {pain_divergence_pct:.1f}% {direction} max pain {max_pain}. Expect mean reversion toward {max_pain}.",
            "value": pain_divergence_pct
        })

    # 2. Extreme PCR
    if pcr > 1.5:
        alerts.append({
            "type": "EXTREME_PUT_BUILDUP",
            "severity": "HIGH",
            "message": f"PCR = {pcr} (> 1.5). Extremely bearish OI positioning OR contrarian bullish signal.",
            "value": pcr
        })
    elif pcr < 0.5:
        alerts.append({
            "type": "EXTREME_CALL_BUILDUP",
            "severity": "HIGH",
            "message": f"PCR = {pcr} (< 0.5). Aggressive call writing or strong bullish positioning.",
            "value": pcr
        })

    # 3. OI concentration at specific strikes
    all_oi = [(c["strike"], c["openInterest"], "CE") for c in calls] + \
             [(p["strike"], p["openInterest"], "PE") for p in puts]
    all_oi.sort(key=lambda x: x[1], reverse=True)

    if all_oi:
        top_oi = all_oi[:3]
        for strike, oi, opt_type in top_oi:
            if oi > 0:
                alerts.append({
                    "type": "HIGH_OI_CONCENTRATION",
                    "severity": "MEDIUM",
                    "message": f"Massive {opt_type} OI of {oi:,} contracts at {strike} strike. Key support/resistance level.",
                    "strike": strike,
                    "oi": oi,
                    "option_type": opt_type
                })

    # 4. Volume/OI spikes (unusually high volume vs OI ratio)
    volume_anomalies = []
    for opt_list, opt_type in [(calls, "CE"), (puts, "PE")]:
        for opt in opt_list:
            if opt["openInterest"] > 0 and opt["volume"] > 0:
                vol_oi_ratio = opt["volume"] / opt["openInterest"]
                if vol_oi_ratio > 0.5 and opt["volume"] > 1000:
                    volume_anomalies.append({
                        "strike": opt["strike"],
                        "type": opt_type,
                        "volume": opt["volume"],
                        "oi": opt["openInterest"],
                        "vol_oi_ratio": round(vol_oi_ratio, 2)
                    })

    volume_anomalies.sort(key=lambda x: x["vol_oi_ratio"], reverse=True)
    if volume_anomalies:
        top = volume_anomalies[0]
        alerts.append({
            "type": "VOLUME_OI_SPIKE",
            "severity": "HIGH",
            "message": f"Unusual activity: {top['type']} {top['strike']} strike has vol/OI ratio of {top['vol_oi_ratio']} with {top['volume']:,} contracts traded.",
            "details": volume_anomalies[:3]
        })

    # 5. Implied move calculation
    atm_calls = [c for c in calls if abs(c["strike"] - spot) / spot < 0.02]
    atm_iv = sum(c["iv"] for c in atm_calls) / len(atm_calls) if atm_calls else 25
    dte = chain_data["days_to_expiry"]
    implied_move_pct = (atm_iv / 100) * math.sqrt(dte / 365) * 100 if dte > 0 else 0

    return {
        "symbol": symbol.upper(),
        "spot_price": spot,
        "expiry": chain_data["expiry"],
        "days_to_expiry": dte,
        "max_pain": max_pain,
        "max_pain_divergence_pct": round(pain_divergence_pct, 2),
        "pcr": pcr,
        "pcr_signal": chain_data["pcr_signal"],
        "atm_iv": round(atm_iv, 2),
        "implied_move_pct": round(implied_move_pct, 2),
        "alerts": alerts,
        "alert_count": len(alerts),
        "overall_signal": _derive_unusual_signal(alerts, pcr),
        "timestamp": datetime.now().isoformat()
    }


def _derive_unusual_signal(alerts: list, pcr: float) -> str:
    high_alerts = [a for a in alerts if a.get("severity") == "HIGH"]
    if not high_alerts:
        return "NO_UNUSUAL_ACTIVITY"
    bearish_signals = sum(1 for a in alerts if "BEARISH" in a.get("message", "") or "PUT" in a.get("type", ""))
    bullish_signals = sum(1 for a in alerts if "BULLISH" in a.get("message", "") or "CALL" in a.get("type", ""))
    if bearish_signals > bullish_signals:
        return "UNUSUAL_BEARISH_POSITIONING"
    elif bullish_signals > bearish_signals:
        return "UNUSUAL_BULLISH_POSITIONING"
    return "MIXED_UNUSUAL_ACTIVITY"
