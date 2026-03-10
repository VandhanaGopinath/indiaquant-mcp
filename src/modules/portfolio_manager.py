"""
Portfolio Risk Manager
- SQLite-backed virtual portfolio
- Real-time P&L with live prices
- Auto stop-loss and target management
- Historical volatility-based risk scoring
"""

import asyncio
import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.modules.market_data import fetch_live_price, fetch_ohlc

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "portfolio.db"

INITIAL_CASH = 1_000_000.0  # ₹10 lakh starting capital


def init_db():
    """Initialize SQLite database with required tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            qty INTEGER NOT NULL,
            avg_price REAL NOT NULL,
            side TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            stop_loss REAL,
            target REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            qty INTEGER NOT NULL,
            price REAL NOT NULL,
            side TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,
            pnl REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS account (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            cash REAL NOT NULL,
            total_pnl REAL NOT NULL DEFAULT 0
        )
    """)

    # Insert default account if not exists
    cur.execute("INSERT OR IGNORE INTO account (id, cash, total_pnl) VALUES (1, ?, 0)", (INITIAL_CASH,))
    conn.commit()
    conn.close()


def get_connection():
    init_db()
    return sqlite3.connect(str(DB_PATH))


# ─────────────────────────────────────────────
# TRADE EXECUTION
# ─────────────────────────────────────────────

async def place_trade(symbol: str, qty: int, side: str) -> dict:
    """Execute a virtual trade against live market price."""
    price_data = await fetch_live_price(symbol)
    price = price_data["price"]
    trade_value = price * qty

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("SELECT cash FROM account WHERE id = 1")
        row = cur.fetchone()
        cash = row[0] if row else INITIAL_CASH

        if side == "BUY":
            if cash < trade_value:
                return {
                    "success": False,
                    "error": f"Insufficient funds. Need ₹{trade_value:,.0f}, have ₹{cash:,.0f}",
                    "symbol": symbol,
                    "qty": qty,
                    "side": side
                }

            # Check if position already exists
            cur.execute("SELECT id, qty, avg_price FROM positions WHERE symbol = ?", (symbol.upper(),))
            existing = cur.fetchone()

            if existing:
                pos_id, existing_qty, existing_avg = existing
                new_qty = existing_qty + qty
                new_avg = (existing_avg * existing_qty + price * qty) / new_qty
                cur.execute(
                    "UPDATE positions SET qty = ?, avg_price = ? WHERE id = ?",
                    (new_qty, new_avg, pos_id)
                )
            else:
                # Auto stop-loss at 5%, target at 10%
                stop_loss = round(price * 0.95, 2)
                target = round(price * 1.10, 2)
                cur.execute(
                    "INSERT INTO positions (id, symbol, qty, avg_price, side, entry_time, stop_loss, target) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), symbol.upper(), qty, price, "LONG", datetime.now().isoformat(), stop_loss, target)
                )

            # Deduct cash
            cur.execute("UPDATE account SET cash = cash - ? WHERE id = 1", (trade_value,))

        elif side == "SELL":
            cur.execute("SELECT id, qty, avg_price FROM positions WHERE symbol = ?", (symbol.upper(),))
            existing = cur.fetchone()

            if not existing or existing[1] < qty:
                held = existing[1] if existing else 0
                return {
                    "success": False,
                    "error": f"Insufficient shares. Holding {held}, trying to sell {qty}",
                    "symbol": symbol
                }

            pos_id, existing_qty, avg_price = existing
            realized_pnl = (price - avg_price) * qty
            new_qty = existing_qty - qty

            if new_qty == 0:
                cur.execute("DELETE FROM positions WHERE id = ?", (pos_id,))
            else:
                cur.execute("UPDATE positions SET qty = ? WHERE id = ?", (new_qty, pos_id))

            # Add proceeds and update total PnL
            cur.execute("UPDATE account SET cash = cash + ?, total_pnl = total_pnl + ? WHERE id = 1",
                        (trade_value, realized_pnl))

        # Record trade
        trade_id = f"TRD-{uuid.uuid4().hex[:8].upper()}"
        cur.execute(
            "INSERT INTO trades (id, symbol, qty, price, side, timestamp, status, pnl) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (trade_id, symbol.upper(), qty, price, side, datetime.now().isoformat(), "EXECUTED",
             None if side == "BUY" else (price - 0) * qty)
        )

        conn.commit()

        return {
            "success": True,
            "order_id": trade_id,
            "symbol": symbol.upper(),
            "qty": qty,
            "side": side,
            "price": price,
            "trade_value": round(trade_value, 2),
            "status": "EXECUTED",
            "message": f"{'Bought' if side == 'BUY' else 'Sold'} {qty} shares of {symbol.upper()} at ₹{price}",
            "stop_loss": round(price * 0.95, 2) if side == "BUY" else None,
            "target": round(price * 1.10, 2) if side == "BUY" else None,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


# ─────────────────────────────────────────────
# PORTFOLIO P&L
# ─────────────────────────────────────────────

async def get_portfolio_pnl() -> dict:
    """Calculate real-time portfolio P&L with risk scores."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT cash, total_pnl FROM account WHERE id = 1")
    account = cur.fetchone()
    cash = account[0] if account else INITIAL_CASH
    realized_pnl = account[1] if account else 0

    cur.execute("SELECT id, symbol, qty, avg_price, side, entry_time, stop_loss, target FROM positions")
    positions_raw = cur.fetchall()
    conn.close()

    if not positions_raw:
        return {
            "positions": [],
            "summary": {
                "cash_balance": round(cash, 2),
                "invested_value": 0,
                "current_value": 0,
                "unrealized_pnl": 0,
                "realized_pnl": round(realized_pnl, 2),
                "total_pnl": round(realized_pnl, 2),
                "total_return_pct": 0,
                "portfolio_risk": "LOW"
            },
            "message": "Portfolio is empty. Use place_virtual_trade to start trading.",
            "timestamp": datetime.now().isoformat()
        }

    # Fetch live prices concurrently
    symbols = [row[1] for row in positions_raw]
    price_tasks = [fetch_live_price(sym) for sym in symbols]
    prices_data = await asyncio.gather(*price_tasks, return_exceptions=True)
    prices = {}
    for sym, p_data in zip(symbols, prices_data):
        if isinstance(p_data, Exception):
            logger.warning(f"Price fetch failed for {sym}: {p_data}")
            prices[sym] = None
        else:
            prices[sym] = p_data

    # Fetch volatility for risk scoring
    positions = []
    total_invested = 0
    total_current = 0
    total_unrealized = 0
    total_risk_score = 0

    for row in positions_raw:
        pos_id, symbol, qty, avg_price, side, entry_time, stop_loss, target = row
        price_info = prices.get(symbol)

        if price_info is None:
            ltp = avg_price
            change_pct = 0.0
        else:
            ltp = price_info["price"]
            change_pct = price_info["change_pct"]

        invested = avg_price * qty
        current = ltp * qty
        unrealized_pnl = current - invested
        unrealized_pct = (unrealized_pnl / invested * 100) if invested else 0

        # Risk score based on daily volatility (use change_pct as proxy)
        risk_score = _compute_position_risk(ltp, avg_price, abs(change_pct), stop_loss)

        # Stop-loss and target alerts
        alerts = []
        if stop_loss and ltp <= stop_loss:
            alerts.append(f"⚠️ STOP LOSS HIT at {stop_loss}! Consider exiting.")
        if target and ltp >= target:
            alerts.append(f"🎯 TARGET REACHED at {target}! Consider booking profit.")
        if unrealized_pct < -8:
            alerts.append(f"🔴 Position down {unrealized_pct:.1f}%. Review position.")

        positions.append({
            "symbol": symbol,
            "qty": qty,
            "avg_price": round(avg_price, 2),
            "ltp": round(ltp, 2),
            "change_pct": round(change_pct, 2),
            "invested": round(invested, 2),
            "current_value": round(current, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "unrealized_pct": round(unrealized_pct, 2),
            "stop_loss": stop_loss,
            "target": target,
            "risk_score": risk_score,
            "risk_level": "HIGH" if risk_score > 70 else "MEDIUM" if risk_score > 40 else "LOW",
            "alerts": alerts,
            "entry_time": entry_time
        })

        total_invested += invested
        total_current += current
        total_unrealized += unrealized_pnl
        total_risk_score += risk_score

    portfolio_risk_score = total_risk_score / len(positions) if positions else 0
    total_portfolio_value = cash + total_current
    total_pnl = total_unrealized + realized_pnl
    total_return_pct = (total_pnl / INITIAL_CASH * 100) if INITIAL_CASH else 0

    positions.sort(key=lambda x: abs(x["unrealized_pnl"]), reverse=True)

    return {
        "positions": positions,
        "summary": {
            "cash_balance": round(cash, 2),
            "invested_value": round(total_invested, 2),
            "current_value": round(total_current, 2),
            "total_portfolio_value": round(total_portfolio_value, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "realized_pnl": round(realized_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "portfolio_risk_score": round(portfolio_risk_score, 1),
            "portfolio_risk": "HIGH" if portfolio_risk_score > 70 else "MEDIUM" if portfolio_risk_score > 40 else "LOW",
            "position_count": len(positions),
            "initial_capital": INITIAL_CASH
        },
        "timestamp": datetime.now().isoformat()
    }


def _compute_position_risk(ltp: float, avg_price: float, daily_vol: float, stop_loss: Optional[float]) -> float:
    """
    Risk score 0-100 based on:
    - Daily volatility (higher vol = higher risk)
    - Drawdown from entry
    - Proximity to stop loss
    """
    # Drawdown component (0-40 pts)
    drawdown_pct = (avg_price - ltp) / avg_price * 100 if avg_price > 0 else 0
    drawdown_score = min(max(drawdown_pct * 4, 0), 40)

    # Volatility component (0-40 pts)
    vol_score = min(daily_vol * 4, 40)

    # Stop loss proximity (0-20 pts)
    if stop_loss and ltp > 0:
        sl_pct = (ltp - stop_loss) / ltp * 100
        sl_score = max(20 - sl_pct * 2, 0)
    else:
        sl_score = 10  # No stop loss = moderate risk

    return round(drawdown_score + vol_score + sl_score, 1)
