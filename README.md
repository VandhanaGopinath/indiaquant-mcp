# IndiaQuant MCP — Indian Stock Market AI Assistant

> **Real-time Indian stock market intelligence for Claude via Model Context Protocol.**
> Built with 100% free APIs. No broker account required. No mock data.

---

## What It Does

IndiaQuant MCP gives Claude full awareness of live Indian markets. Ask it anything:

```
"Should I buy Reliance right now?"
"What's the max pain for Nifty this expiry?"
"Show me oversold IT stocks with RSI below 30"
"Place a virtual buy of 10 shares of HDFCBANK"
"Is there unusual options activity on Infosys?"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Claude Desktop                          │
│                   (MCP Host / AI Agent)                      │
└───────────────────────┬─────────────────────────────────────┘
                        │  MCP stdio protocol
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   server.py (MCP Server)                     │
│   Tool registration, schema validation, error handling       │
└──┬─────────┬──────────┬──────────┬──────────┬──────────────┘
   │         │          │          │          │
   ▼         ▼          ▼          ▼          ▼
┌──────┐ ┌──────┐  ┌────────┐ ┌───────┐ ┌──────────┐
│Market│ │Options│  │Signal  │ │Sentim.│ │Portfolio │
│Data  │ │Engine │  │Generat.│ │Engine │ │Manager   │
│Engine│ │(BS)   │  │(TA)    │ │(NLP)  │ │(SQLite)  │
└──┬───┘ └──┬───┘  └───┬────┘ └──┬────┘ └────┬─────┘
   │        │          │         │            │
   ▼        ▼          ▼         ▼            ▼
┌──────┐ ┌─────────┐ ┌──────┐ ┌──────────┐ ┌─────────┐
│yfinance│ │yfinance │ │pandas│ │NewsAPI   │ │SQLite DB│
│(NSE/BSE│ │options  │ │-ta   │ │(free)    │ │portfolio│
│prices) │ │chain    │ │      │ │          │ │.db      │
└────────┘ └─────────┘ └──────┘ └──────────┘ └─────────┘
```

### Module Breakdown

| Module | File | Responsibility |
|--------|------|----------------|
| Market Data Engine | `src/modules/market_data.py` | Live prices, OHLC history, sector data, batch fetching |
| Options Engine | `src/modules/options_engine.py` | Options chain, Black-Scholes from scratch, max pain, PCR |
| Signal Generator | `src/modules/signal_generator.py` | RSI, MACD, BB, chart patterns, sentiment fusion |
| Portfolio Manager | `src/modules/portfolio_manager.py` | Virtual trades, P&L, risk scoring, SQLite persistence |
| MCP Tools Layer | `src/tools/` | 10 MCP tool handlers with schema validation |

---

## 10 MCP Tools

| Tool | Inputs | Returns |
|------|--------|---------|
| `get_live_price` | symbol | price, change%, volume, 52w range |
| `get_options_chain` | symbol, expiry? | CE/PE OI, Greeks, PCR, max pain |
| `analyze_sentiment` | symbol | score 0-100, headlines, signal |
| `generate_signal` | symbol, timeframe? | BUY/SELL/HOLD, confidence %, component breakdown |
| `get_portfolio_pnl` | — | positions, P&L, risk scores |
| `place_virtual_trade` | symbol, qty, side | order_id, execution price, stop-loss |
| `calculate_greeks` | symbol, strike, expiry, type | delta, gamma, theta, vega, rho |
| `detect_unusual_activity` | symbol | alerts, PCR anomalies, OI spikes, max pain divergence |
| `scan_market` | filter_criteria | matching Nifty 50 stocks with RSI/volume data |
| `get_sector_heatmap` | — | all sectors sorted by % change |

---

## Setup Guide

### 1. Clone & Install

```bash
git clone https://github.com/yourname/indiaquant-mcp
cd indiaquant-mcp
pip install -r requirements.txt
```

### 2. Get Free API Keys

| Service | URL | Limits | Usage |
|---------|-----|--------|-------|
| **NewsAPI** | https://newsapi.org/register | 100 req/day free | Sentiment analysis |
| **Alpha Vantage** | https://www.alphavantage.co/support/#api-key | 25 req/day free | Macro indicators |
| **yfinance** | No key needed | Unlimited | Prices, options, history |

### 3. Configure Environment

```bash
export NEWSAPI_KEY="your_key_here"
export ALPHA_VANTAGE_KEY="your_key_here"
```

Or create a `.env` file:
```
NEWSAPI_KEY=your_key_here
ALPHA_VANTAGE_KEY=your_key_here
```

### 4. Test the Server

```bash
# Run unit tests (no API calls needed)
python -m pytest tests/ -v

# Quick smoke test
python -c "
import asyncio
from src.modules.market_data import fetch_live_price
result = asyncio.run(fetch_live_price('RELIANCE'))
print(result)
"
```

### 5. Connect to Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "indiaquant": {
      "command": "python",
      "args": ["/absolute/path/to/indiaquant-mcp/server.py"],
      "env": {
        "NEWSAPI_KEY": "your_key",
        "ALPHA_VANTAGE_KEY": "your_key",
        "PYTHONPATH": "/absolute/path/to/indiaquant-mcp"
      }
    }
  }
}
```

Restart Claude Desktop. You should see `indiaquant` in the tools panel.

---

## Symbol Format Reference

| You type | Meaning |
|----------|---------|
| `RELIANCE` | Reliance Industries (NSE) |
| `HDFCBANK` | HDFC Bank (NSE) |
| `NIFTY` | Nifty 50 Index |
| `BANKNIFTY` | Bank Nifty Index |
| `SENSEX` | BSE Sensex |
| `TCS` | Tata Consultancy Services |

---

## Architecture Decisions & Trade-offs

### 1. yfinance as Primary Data Source
**Decision:** Use yfinance (Yahoo Finance) for all market data.  
**Rationale:** Free, unlimited, covers NSE/BSE symbols with `.NS`/`.BO` suffix, provides options chain data.  
**Trade-off:** Not real-time tick data (15-min delay during market hours). For trading signals this is acceptable; for HFT it would not be.  
**Alternative considered:** NSE official API — rate limited and requires login. Rejected.

### 2. Black-Scholes from Scratch
**Decision:** Implement all Greeks using pure Python math (no scipy, no TA-Lib).  
**Rationale:** Assignment requirement. Also avoids heavy C dependency (TA-Lib install often fails on various platforms).  
**Implementation:** Uses `math.erfc` for the normal CDF — more numerically stable than polynomial approximations. Bisection method for IV (not Newton-Raphson) because bisection is guaranteed to converge even for edge cases (deep ITM/OTM).  
**Trade-off:** Slightly slower than vectorized scipy, but negligible for per-request computation.

### 3. Composite Signal with Weighted Components
**Decision:** Combine RSI (25%) + MACD (25%) + Bollinger Bands (20%) + Chart Patterns (15%) + Sentiment (15%).  
**Rationale:** No single indicator is reliable alone. RSI and MACD together handle both momentum and trend. BB adds mean-reversion context. Patterns add structure awareness. Sentiment adds news flow.  
**Calibration:** Volume surge acts as a confidence multiplier (1.05x for signal, 0.95x against). This prevents high-confidence calls on low-liquidity moves.  
**Trade-off:** Sentiment weight is lower (15%) because NewsAPI free tier has 100 req/day limit — we can't afford to always fetch it.

### 4. SQLite for Portfolio Persistence
**Decision:** Use SQLite for virtual portfolio storage.  
**Rationale:** Zero-dependency embedded database. Survives Claude Desktop restarts. Simple schema for positions, trades, account.  
**Trade-off:** Not concurrent-safe for multiple simultaneous write requests. For production, use PostgreSQL with proper transaction isolation.

### 5. In-Memory Caching with TTL
**Decision:** Cache live prices in memory with 60-second TTL.  
**Rationale:** yfinance makes HTTP calls to Yahoo servers. Calling it per-message without caching would be slow and potentially rate-limited.  
**Trade-off:** Price data can be up to 60 seconds stale. Acceptable for signals; not acceptable for tick-by-tick scalping.

### 6. MCP stdio Transport
**Decision:** Use stdio transport (not HTTP/SSE).  
**Rationale:** Claude Desktop only supports stdio MCP servers as of 2024. Simpler, no port management needed.  
**Future:** HTTP/SSE transport would enable cloud deployment and multi-client access.

### 7. Async Architecture
**Decision:** All handlers are async. Blocking yfinance calls run in `asyncio.run_in_executor`.  
**Rationale:** MCP server is async by nature. yfinance is synchronous. The executor pattern prevents blocking the event loop while fetching data.

### 8. ATM-Centric Options Chain Filtering
**Decision:** Return only 24 strikes nearest to ATM spot price.  
**Rationale:** Full Nifty chain has 200+ strikes. Sending all to Claude would overwhelm the context window. Traders care most about near-ATM strikes.

---

## Free API Stack Summary

| Purpose | API | Key Required | Daily Limit |
|---------|-----|-------------|-------------|
| Live prices | yfinance | ❌ None | Unlimited |
| Historical OHLC | yfinance | ❌ None | Unlimited |
| Options chain | yfinance | ❌ None | Unlimited |
| News headlines | NewsAPI | ✅ Free tier | 100 req/day |
| Macro indicators | Alpha Vantage | ✅ Free tier | 25 req/day |
| Technical analysis | pandas (manual) | ❌ None | Unlimited |
| Greeks | Black-Scholes (scratch) | ❌ None | Unlimited |
| Portfolio DB | SQLite | ❌ None | Unlimited |

---

## Example Claude Conversations

```
User: What's the max pain for Nifty this expiry?
Claude: [calls detect_unusual_activity("NIFTY")]
→ Returns max pain strike, PCR, OI concentration alerts

User: Find oversold IT stocks
Claude: [calls scan_market({"sector": "IT", "rsi_max": 30})]
→ Returns filtered list with RSI, price, volume data

User: Should I buy HDFCBANK right now?
Claude: [calls generate_signal("HDFCBANK", "1d")]
→ Returns BUY/SELL/HOLD with full component breakdown

User: Buy 10 shares of TCS
Claude: [calls place_virtual_trade("TCS", 10, "BUY")]
→ Executes at live price, sets auto stop-loss and target

User: What's my portfolio P&L?
Claude: [calls get_portfolio_pnl()]
→ Shows all positions with live P&L and risk scores
```

---

## Running Tests

```bash
# All tests (no API keys needed — pure logic tests)
python -m pytest tests/test_core.py -v

# Specific test class
python -m pytest tests/test_core.py::TestBlackScholes -v

# With coverage
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=term-missing
```

**Test coverage includes:**
- 13 Black-Scholes tests: put-call parity, Greeks ranges, IV roundtrip, Nifty realistic values
- RSI/MACD/Bollinger Band correctness
- Max pain calculation
- Symbol normalization
- Headline sentiment classification
- Portfolio risk scoring

---

## Bonus: Cloud Deployment

Deploy on [Railway](https://railway.app) or [Render](https://render.com) free tier for 24/7 availability:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "server.py"]
```

For cloud deployment, switch from stdio to SSE transport (update `server.py` accordingly and configure Claude to connect via URL).

---

## License

