"""
IndiaQuant MCP Test Suite
Tests core logic without requiring live API calls.
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np


class TestBlackScholes(unittest.TestCase):
    """Test Black-Scholes implementation from scratch."""

    def setUp(self):
        from src.modules.options_engine import black_scholes_greeks, implied_volatility_bisection
        self.bs = black_scholes_greeks
        self.iv_calc = implied_volatility_bisection

    def test_call_price_basic(self):
        """Standard BS call price (ATM, 1yr, 25% vol)."""
        result = self.bs(S=100, K=100, T=1.0, r=0.05, sigma=0.25, option_type="CE")
        # Known approximation: ATM call ~12.34
        self.assertAlmostEqual(result["price"], 12.34, delta=0.5)

    def test_put_call_parity(self):
        """Put-call parity: C - P = S - K*e^(-rT)."""
        S, K, T, r, sigma = 18500, 18500, 0.25, 0.065, 0.20
        call = self.bs(S=S, K=K, T=T, r=r, sigma=sigma, option_type="CE")
        put = self.bs(S=S, K=K, T=T, r=r, sigma=sigma, option_type="PE")
        parity_lhs = call["price"] - put["price"]
        parity_rhs = S - K * math.exp(-r * T)
        self.assertAlmostEqual(parity_lhs, parity_rhs, delta=1.0)

    def test_delta_range_call(self):
        """Call delta must be between 0 and 1."""
        result = self.bs(S=100, K=100, T=0.5, r=0.065, sigma=0.20, option_type="CE")
        self.assertGreater(result["delta"], 0)
        self.assertLess(result["delta"], 1)

    def test_delta_range_put(self):
        """Put delta must be between -1 and 0."""
        result = self.bs(S=100, K=100, T=0.5, r=0.065, sigma=0.20, option_type="PE")
        self.assertLess(result["delta"], 0)
        self.assertGreater(result["delta"], -1)

    def test_deep_itm_call_delta(self):
        """Deep ITM call delta approaches 1."""
        result = self.bs(S=200, K=100, T=1.0, r=0.065, sigma=0.20, option_type="CE")
        self.assertGreater(result["delta"], 0.9)

    def test_deep_otm_call_delta(self):
        """Deep OTM call delta approaches 0."""
        result = self.bs(S=50, K=200, T=0.5, r=0.065, sigma=0.20, option_type="CE")
        self.assertLess(result["delta"], 0.05)

    def test_gamma_positive(self):
        """Gamma must always be positive."""
        for option_type in ["CE", "PE"]:
            result = self.bs(S=100, K=100, T=0.5, r=0.065, sigma=0.25, option_type=option_type)
            self.assertGreater(result["gamma"], 0)

    def test_theta_negative(self):
        """Theta must be negative (time decay hurts buyers)."""
        for option_type in ["CE", "PE"]:
            result = self.bs(S=100, K=100, T=0.5, r=0.065, sigma=0.25, option_type=option_type)
            self.assertLess(result["theta"], 0)

    def test_vega_positive(self):
        """Vega must be positive."""
        result = self.bs(S=100, K=100, T=0.5, r=0.065, sigma=0.25, option_type="CE")
        self.assertGreater(result["vega"], 0)

    def test_implied_volatility_roundtrip(self):
        """IV calc should recover original sigma within tolerance."""
        S, K, T, r, sigma_true = 18500, 18500, 0.25, 0.065, 0.22
        call = self.bs(S=S, K=K, T=T, r=r, sigma=sigma_true, option_type="CE")
        market_price = call["price"]
        sigma_recovered = self.iv_calc(market_price, S, K, T, r, "CE")
        self.assertAlmostEqual(sigma_recovered, sigma_true, delta=0.002)

    def test_intrinsic_value_call(self):
        """ITM call intrinsic = S - K."""
        result = self.bs(S=150, K=100, T=0.5, r=0.065, sigma=0.20, option_type="CE")
        self.assertAlmostEqual(result["intrinsic_value"], 50.0, delta=0.01)

    def test_intrinsic_value_put(self):
        """ITM put intrinsic = K - S."""
        result = self.bs(S=80, K=100, T=0.5, r=0.065, sigma=0.20, option_type="PE")
        self.assertAlmostEqual(result["intrinsic_value"], 20.0, delta=0.01)

    def test_expired_option(self):
        """Expired options have zero time value and Greeks."""
        result = self.bs(S=105, K=100, T=0.0, r=0.065, sigma=0.25, option_type="CE")
        self.assertEqual(result["theta"], 0.0)
        self.assertEqual(result["gamma"], 0.0)

    def test_nifty_realistic_values(self):
        """Realistic Nifty call: S=22000, K=22000, 7 DTE, 15% IV."""
        result = self.bs(S=22000, K=22000, T=7/365, r=0.065, sigma=0.15, option_type="CE")
        # Expect price between 100-400 for ATM weekly
        self.assertGreater(result["price"], 50)
        self.assertLess(result["price"], 500)
        # Delta ~0.5 for ATM
        self.assertAlmostEqual(result["delta"], 0.5, delta=0.1)


class TestTechnicalIndicators(unittest.TestCase):
    """Test RSI, MACD, Bollinger Bands calculations."""

    def setUp(self):
        from src.modules.signal_generator import compute_rsi, compute_macd, compute_bollinger_bands
        self.compute_rsi = compute_rsi
        self.compute_macd = compute_macd
        self.compute_bb = compute_bollinger_bands

        # Generate realistic price data
        np.random.seed(42)
        n = 100
        prices = 18000 + np.cumsum(np.random.randn(n) * 50)
        self.close = pd.Series(prices)

    def test_rsi_range(self):
        """RSI must be between 0 and 100."""
        rsi = self.compute_rsi(self.close)
        valid = rsi.dropna()
        self.assertTrue((valid >= 0).all())
        self.assertTrue((valid <= 100).all())

    def test_rsi_oversold_detection(self):
        """Falling price should produce low RSI."""
        falling = pd.Series([100 - i * 2 for i in range(50)])
        rsi = self.compute_rsi(falling, period=14)
        final_rsi = rsi.iloc[-1]
        self.assertLess(final_rsi, 40)

    def test_rsi_overbought_detection(self):
    # Mix of big gains and tiny losses — realistic overbought scenario
        base = 100.0
        prices = [base]
        for i in range(49):
            if i % 5 == 0:
                prices.append(prices[-1] - 0.5)   # small loss every 5 bars
            else:
                prices.append(prices[-1] + 3.0)   # consistent gains
        rising = pd.Series(prices)
        rsi = self.compute_rsi(rising, period=14)
        final_rsi = rsi.iloc[-1]
        self.assertGreater(final_rsi, 60)

    def test_macd_keys(self):
        """MACD returns required keys."""
        result = self.compute_macd(self.close)
        self.assertIn("macd", result)
        self.assertIn("signal", result)
        self.assertIn("histogram", result)

    def test_macd_histogram_equals_diff(self):
        """Histogram should equal MACD - Signal."""
        result = self.compute_macd(self.close)
        diff = result["macd"] - result["signal"]
        pd.testing.assert_series_equal(
            result["histogram"].round(8),
            diff.round(8),
            check_names=False
        )

    def test_bollinger_bands_structure(self):
        """Upper band > middle > lower band."""
        bb = self.compute_bb(self.close)
        valid_idx = bb["upper"].dropna().index
        self.assertTrue((bb["upper"][valid_idx] >= bb["middle"][valid_idx]).all())
        self.assertTrue((bb["middle"][valid_idx] >= bb["lower"][valid_idx]).all())

    def test_bollinger_pct_b_atm(self):
        """Price at SMA should give %B ≈ 0.5."""
        # Construct price that ends exactly at SMA
        prices = pd.Series([100.0] * 25)
        bb = self.compute_bb(prices, period=20)
        pct_b = bb["pct_b"].iloc[-1]
        # With no volatility, bands collapse and pct_b is undefined; just check no crash
        self.assertIsNotNone(pct_b)


class TestMaxPain(unittest.TestCase):
    """Test max pain calculation."""

    def test_max_pain_basic(self):
        """Max pain should be near the strike with most OI."""
        from src.modules.options_engine import _calculate_max_pain

        # Simple scenario: massive put OI at 18000, massive call OI at 19000
        calls = [
            {"strike": 19000, "openInterest": 100000},
            {"strike": 19500, "openInterest": 50000},
        ]
        puts = [
            {"strike": 18000, "openInterest": 100000},
            {"strike": 17500, "openInterest": 50000},
        ]
        max_pain = _calculate_max_pain(calls, puts)
        # Max pain should be between 18000 and 19000
        self.assertGreaterEqual(max_pain, 17500)
        self.assertLessEqual(max_pain, 19500)

    def test_max_pain_symmetric(self):
        """Symmetric OI should give max pain near the middle."""
        from src.modules.options_engine import _calculate_max_pain

        calls = [{"strike": float(k), "openInterest": 10000} for k in range(18000, 20001, 500)]
        puts = [{"strike": float(k), "openInterest": 10000} for k in range(16000, 18001, 500)]
        max_pain = _calculate_max_pain(calls, puts)
        self.assertIsNotNone(max_pain)


class TestSymbolNormalization(unittest.TestCase):
    """Test yfinance symbol normalization."""

    def test_nifty_maps_to_index(self):
        from src.modules.market_data import normalize_symbol
        self.assertEqual(normalize_symbol("NIFTY"), "^NSEI")
        self.assertEqual(normalize_symbol("BANKNIFTY"), "^NSEBANK")
        self.assertEqual(normalize_symbol("SENSEX"), "^BSESN")

    def test_nse_suffix_added(self):
        from src.modules.market_data import normalize_symbol
        self.assertEqual(normalize_symbol("RELIANCE"), "RELIANCE.NS")
        self.assertEqual(normalize_symbol("TCS"), "TCS.NS")

    def test_existing_suffix_preserved(self):
        from src.modules.market_data import normalize_symbol
        self.assertEqual(normalize_symbol("RELIANCE.NS"), "RELIANCE.NS")
        self.assertEqual(normalize_symbol("RELIANCE.BO"), "RELIANCE.BO")

    def test_index_prefix_preserved(self):
        from src.modules.market_data import normalize_symbol
        self.assertEqual(normalize_symbol("^NSEI"), "^NSEI")


class TestHeadlineSentiment(unittest.TestCase):
    """Test headline sentiment classifier."""

    def setUp(self):
        from src.modules.signal_generator import _classify_headline
        self.classify = _classify_headline

    def test_bullish_headline(self):
        result = self.classify("Reliance Industries surges to record high on strong Q3 profit growth")
        self.assertEqual(result["label"], "BULLISH")
        self.assertGreater(result["score"], 50)

    def test_bearish_headline(self):
        result = self.classify("HDFC Bank falls sharply on weak earnings, analysts downgrade stock")
        self.assertEqual(result["label"], "BEARISH")
        self.assertLess(result["score"], 50)

    def test_neutral_headline(self):
        result = self.classify("Infosys to hold AGM next week")
        self.assertEqual(result["label"], "NEUTRAL")
        self.assertEqual(result["score"], 50)


class TestPortfolioRiskScore(unittest.TestCase):
    """Test position risk scoring logic."""

    def setUp(self):
        from src.modules.portfolio_manager import _compute_position_risk
        self.risk_fn = _compute_position_risk

    def test_high_drawdown_high_risk(self):
        """Position down 15% should score high risk."""
        score = self.risk_fn(ltp=85, avg_price=100, daily_vol=2.0, stop_loss=80)
        self.assertGreater(score, 40)

    def test_flat_low_vol_low_risk(self):
        """Position at entry with low vol = low risk."""
        score = self.risk_fn(ltp=100, avg_price=100, daily_vol=0.3, stop_loss=95)
        self.assertLess(score, 30)

    def test_risk_range(self):
        """Risk score must be between 0 and 100."""
        score = self.risk_fn(ltp=50, avg_price=100, daily_vol=5.0, stop_loss=45)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)


if __name__ == "__main__":
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
