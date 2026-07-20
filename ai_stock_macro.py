# -*- coding: utf-8 -*-
"""
TradeBot Model - Streamlit Dashboard

A comprehensive trading system combining classic technical analysis with
macroeconomic data from FRED (Federal Reserve Economic Data).

Features:
- Technical Indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Macroeconomic Data (GDP, Unemployment, CPI, Fed Funds Rate)
- Regime Detection (Bull/Bear/Chop)
- Trading Signal Generation with Confidence Scoring
- Backtesting with Performance Metrics
- Interactive Visualizations
"""

import warnings

import streamlit as st

# --- Check for required dependencies first ---
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.error(
        """
        ❌ **Missing Required Package: plotly**

        Please install it using:
        ```bash
        pip install plotly
        ```
        Then restart the Streamlit app.
        """
    )
    st.stop()

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from datetime import datetime, timedelta
    from pandas_datareader import data as web

    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    missing = str(e).split("'")[1] if "'" in str(e) else "unknown"
    st.error(
        f"""
        ❌ **Missing Required Package: {missing}**

        Please install all required packages:
        ```bash
        pip install streamlit yfinance pandas numpy plotly pandas-datareader
        ```
        Then restart the Streamlit app.
        """
    )
    st.stop()

warnings.filterwarnings("ignore")

# --- 1. Configuration ---

st.set_page_config(page_title="TradeBot Model", layout="wide")
st.title("🤖 TradeBot Model")
st.markdown("Combining classic technical analysis with macroeconomic data")

# FRED API key used when "Fetch FRED Data" is checked in the sidebar.
# Note: hardcoding a key means it ships with the source file — anyone with
# the code (or the deployed repo) has the key. If this app is ever made
# public or pushed to a shared/public repo, consider moving this to
# st.secrets or an environment variable instead.
FRED_API_KEY = "ca801c73ee34b7fb11b94352484bc07d"


# --- 2. Core TradeBot Engine ---

class TradeBot:
    """
    Advanced trading system combining technical indicators and macroeconomic data.

    Features:
    - Technical indicator calculation (SMA, EMA, RSI, MACD, Bollinger Bands)
    - Macroeconomic data fetching from FRED
    - Market regime detection
    - Signal generation with confidence scoring
    - Backtesting engine
    """

    def __init__(self, ticker, period_days=252):
        self.ticker = ticker
        self.period_days = period_days
        self.data = None
        self.indicators = None
        self.macro_data = None
        self.signals = None
        self.regime = None

    def fetch_stock_data(self):
        """Fetch historical stock data from Yahoo Finance."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.period_days + 30)

        try:
            df = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                return None

            # Newer yfinance versions return MultiIndex columns (Price, Ticker)
            # even for a single ticker. Flatten to plain 1-D columns so that
            # df['Close'] etc. are Series, not DataFrames.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            self.data = df
            return df
        except Exception as e:
            st.error(f"Error fetching data for {self.ticker}: {e}")
            return None

    def fetch_macro_data(self, fred_api_key=None):
        """
        Fetch macroeconomic data from FRED.

        Key indicators:
        - GDP (Gross Domestic Product)
        - UNRATE (Unemployment Rate)
        - CPIAUCSL (Consumer Price Index)
        - FEDFUNDS (Federal Funds Rate)
        - DGS10 (10-Year Treasury Yield)
        - M2SL (M2 Money Supply)
        """
        try:
            indicators = {
                "GDP": "GDP",
                "Unemployment Rate": "UNRATE",
                "CPI": "CPIAUCSL",
                "Fed Funds Rate": "FEDFUNDS",
                "10Y Treasury Yield": "DGS10",
                "M2 Money Supply": "M2SL",
            }

            # Try to fetch with API key if provided, otherwise fall back to simulation
            if fred_api_key:
                try:
                    macro_dict = {}
                    for name, code in indicators.items():
                        series = web.DataReader(
                            code,
                            "fred",
                            start=self.data.index[0] - timedelta(days=30),
                            end=datetime.now(),
                        )
                        macro_dict[name] = series.iloc[:, 0]
                except Exception:
                    st.warning("FRED data fetch failed. Using simulated macro data.")
                    macro_dict = self._simulate_macro_data()
            else:
                st.info("ℹ️ No FRED API key provided. Using simulated macroeconomic data.")
                macro_dict = self._simulate_macro_data()

            self.macro_data = pd.DataFrame(macro_dict)
            return self.macro_data
        except Exception as e:
            st.warning(f"Could not fetch FRED data: {e}")
            self.macro_data = pd.DataFrame(self._simulate_macro_data())
            return self.macro_data

    def _simulate_macro_data(self):
        """Generate simulated macroeconomic data for demonstration."""
        if self.data is None:
            return {}

        dates = self.data.index
        n = len(dates)
        time = np.arange(n)

        # GDP: slightly upward trend with cycles
        gdp = 18000 + 2000 * (time / n) + 1000 * np.sin(time / 50)

        # Unemployment: cycles between ~3.5% and ~6%
        unrate = 4.5 + 1.5 * np.sin(time / 80 + 0.5) + 0.3 * np.random.randn(n)

        # CPI: ~2% annual inflation with monthly variation
        cpi = 250 + 0.5 * (time / n) * 10 + 1.5 * np.cumsum(np.random.randn(n) / 20)

        # Fed Funds Rate: follows economic cycle
        fedfunds = 3 + 2 * np.sin(time / 100 - 1) + 0.5 * np.random.randn(n)

        # 10Y Treasury Yield: ~4% average with variation
        dgs10 = 4 + 1.5 * np.sin(time / 120 + 0.8) + 0.3 * np.random.randn(n)

        # M2 Money Supply: upward with acceleration
        m2 = 15000 + 300 * (time / n) * 20 + 200 * np.cumsum(np.random.randn(n) / 30)

        return {
            "GDP": pd.Series(gdp, index=dates),
            "Unemployment Rate": pd.Series(unrate, index=dates),
            "CPI": pd.Series(cpi, index=dates),
            "Fed Funds Rate": pd.Series(fedfunds, index=dates),
            "10Y Treasury Yield": pd.Series(dgs10, index=dates),
            "M2 Money Supply": pd.Series(m2, index=dates),
        }

    def calculate_technical_indicators(self):
        """
        Calculate comprehensive technical indicators.

        Indicators:
        - SMA (Simple Moving Average): 5, 10, 20, 50, 200
        - EMA (Exponential Moving Average): 5, 10, 20, 50, 200
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        - Volume indicators
        """
        if self.data is None:
            return None

        df = self.data.copy()
        close = df["Close"]

        # --- Moving Averages ---
        for window in [5, 10, 20, 50, 200]:
            df[f"SMA_{window}"] = close.rolling(window).mean()
            df[f"EMA_{window}"] = close.ewm(span=window, adjust=False).mean()

        # --- RSI ---
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # --- MACD ---
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

        # --- Bollinger Bands ---
        df["BB_Middle"] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
        df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std
        df["BB_Position"] = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

        # --- Volume Indicators ---
        df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"]

        # --- Price Action ---
        df["High_Low_Ratio"] = df["High"] / df["Low"]
        df["Close_Open_Ratio"] = df["Close"] / df["Open"]
        df["ATR_14"] = df["High"].rolling(14).max() - df["Low"].rolling(14).min()

        self.indicators = df
        return df

    def detect_regime(self):
        """
        Detect market regime: Bull, Bear, or Chop.

        Uses multiple factors:
        - Price vs SMA_200 (long-term trend)
        - RSI levels (momentum)
        - Volatility (ATR)
        """
        if self.indicators is None:
            return None

        df = self.indicators
        latest = df.iloc[-1]

        bull_score = 0
        bear_score = 0

        # 1. Price vs SMA_200 (primary trend)
        if latest["Close"] > latest["SMA_200"]:
            bull_score += 3
        else:
            bear_score += 3

        # 2. RSI momentum
        if latest["RSI_14"] > 60:
            bull_score += 2
        elif latest["RSI_14"] < 40:
            bear_score += 2

        # 3. Price vs SMA_50 (short-term trend)
        if latest["Close"] > latest["SMA_50"]:
            bull_score += 2
        else:
            bear_score += 2

        # 4. MACD signal
        if latest["MACD"] > latest["MACD_Signal"]:
            bull_score += 1
        else:
            bear_score += 1

        # 5. Volatility regime
        atr_ratio = latest["ATR_14"] / latest["Close"]
        if atr_ratio > 0.03:
            # High volatility - add to both
            bull_score += 1
            bear_score += 1

        # Determine regime
        if bull_score > bear_score + 2:
            regime = "bull"
            confidence = bull_score / (bull_score + bear_score)
        elif bear_score > bull_score + 2:
            regime = "bear"
            confidence = bear_score / (bull_score + bear_score)
        else:
            regime = "chop"
            confidence = 0.5

        self.regime = {
            "regime": regime,
            "confidence": confidence,
            "bull_score": bull_score,
            "bear_score": bear_score,
            "volatility": atr_ratio,
        }

        return self.regime

    def generate_signals(self):
        """Generate trading signals with confidence scoring."""
        if self.indicators is None or self.regime is None:
            return None

        df = self.indicators
        latest = df.iloc[-1]

        # -1: bearish, 0: neutral, 1: bullish
        signals = {
            "sma_crossover": 0,
            "rsi": 0,
            "macd": 0,
            "bb": 0,
            "volume": 0,
            "regime_factor": 0,
        }

        # 1. SMA Crossover (50 vs 200)
        signals["sma_crossover"] = 1 if latest["SMA_50"] > latest["SMA_200"] else -1

        # 2. RSI
        if latest["RSI_14"] < 30:
            signals["rsi"] = 1  # Oversold - bullish
        elif latest["RSI_14"] > 70:
            signals["rsi"] = -1  # Overbought - bearish

        # 3. MACD
        signals["macd"] = 1 if latest["MACD"] > latest["MACD_Signal"] else -1

        # 4. Bollinger Bands
        if latest["BB_Position"] < 0.2:
            signals["bb"] = 1  # Near lower band - bullish
        elif latest["BB_Position"] > 0.8:
            signals["bb"] = -1  # Near upper band - bearish

        # 5. Volume
        if latest["Volume_Ratio"] > 1.5 and latest["Close"] > latest["Open"]:
            signals["volume"] = 1  # Strong volume on up day
        elif latest["Volume_Ratio"] > 1.5 and latest["Close"] < latest["Open"]:
            signals["volume"] = -1  # Strong volume on down day

        # 6. Regime Factor
        if self.regime["regime"] == "bull":
            signals["regime_factor"] = 1
        elif self.regime["regime"] == "bear":
            signals["regime_factor"] = -1

        # Calculate final signal and confidence
        signal_sum = sum(
            [
                signals["sma_crossover"],
                signals["rsi"] * 0.5,
                signals["macd"] * 0.5,
                signals["bb"] * 0.5,
                signals["volume"] * 0.3,
                signals["regime_factor"] * 0.7,
            ]
        )

        max_score = 1 + 0.5 + 0.5 + 0.5 + 0.3 + 0.7  # = 3.5
        normalized_score = signal_sum / max_score
        confidence = abs(normalized_score)

        if normalized_score > 0.2:
            direction = "BUY"
            confidence = min(confidence, 0.95)
        elif normalized_score < -0.2:
            direction = "SELL"
            confidence = min(confidence, 0.95)
        else:
            direction = "HOLD"
            confidence = 0.5

        self.signals = {
            "direction": direction,
            "confidence": confidence,
            "components": signals,
            "weighted_score": normalized_score,
        }

        return self.signals

    def run_backtest(self, initial_capital=10000, trade_size=0.95):
        """
        Run a backtest of the strategy (SMA 50/200 crossover).

        Returns performance metrics and trade history.
        """
        if self.indicators is None:
            return None

        df = self.indicators.copy()
        capital = float(initial_capital)
        position = 0
        trades = []
        portfolio_values = []

        for i in range(50, len(df) - 1):
            current_price = df["Close"].iloc[i]

            # Buy signal: SMA 50 crosses above SMA 200
            if (
                df["SMA_50"].iloc[i] > df["SMA_200"].iloc[i]
                and df["SMA_50"].iloc[i - 1] <= df["SMA_200"].iloc[i - 1]
            ):
                if position == 0:
                    shares = capital * trade_size // current_price
                    position = shares
                    capital -= shares * current_price
                    trades.append(
                        {"date": df.index[i], "action": "BUY", "price": current_price, "shares": shares}
                    )

            # Sell signal: SMA 50 crosses below SMA 200
            elif (
                df["SMA_50"].iloc[i] < df["SMA_200"].iloc[i]
                and df["SMA_50"].iloc[i - 1] >= df["SMA_200"].iloc[i - 1]
            ):
                if position > 0:
                    capital += position * current_price
                    trades.append(
                        {"date": df.index[i], "action": "SELL", "price": current_price, "shares": position}
                    )
                    position = 0

            portfolio_value = capital + position * current_price
            portfolio_values.append(portfolio_value)

        # Close any remaining position
        if position > 0:
            final_price = df["Close"].iloc[-1]
            capital += position * final_price
            trades.append({"date": df.index[-1], "action": "SELL", "price": final_price, "shares": position})

        final_value = capital
        total_return = (final_value - initial_capital) / initial_capital

        # Performance metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        cummax = pd.Series(portfolio_values).expanding().max()
        drawdown = (pd.Series(portfolio_values) - cummax) / cummax
        max_drawdown = drawdown.min() if len(drawdown) else 0

        # Win rate: pair up BUY/SELL trades in order
        winning_trades = 0
        total_trades = len([t for t in trades if t["action"] == "SELL"])
        if total_trades > 0:
            for i in range(0, len(trades) - 1, 2):
                if trades[i]["action"] == "BUY" and trades[i + 1]["action"] == "SELL":
                    if trades[i + 1]["price"] > trades[i]["price"]:
                        winning_trades += 1
            win_rate = winning_trades / total_trades
        else:
            win_rate = 0

        return {
            "initial_capital": initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "portfolio_values": portfolio_values,
            "trades": trades,
        }

    def get_recommendation(self):
        """Get final trading recommendation with justification."""
        if self.signals is None or self.regime is None:
            return None

        signal = self.signals
        regime = self.regime

        recommendation = {
            "action": signal["direction"],
            "confidence": signal["confidence"],
            "regime": regime["regime"],
            "regime_confidence": regime["confidence"],
        }

        components = signal["components"]
        justifications = []

        if components["sma_crossover"] == 1:
            justifications.append("SMA 50 > SMA 200 (bullish trend)")
        elif components["sma_crossover"] == -1:
            justifications.append("SMA 50 < SMA 200 (bearish trend)")

        if components["rsi"] == 1:
            justifications.append("RSI oversold (potential reversal)")
        elif components["rsi"] == -1:
            justifications.append("RSI overbought (potential pullback)")

        if components["macd"] == 1:
            justifications.append("MACD above signal line (bullish momentum)")
        elif components["macd"] == -1:
            justifications.append("MACD below signal line (bearish momentum)")

        if components["bb"] == 1:
            justifications.append("Price near lower Bollinger Band (oversold)")
        elif components["bb"] == -1:
            justifications.append("Price near upper Bollinger Band (overbought)")

        if components["volume"] == 1:
            justifications.append("Strong volume confirming upward move")
        elif components["volume"] == -1:
            justifications.append("Strong volume confirming downward move")

        if regime["regime"] == "bull":
            justifications.append(f"Bull market regime detected ({regime['confidence']:.0%} confidence)")
        elif regime["regime"] == "bear":
            justifications.append(f"Bear market regime detected ({regime['confidence']:.0%} confidence)")
        else:
            justifications.append("Choppy market regime - reduce position size")

        recommendation["justifications"] = justifications
        recommendation["regime_info"] = regime

        return recommendation


# --- 3. Streamlit Dashboard ---

with st.sidebar:
    st.header("⚙️ Configuration")

    ticker = st.text_input("Ticker Symbol", value="SPY")
    period_days = st.selectbox("Historical Period (days)", [90, 180, 252, 365, 730], index=2)

    st.markdown("---")
    st.subheader("📊 Data Sources")

    use_fred = st.checkbox("Fetch FRED Data", value=False)
    fred_api_key = FRED_API_KEY if use_fred else None

    st.markdown("---")
    st.subheader("🎯 Strategy Parameters")

    initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)

    if st.button("🚀 Run TradeBot Analysis", type="primary"):
        with st.spinner(f"Analyzing {ticker} with technical and macroeconomic data..."):
            bot = TradeBot(ticker, period_days)

            data = bot.fetch_stock_data()
            if data is None or data.empty:
                st.error("No data found. Please check the ticker symbol.")
                st.stop()

            indicators = bot.calculate_technical_indicators()
            macro_data = bot.fetch_macro_data(fred_api_key)
            regime = bot.detect_regime()
            signals = bot.generate_signals()
            backtest_results = bot.run_backtest(initial_capital)
            recommendation = bot.get_recommendation()

            st.session_state["bot"] = bot
            st.session_state["data"] = data
            st.session_state["indicators"] = indicators
            st.session_state["macro_data"] = macro_data
            st.session_state["regime"] = regime
            st.session_state["signals"] = signals
            st.session_state["backtest_results"] = backtest_results
            st.session_state["recommendation"] = recommendation
            st.session_state["ticker"] = ticker

            st.success("Analysis complete!")

# --- 4. Dashboard Display ---

if "bot" in st.session_state:
    data = st.session_state["data"]
    indicators = st.session_state["indicators"]
    macro_data = st.session_state["macro_data"]
    backtest_results = st.session_state["backtest_results"]
    signals = st.session_state["signals"]
    recommendation = st.session_state["recommendation"]
    ticker = st.session_state["ticker"]

    st.subheader(f"📊 {ticker} Analysis")

    # --- Row 1: Trading Recommendation ---
    st.markdown("### 🎯 Trading Recommendation")

    if recommendation:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Action", recommendation["action"])
        with col2:
            st.metric("Confidence", f"{recommendation['confidence']:.0%}")
        with col3:
            st.metric("Market Regime", recommendation["regime"].upper())
        with col4:
            st.metric("Regime Confidence", f"{recommendation['regime_confidence']:.0%}")

        with st.expander("📝 Justifications"):
            for justification in recommendation["justifications"]:
                st.write(f"• {justification}")

    # --- Row 2: Key Metrics ---
    st.markdown("### 📈 Key Performance Metrics")

    if backtest_results:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Return", f"{backtest_results['total_return']:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:.2%}")
        with col4:
            st.metric("Total Trades", backtest_results["total_trades"])
        with col5:
            st.metric("Win Rate", f"{backtest_results['win_rate']:.0%}")

    # --- Row 3: Price Chart with Indicators ---
    st.markdown("### 📉 Price Chart with Technical Indicators")

    if indicators is not None:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Price & Moving Averages", "RSI", "MACD"),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.25, 0.25],
        )

        fig.add_trace(
            go.Candlestick(
                x=indicators.index[-100:],
                open=indicators["Open"].iloc[-100:],
                high=indicators["High"].iloc[-100:],
                low=indicators["Low"].iloc[-100:],
                close=indicators["Close"].iloc[-100:],
                name="Price",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=indicators.index[-100:],
                y=indicators["SMA_20"].iloc[-100:],
                mode="lines",
                name="SMA 20",
                line=dict(color="blue", width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=indicators.index[-100:],
                y=indicators["SMA_50"].iloc[-100:],
                mode="lines",
                name="SMA 50",
                line=dict(color="orange", width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=indicators.index[-100:],
                y=indicators["SMA_200"].iloc[-100:],
                mode="lines",
                name="SMA 200",
                line=dict(color="red", width=1),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=indicators.index[-100:],
                y=indicators["RSI_14"].iloc[-100:],
                mode="lines",
                name="RSI 14",
                line=dict(color="purple"),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.add_trace(
            go.Scatter(
                x=indicators.index[-100:],
                y=indicators["MACD"].iloc[-100:],
                mode="lines",
                name="MACD",
                line=dict(color="blue"),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=indicators.index[-100:],
                y=indicators["MACD_Signal"].iloc[-100:],
                mode="lines",
                name="MACD Signal",
                line=dict(color="red"),
            ),
            row=3,
            col=1,
        )

        fig.update_layout(height=700, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- Row 4: Macroeconomic Data ---
    st.markdown("### 🌍 Macroeconomic Data")

    if macro_data is not None and not macro_data.empty:
        macro_latest = macro_data.tail(10)

        fig = make_subplots(rows=2, cols=3, subplot_titles=list(macro_data.columns[:6]))

        for i, col in enumerate(macro_data.columns[:6]):
            row = i // 3 + 1
            col_num = i % 3 + 1
            fig.add_trace(
                go.Scatter(x=macro_data.index[-50:], y=macro_data[col].iloc[-50:], mode="lines", name=col),
                row=row,
                col=col_num,
            )

        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Macroeconomic Data Table"):
            st.dataframe(macro_latest.style.format("{:.2f}"))

    # --- Row 5: Portfolio Performance ---
    st.markdown("### 💰 Portfolio Performance")

    if backtest_results and backtest_results["portfolio_values"]:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=backtest_results["portfolio_values"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="blue", width=2),
            )
        )
        fig.add_hline(
            y=backtest_results["initial_capital"],
            line_dash="dash",
            line_color="green",
            annotation_text="Initial Capital",
        )
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Trading Days",
            yaxis_title="Portfolio Value ($)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Row 6: Trade History ---
    st.markdown("### 📝 Trade History")

    if backtest_results and backtest_results["trades"]:
        trades_df = pd.DataFrame(backtest_results["trades"])
        if not trades_df.empty:
            trades_df["price"] = trades_df["price"].round(2)
            st.dataframe(trades_df)

    # --- Row 7: Signal Components ---
    st.markdown("### 📊 Signal Components")

    if signals:
        col1, col2 = st.columns(2)

        with col1:
            st.write("Signal Components")
            components_df = pd.DataFrame(
                {
                    "Component": [
                        "SMA Crossover",
                        "RSI",
                        "MACD",
                        "Bollinger Bands",
                        "Volume",
                        "Regime Factor",
                    ],
                    "Value": [
                        signals["components"]["sma_crossover"],
                        signals["components"]["rsi"],
                        signals["components"]["macd"],
                        signals["components"]["bb"],
                        signals["components"]["volume"],
                        signals["components"]["regime_factor"],
                    ],
                }
            )
            st.dataframe(components_df, hide_index=True)

        with col2:
            st.write("Signal Summary")
            st.info(
                f"""
                Direction: {signals['direction']}

                Confidence: {signals['confidence']:.0%}

                Weighted Score: {signals['weighted_score']:.3f}
                """
            )
else:
    st.info("👈 Configure the parameters in the sidebar and click 'Run TradeBot Analysis' to start.")
