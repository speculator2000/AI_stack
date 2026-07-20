# -*- coding: utf-8 -*-
"""
Risk Management Model - Streamlit Dashboard

A comprehensive risk management system for trading portfolios featuring:
- Value at Risk (VaR) calculations
- Expected Shortfall (CVaR)
- Portfolio optimization (Markowitz)
- Monte Carlo simulations
- Stress testing
- Risk factor analysis
- Maximum Drawdown analysis
- Sharpe and Sortino ratios
- Correlation matrix with heatmaps
- Scenario analysis

IMPORTANT: This is a comprehensive simulation for educational and professional use.

Target runtime: Python 3.13.x
"""

import sys

if sys.version_info < (3, 13):
    raise RuntimeError(
        f"This app targets Python 3.13.x. Detected {sys.version.split()[0]}. "
        "Create a fresh virtualenv with Python 3.13 (e.g. `python3.13 -m venv .venv`) "
        "and reinstall dependencies from requirements.txt."
    )

import streamlit as st

# --- CHECK FOR REQUIRED DEPENDENCIES FIRST ---
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.error("""
    ❌ **Missing Required Package: plotly**

    Please install it using:
    ```bash
    pip install plotly
    ```

    Then restart the Streamlit app.
    """)
    st.stop()

try:
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import yfinance as yf
    from scipy import stats
    from scipy.optimize import minimize
    import warnings

    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    st.error(f"""
    ❌ **Missing Required Package: {str(e).split("'")[1] if "'" in str(e) else "unknown"}**

    Please install all required packages:
    ```bash
    pip install streamlit yfinance pandas numpy plotly scipy
    ```

    Then restart the Streamlit app.
    """)
    st.stop()

warnings.filterwarnings('ignore')

# --- 1. Configuration ---
st.set_page_config(page_title="Risk Management Model", page_icon="🛡️", layout="wide")


# =============================================================================
# DESIGN SYSTEM
# -----------------------------------------------------------------------------
# A private-research-desk aesthetic: ink navy + ledger ivory, a deep emerald
# and antique gold as the two accents, set in a serif display face (Fraunces)
# over a quiet grotesque body face (Inter), with figures in a monospace
# (IBM Plex Mono) so numbers line up the way they would on a real risk memo.
# =============================================================================

PALETTE = {
    "ink": "#0E1A2B",        # near-black navy — sidebar, headings
    "ink_2": "#1B2F49",      # secondary ink surface
    "paper": "#F7F4EC",      # warm ivory — page background
    "paper_2": "#EFEADA",    # card / metric surface
    "rule": "rgba(14,26,43,0.14)",   # hairline dividers
    "text": "#1B2430",       # body text on paper
    "muted": "#5B6472",      # secondary text
    "paper_text": "#EEEAE0", # text on ink surfaces
    "emerald": "#1F4D3D",    # primary accent — gains, confidence
    "emerald_soft": "rgba(31,77,61,0.10)",
    "gold": "#B08D57",       # secondary accent — highlights, rules
    "gold_soft": "rgba(176,141,87,0.14)",
    "burgundy": "#7A2E2E",   # risk / loss accent
    "burgundy_soft": "rgba(122,46,46,0.10)",
}

PLOTLY_COLORWAY = [
    PALETTE["emerald"], PALETTE["gold"], PALETTE["burgundy"],
    "#3F6B57", "#8C6E4A", "#4A5A73",
]


def inject_design_system():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        .stApp {{
            background: {PALETTE["paper"]};
            color: {PALETTE["text"]};
        }}

        /* ---------- Typography ---------- */
        h1, h2, h3, h4 {{
            font-family: 'Fraunces', serif !important;
            color: {PALETTE["ink"]} !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
        }}
        h3 {{
            border-bottom: 1px solid {PALETTE["rule"]};
            padding-bottom: 0.5rem;
            margin-top: 2.2rem !important;
        }}
        .eyebrow {{
            display: block;
            font-family: 'Inter', sans-serif;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: {PALETTE["gold"]};
            margin-bottom: 0.15rem;
        }}

        /* ---------- Masthead ---------- */
        .masthead {{
            border-top: 3px solid {PALETTE["ink"]};
            border-bottom: 1px solid {PALETTE["rule"]};
            padding: 0.9rem 0 1.1rem 0;
            margin-bottom: 1.6rem;
        }}
        .masthead .eyebrow {{ margin-bottom: 0.35rem; }}
        .masthead h1 {{
            font-size: 2.1rem !important;
            margin: 0 !important;
            line-height: 1.15;
        }}
        .masthead .dek {{
            font-family: 'Inter', sans-serif;
            color: {PALETTE["muted"]};
            font-size: 0.95rem;
            margin-top: 0.35rem;
        }}
        .masthead .meta {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            color: {PALETTE["muted"]};
            letter-spacing: 0.02em;
            margin-top: 0.6rem;
        }}

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {{
            background: {PALETTE["ink"]};
        }}
        [data-testid="stSidebar"] * {{
            color: {PALETTE["paper_text"]} !important;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            font-family: 'Fraunces', serif !important;
            color: {PALETTE["paper_text"]} !important;
            border-bottom: 1px solid rgba(238,234,224,0.18);
            padding-bottom: 0.4rem;
        }}
        [data-testid="stSidebar"] .eyebrow {{ color: {PALETTE["gold"]}; }}
        [data-testid="stSidebar"] hr {{ border-color: rgba(238,234,224,0.16) !important; }}
        [data-testid="stSidebar"] label {{ color: {PALETTE["paper_text"]} !important; opacity: 0.85; }}

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: {PALETTE["ink_2"]} !important;
            border: 1px solid rgba(238,234,224,0.20) !important;
            color: {PALETTE["paper_text"]} !important;
            border-radius: 4px !important;
        }}

        /* ---------- Buttons ---------- */
        .stButton > button, button[kind="primary"] {{
            background: {PALETTE["emerald"]} !important;
            color: {PALETTE["paper_text"]} !important;
            border: 1px solid {PALETTE["emerald"]} !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            font-size: 0.78rem !important;
            padding: 0.55rem 1.1rem !important;
            transition: all 0.15s ease-in-out;
        }}
        .stButton > button:hover, button[kind="primary"]:hover {{
            background: {PALETTE["ink"]} !important;
            border-color: {PALETTE["gold"]} !important;
            color: {PALETTE["gold"]} !important;
        }}

        /* ---------- Metrics ---------- */
        [data-testid="stMetric"] {{
            background: {PALETTE["paper_2"]};
            border: 1px solid {PALETTE["rule"]};
            border-radius: 6px;
            padding: 0.9rem 1rem 0.7rem 1rem;
        }}
        [data-testid="stMetricLabel"] {{
            font-family: 'Inter', sans-serif !important;
            font-size: 0.72rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {PALETTE["muted"]} !important;
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'IBM Plex Mono', monospace !important;
            color: {PALETTE["ink"]} !important;
            font-weight: 600 !important;
        }}

        /* ---------- Dataframes & expanders ---------- */
        [data-testid="stDataFrame"] {{
            border: 1px solid {PALETTE["rule"]};
            border-radius: 6px;
            overflow: hidden;
        }}
        [data-testid="stExpander"] {{
            border: 1px solid {PALETTE["rule"]} !important;
            border-radius: 6px !important;
            background: {PALETTE["paper_2"]};
        }}

        /* ---------- Alert boxes retinted to the palette ---------- */
        [data-testid="stAlertContentInfo"] {{ color: {PALETTE["ink"]} !important; }}
        div[data-baseweb="notification"] {{ border-radius: 6px !important; }}
        .stAlert {{ border-radius: 6px !important; }}

        /* ---------- Rules ---------- */
        hr {{ border-color: {PALETTE["rule"]} !important; }}

        /* ---------- Section header block ---------- */
        .section-head h3 {{ margin-top: 0 !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def masthead(eyebrow, title, dek, meta):
    st.markdown(
        f"""
        <div class="masthead">
            <span class="eyebrow">{eyebrow}</span>
            <h1>{title}</h1>
            <div class="dek">{dek}</div>
            <div class="meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(eyebrow, title):
    st.markdown(
        f"""
        <div class="section-head">
            <span class="eyebrow">{eyebrow}</span>
            <h3>{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def themed_layout(fig, height=None, title=None):
    """Apply the house plotly theme to a figure in place, and return it."""
    layout_kwargs = dict(
        paper_bgcolor=PALETTE["paper"],
        plot_bgcolor=PALETTE["paper"],
        font=dict(family="Inter, sans-serif", color=PALETTE["text"], size=13),
        colorway=PLOTLY_COLORWAY,
        margin=dict(t=54 if title else 24, l=10, r=10, b=10),
        legend=dict(font=dict(family="Inter, sans-serif", size=12)),
    )
    if title:
        layout_kwargs["title"] = dict(
            text=title, font=dict(family="Fraunces, serif", size=17, color=PALETTE["ink"])
        )
    if height:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor="rgba(14,26,43,0.08)", zerolinecolor="rgba(14,26,43,0.15)", linecolor=PALETTE["rule"])
    fig.update_yaxes(gridcolor="rgba(14,26,43,0.08)", zerolinecolor="rgba(14,26,43,0.15)", linecolor=PALETTE["rule"])
    return fig


def _clean_close_series(df, label):
    """
    Return a guaranteed 1-D pd.Series of Close prices from a yfinance
    download result, regardless of whether yfinance returned single-level
    or MultiIndex (Price, Ticker) columns for this particular ticker/version.

    Without this normalization, a dict of mixed scalar/odd-shaped values
    passed to pd.DataFrame(...) can raise:
    "ValueError: If using all scalar values, you must pass an index"
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    close = df['Close']
    if isinstance(close, pd.DataFrame):
        # Single-column DataFrame (e.g. leftover Ticker level) -> squeeze to Series
        close = close.iloc[:, 0]
    close = close.dropna()
    close.name = label
    return close


# --- 2. Core Risk Management Engine ---
class RiskManager:
    """Comprehensive risk management engine with advanced analytics."""

    def __init__(self, tickers, weights=None):
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.weights = weights if weights is not None else np.ones(len(self.tickers)) / len(self.tickers)
        self.data = None
        self.returns = None
        self.cov_matrix = None
        self.corr_matrix = None
        self.metrics = {}

    def fetch_data(self, period=252, start_date=None):
        """Fetch historical data for all tickers."""
        if start_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period + 30)
        else:
            end_date = datetime.now()

        data = {}
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if df.empty:
                    continue

                close = _clean_close_series(df, ticker)
                if close.empty:
                    continue

                data[ticker] = close
            except Exception:
                st.warning(f"Could not fetch data for {ticker}")

        if not data:
            return None

        self.data = pd.DataFrame(data)
        self.data = self.data.dropna()
        self.returns = self.data.pct_change().dropna()
        self.cov_matrix = self.returns.cov() * 252  # Annualized
        self.corr_matrix = self.returns.corr()

        return self.data

    def calculate_var_historical(self, confidence_level=0.95, horizon=1):
        """Calculate Value at Risk using historical method."""
        if self.returns is None:
            return None

        portfolio_returns = self._calculate_portfolio_returns()
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100) * np.sqrt(horizon)
        return var

    def calculate_var_parametric(self, confidence_level=0.95, horizon=1):
        """Calculate Value at Risk using parametric (variance-covariance) method."""
        if self.returns is None:
            return None

        portfolio_returns = self._calculate_portfolio_returns()
        mean = portfolio_returns.mean()
        std = portfolio_returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        var = (mean + z_score * std) * np.sqrt(horizon)
        return var

    def calculate_var_monte_carlo(self, confidence_level=0.95, horizon=1, n_simulations=10000):
        """Calculate Value at Risk using Monte Carlo simulation."""
        if self.returns is None:
            return None

        mean_returns = self.returns.mean()
        cov_matrix = self.returns.cov()

        # Generate correlated random returns
        simulated_returns = np.random.multivariate_normal(
            mean_returns,
            cov_matrix,
            n_simulations
        )

        # Calculate portfolio returns for each simulation
        portfolio_returns = simulated_returns @ self.weights
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100) * np.sqrt(horizon)

        return var

    def calculate_cvar(self, confidence_level=0.95):
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if self.returns is None:
            return None

        portfolio_returns = self._calculate_portfolio_returns()
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return cvar

    def calculate_max_drawdown(self):
        """Calculate Maximum Drawdown."""
        if self.data is None:
            return None

        portfolio_values = self._calculate_portfolio_values()
        cumulative_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        current_max = portfolio_values.iloc[0]

        for idx, value in portfolio_values.items():
            if value > current_max:
                current_max = value
                if in_drawdown:
                    drawdown_periods.append({
                        'start': start_date,
                        'end': idx,
                        'recovery': True
                    })
                    in_drawdown = False
            elif not in_drawdown:
                in_drawdown = True
                start_date = idx

        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': drawdown.iloc[-1] if not drawdown.empty else 0,
            'drawdown_series': drawdown,
            'periods': drawdown_periods
        }

    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe Ratio (annualized)."""
        if self.returns is None:
            return None

        portfolio_returns = self._calculate_portfolio_returns()
        excess_returns = portfolio_returns - risk_free_rate / 252
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe

    def calculate_sortino_ratio(self, risk_free_rate=0.02):
        """Calculate Sortino Ratio (downside risk only)."""
        if self.returns is None:
            return None

        portfolio_returns = self._calculate_portfolio_returns()
        excess_returns = portfolio_returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 1
        sortino = excess_returns.mean() / downside_std * np.sqrt(252)
        return sortino

    def calculate_calmar_ratio(self):
        """Calculate Calmar Ratio (return / max drawdown)."""
        if self.returns is None:
            return None

        portfolio_returns = self._calculate_portfolio_returns()
        annual_return = portfolio_returns.mean() * 252
        max_dd = self.calculate_max_drawdown()
        if max_dd and max_dd['max_drawdown'] != 0:
            calmar = annual_return / abs(max_dd['max_drawdown'])
        else:
            calmar = 0
        return calmar

    def optimize_portfolio(self, method='sharpe', risk_free_rate=0.02):
        """Optimize portfolio weights using Markowitz optimization."""
        if self.returns is None:
            return None

        mean_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252

        def portfolio_stats(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_return, portfolio_volatility

        def negative_sharpe(weights):
            ret, vol = portfolio_stats(weights)
            sharpe = (ret - risk_free_rate) / vol if vol > 0 else -np.inf
            return -sharpe

        def portfolio_variance(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.tickers)))
        initial_weights = np.array([1 / len(self.tickers)] * len(self.tickers))

        if method == 'sharpe':
            result = minimize(negative_sharpe, initial_weights,
                              method='SLSQP', bounds=bounds, constraints=constraints)
        elif method == 'min_variance':
            result = minimize(portfolio_variance, initial_weights,
                              method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            return None

        if result.success:
            optimized_weights = result.x
            ret, vol = portfolio_stats(optimized_weights)
            return {
                'weights': optimized_weights,
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': (ret - risk_free_rate) / vol if vol > 0 else 0
            }
        return None

    def stress_test(self, scenario='market_crash'):
        """Perform stress testing under various scenarios."""
        if self.returns is None:
            return None

        portfolio_returns = self._calculate_portfolio_returns()

        scenarios = {
            'market_crash': -0.15,  # 15% decline
            'volatility_shock': 0.25,  # 25% increase in vol
            'interest_rate_hike': -0.05,  # 5% decline
            'flash_crash': -0.30,  # 30% decline
            'normal_conditions': 0.01  # normal day
        }

        stress_results = {}

        if scenario == 'market_crash':
            stress_results = {
                'shock_return': -0.15,
                'portfolio_loss': -0.15 * portfolio_returns.std() * 10,
                'var_shock': self.calculate_var_historical(0.99) * 1.5,
                'impact': 'Severe'
            }
        elif scenario == 'volatility_shock':
            stress_results = {
                'shock_return': 0,
                'portfolio_loss': portfolio_returns.std() * 0.25 * 2,
                'var_shock': self.calculate_var_historical(0.95) * 2,
                'impact': 'High'
            }
        elif scenario == 'flash_crash':
            stress_results = {
                'shock_return': -0.30,
                'portfolio_loss': -0.30 * portfolio_returns.std() * 15,
                'var_shock': self.calculate_var_historical(0.99) * 2.5,
                'impact': 'Extreme'
            }
        else:
            stress_results = {
                'shock_return': scenarios[scenario],
                'portfolio_loss': scenarios[scenario] * portfolio_returns.std() * 5,
                'var_shock': self.calculate_var_historical(0.95),
                'impact': 'Low'
            }

        return stress_results

    def calculate_beta(self, market_ticker='SPY'):
        """Calculate portfolio beta relative to market."""
        if self.data is None:
            return None

        try:
            market_data = yf.download(market_ticker, start=self.data.index[0], end=datetime.now(), progress=False)
            market_close = _clean_close_series(market_data, market_ticker)
            market_returns = market_close.pct_change().dropna()

            # Align dates
            common_dates = self.returns.index.intersection(market_returns.index)
            portfolio_returns = self._calculate_portfolio_returns().loc[common_dates]
            market_returns_aligned = market_returns.loc[common_dates]

            covariance = np.cov(portfolio_returns, market_returns_aligned)[0, 1]
            variance = np.var(market_returns_aligned)
            beta = covariance / variance if variance > 0 else 1

            return beta
        except Exception:
            return 1

    def calculate_risk_contribution(self):
        """Calculate each asset's contribution to total portfolio risk."""
        if self.returns is None:
            return None

        portfolio_volatility = np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))

        contributions = {}
        for i, ticker in enumerate(self.tickers):
            marginal_risk = np.dot(self.cov_matrix, self.weights)[i] / portfolio_volatility
            risk_contribution = self.weights[i] * marginal_risk
            contributions[ticker] = risk_contribution / portfolio_volatility * 100

        return contributions

    def generate_report(self):
        """Generate comprehensive risk report."""
        if self.data is None:
            return None

        report = {}

        # Basic metrics
        report['total_return'] = (
                    self._calculate_portfolio_values().iloc[-1] / self._calculate_portfolio_values().iloc[0] - 1)
        report['annualized_return'] = report['total_return'] * (252 / len(self.data))
        report['volatility'] = self._calculate_portfolio_returns().std() * np.sqrt(252)

        # Risk metrics
        report['var_95_historical'] = self.calculate_var_historical(0.95)
        report['var_95_parametric'] = self.calculate_var_parametric(0.95)
        report['var_95_monte_carlo'] = self.calculate_var_monte_carlo(0.95)
        report['var_99_historical'] = self.calculate_var_historical(0.99)
        report['cvar_95'] = self.calculate_cvar(0.95)

        # Drawdown
        dd = self.calculate_max_drawdown()
        report['max_drawdown'] = dd['max_drawdown'] if dd else 0
        report['current_drawdown'] = dd['current_drawdown'] if dd else 0

        # Ratios
        report['sharpe_ratio'] = self.calculate_sharpe_ratio()
        report['sortino_ratio'] = self.calculate_sortino_ratio()
        report['calmar_ratio'] = self.calculate_calmar_ratio()

        # Beta
        report['beta'] = self.calculate_beta()

        return report

    def _calculate_portfolio_returns(self):
        """Calculate weighted portfolio returns."""
        if self.returns is None:
            return None
        return self.returns @ self.weights

    def _calculate_portfolio_values(self):
        """Calculate portfolio values over time."""
        if self.data is None:
            return None
        portfolio_returns = self._calculate_portfolio_returns()
        portfolio_values = 10000 * (1 + portfolio_returns).cumprod()
        return portfolio_values

    def run_monte_carlo(self, n_simulations=1000, horizon=252, initial_value=10000):
        """Run Monte Carlo simulation for portfolio value projections."""
        if self.returns is None:
            return None

        mean_returns = self.returns.mean()
        cov_matrix = self.returns.cov()

        # Generate paths
        paths = []
        final_values = []

        for _ in range(n_simulations):
            # Generate returns for the horizon
            simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, horizon)
            portfolio_returns = simulated_returns @ self.weights

            # Calculate path
            # path is a plain NumPy array (portfolio_returns comes from a
            # NumPy matrix multiply, simulated_returns @ self.weights), so
            # it must be indexed with [-1], not pandas' .iloc[-1].
            path = initial_value * (1 + portfolio_returns).cumprod()
            paths.append(path)
            final_values.append(path[-1])

        # Calculate percentiles
        final_values = np.array(final_values)
        percentiles = {
            '5th': np.percentile(final_values, 5),
            '25th': np.percentile(final_values, 25),
            '50th': np.percentile(final_values, 50),
            '75th': np.percentile(final_values, 75),
            '95th': np.percentile(final_values, 95)
        }

        return {
            'paths': paths,
            'final_values': final_values,
            'percentiles': percentiles
        }


# --- 3. Streamlit Dashboard ---

inject_design_system()

masthead(
    eyebrow="Confidential &middot; Portfolio Risk Desk",
    title="Risk Management Model",
    dek="Value at Risk, Expected Shortfall, Markowitz optimization, Monte Carlo projection, and stress testing "
        "for a single portfolio.",
    meta=f"Prepared {datetime.now().strftime('%d %b %Y &middot; %H:%M')} &middot; Simulation for educational and "
         f"professional use",
)

# Sidebar configuration
with st.sidebar:
    st.markdown('<span class="eyebrow">Desk Setup</span>', unsafe_allow_html=True)
    st.header("Portfolio Configuration")

    # Input tickers
    tickers_input = st.text_input(
        "Ticker Symbols (comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN, TSLA"
    )
    tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]

    st.markdown("---")
    st.subheader("Portfolio Weights")

    weight_method = st.selectbox(
        "Weight Allocation Method",
        ["Equal", "Custom", "Optimized (Sharpe)", "Optimized (Min Variance)"]
    )

    weights = None
    if weight_method == "Equal":
        weights = np.ones(len(tickers)) / len(tickers)
    elif weight_method == "Custom":
        weight_inputs = []
        for i, ticker in enumerate(tickers):
            w = st.number_input(f"Weight for {ticker}", 0.0, 1.0, 1.0 / len(tickers), 0.01)
            weight_inputs.append(w)
        if sum(weight_inputs) > 0:
            weights = np.array(weight_inputs) / sum(weight_inputs)

    st.markdown("---")
    st.subheader("Risk Parameters")

    confidence_level = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)
    risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100

    horizon_days = st.selectbox("Risk Horizon (days)", [1, 5, 10, 20, 30], index=0)
    n_simulations = st.slider("Monte Carlo Simulations", 100, 5000, 1000, 100)

    period_days = st.slider("Historical Data (days)", 60, 730, 252, 30)

    if st.button("Run Risk Analysis", type="primary"):
        with st.spinner("Fetching data and calculating risk metrics..."):
            risk_manager = RiskManager(tickers, weights)
            data = risk_manager.fetch_data(period=period_days)

            if data is None or data.empty:
                st.error("No data found. Please check ticker symbols.")
                st.stop()

            # Optimize if selected
            if weight_method == "Optimized (Sharpe)":
                opt_result = risk_manager.optimize_portfolio('sharpe', risk_free_rate)
                if opt_result:
                    risk_manager.weights = opt_result['weights']
                    st.success(f"Portfolio optimized! Expected Sharpe: {opt_result['sharpe_ratio']:.2f}")
            elif weight_method == "Optimized (Min Variance)":
                opt_result = risk_manager.optimize_portfolio('min_variance')
                if opt_result:
                    risk_manager.weights = opt_result['weights']
                    st.success(f"Portfolio optimized! Expected Volatility: {opt_result['volatility']:.2%}")

            st.session_state['risk_manager'] = risk_manager
            st.session_state['data'] = data
            st.session_state['tickers'] = tickers
            st.session_state['weights'] = risk_manager.weights
            st.session_state['confidence_level'] = confidence_level
            st.session_state['risk_free_rate'] = risk_free_rate
            st.session_state['horizon_days'] = horizon_days
            st.session_state['n_simulations'] = n_simulations
            st.success("Risk analysis complete!")

# --- 4. Dashboard Display ---
if 'risk_manager' in st.session_state:
    risk_manager = st.session_state['risk_manager']
    data = st.session_state['data']
    tickers = st.session_state['tickers']
    weights = st.session_state['weights']
    confidence_level = st.session_state['confidence_level']
    risk_free_rate = st.session_state['risk_free_rate']
    horizon_days = st.session_state['horizon_days']
    n_simulations = st.session_state['n_simulations']

    section_header("Holdings", f"Portfolio Risk Analysis &middot; {', '.join(tickers)}")

    # Display weights - Fixed: Convert numpy array to list
    st.write("**Portfolio Weights**")
    weight_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': list(weights)  # Convert numpy array to list
    })
    weight_df['Weight %'] = (weight_df['Weight'] * 100).round(2)
    st.dataframe(weight_df[['Ticker', 'Weight %']], hide_index=True)

    # Get report
    report = risk_manager.generate_report()

    if report:
        # Row 1: Key Metrics
        section_header("Performance", "Key Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Return", f"{report['total_return']:.2%}")
        with col2:
            st.metric("Annualized Return", f"{report['annualized_return']:.2%}")
        with col3:
            st.metric("Volatility", f"{report['volatility']:.2%}")
        with col4:
            st.metric("Sharpe Ratio", f"{report['sharpe_ratio']:.2f}")
        with col5:
            st.metric("Max Drawdown", f"{report['max_drawdown']:.2%}")

        # Row 2: VaR Metrics
        section_header("Downside", "Value at Risk & Expected Shortfall")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(f"VaR {confidence_level:.0%} (Historical)",
                      f"{report['var_95_historical'] * 100:.2f}%")
        with col2:
            st.metric(f"VaR {confidence_level:.0%} (Parametric)",
                      f"{report['var_95_parametric'] * 100:.2f}%")
        with col3:
            st.metric(f"VaR {confidence_level:.0%} (Monte Carlo)",
                      f"{report['var_95_monte_carlo'] * 100:.2f}%")
        with col4:
            st.metric("VaR 99% (Historical)",
                      f"{report['var_99_historical'] * 100:.2f}%")
        with col5:
            st.metric(f"CVaR {confidence_level:.0%}",
                      f"{report['cvar_95'] * 100:.2f}%")

        # Row 3: Additional Risk Metrics
        section_header("Ratios", "Risk Ratios & Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Sortino Ratio", f"{report['sortino_ratio']:.2f}")
        with col2:
            st.metric("Calmar Ratio", f"{report['calmar_ratio']:.2f}")
        with col3:
            st.metric("Beta", f"{report['beta']:.2f}")
        with col4:
            st.metric("Current Drawdown", f"{report['current_drawdown']:.2%}")
        with col5:
            st.metric("Holding Period", f"{horizon_days} days")

        # Row 4: Portfolio Performance Chart
        section_header("Trajectory", "Portfolio Performance")
        portfolio_values = risk_manager._calculate_portfolio_values()
        if portfolio_values is not None:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("Portfolio Value", "Drawdown"),
                                vertical_spacing=0.12)

            # Portfolio value
            fig.add_trace(
                go.Scatter(x=portfolio_values.index, y=portfolio_values,
                           mode='lines', name='Portfolio Value',
                           line=dict(color=PALETTE["emerald"], width=2)),
                row=1, col=1
            )

            # Drawdown
            dd_data = risk_manager.calculate_max_drawdown()
            if dd_data:
                fig.add_trace(
                    go.Scatter(x=dd_data['drawdown_series'].index,
                               y=dd_data['drawdown_series'] * 100,
                               mode='lines', name='Drawdown %',
                               line=dict(color=PALETTE["burgundy"], width=2),
                               fillcolor=PALETTE["burgundy_soft"],
                               fill='tozeroy'),
                    row=2, col=1
                )

            themed_layout(fig, height=520)
            fig.update_layout(showlegend=False)
            for ann in fig['layout']['annotations']:
                ann['font'] = dict(family="Fraunces, serif", size=14, color=PALETTE["ink"])
            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

        # Row 5: Correlation Heatmap
        section_header("Structure", "Correlation Matrix")
        corr_matrix = risk_manager.corr_matrix
        if corr_matrix is not None:
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale=[
                    [0.0, PALETTE["burgundy"]],
                    [0.5, PALETTE["paper"]],
                    [1.0, PALETTE["emerald"]],
                ],
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 11, "family": "IBM Plex Mono, monospace"},
            ))
            themed_layout(fig, height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Row 6: Risk Contribution
        section_header("Composition", "Risk Contribution by Asset")
        risk_contrib = risk_manager.calculate_risk_contribution()
        if risk_contrib:
            contrib_df = pd.DataFrame({
                'Asset': list(risk_contrib.keys()),
                'Risk Contribution %': list(risk_contrib.values())
            })
            fig = px.pie(contrib_df, values='Risk Contribution %', names='Asset',
                         color_discrete_sequence=PLOTLY_COLORWAY, hole=0.45)
            themed_layout(fig, height=440, title="Risk Contribution")
            st.plotly_chart(fig, use_container_width=True)

        # Row 7: Monte Carlo Simulation
        section_header("Projection", "Monte Carlo Simulation")
        with st.spinner("Running Monte Carlo simulation..."):
            mc_results = risk_manager.run_monte_carlo(n_simulations, horizon_days)
            if mc_results:
                # Plot Monte Carlo paths
                fig = go.Figure()

                # Plot selected paths (sample)
                sample_paths = mc_results['paths'][:50]  # Show 50 paths for clarity
                for path in sample_paths:
                    fig.add_trace(go.Scatter(
                        y=path,
                        mode='lines',
                        line=dict(width=0.6, color="rgba(14,26,43,0.10)"),
                        showlegend=False
                    ))

                # Plot percentiles
                percentiles = mc_results['percentiles']
                fig.add_trace(go.Scatter(
                    y=[percentiles['5th']] * len(path),
                    mode='lines',
                    name='5th Percentile',
                    line=dict(color=PALETTE["burgundy"], width=2, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    y=[percentiles['50th']] * len(path),
                    mode='lines',
                    name='Median',
                    line=dict(color=PALETTE["emerald"], width=2)
                ))
                fig.add_trace(go.Scatter(
                    y=[percentiles['95th']] * len(path),
                    mode='lines',
                    name='95th Percentile',
                    line=dict(color=PALETTE["gold"], width=2, dash='dash')
                ))

                themed_layout(fig, height=500, title="Monte Carlo Projections")
                fig.update_layout(xaxis_title='Days', yaxis_title='Portfolio Value ($)')
                st.plotly_chart(fig, use_container_width=True)

                # Show percentiles
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("5th Percentile", f"${percentiles['5th']:.0f}")
                with col2:
                    st.metric("25th Percentile", f"${percentiles['25th']:.0f}")
                with col3:
                    st.metric("Median", f"${percentiles['50th']:.0f}")
                with col4:
                    st.metric("75th Percentile", f"${percentiles['75th']:.0f}")
                with col5:
                    st.metric("95th Percentile", f"${percentiles['95th']:.0f}")

        # Row 8: Stress Testing
        section_header("Scenarios", "Stress Testing")
        col1, col2 = st.columns(2)

        with col1:
            scenario = st.selectbox(
                "Select Stress Scenario",
                ["market_crash", "volatility_shock", "flash_crash", "interest_rate_hike", "normal_conditions"]
            )
            if st.button("Run Stress Test"):
                stress_results = risk_manager.stress_test(scenario)
                if stress_results:
                    st.session_state['stress_results'] = stress_results

        with col2:
            if 'stress_results' in st.session_state:
                stress = st.session_state['stress_results']
                st.info(f"""
                **Stress Test Results &middot; {scenario.replace('_', ' ').title()}**
                - **Shock Return:** {stress['shock_return']:.2%}
                - **Portfolio Loss:** {stress['portfolio_loss']:.2%}
                - **VaR Shock:** {stress['var_shock']:.2%}
                - **Impact Level:** {stress['impact']}
                """)

        # Row 9: Data Table
        with st.expander("Historical Data"):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            st.dataframe(data.tail(20).style.format("{:.2f}", subset=numeric_cols))

else:
    st.info("Configure the portfolio parameters in the sidebar and select **Run Risk Analysis** to begin.")
    section_header("Method", "How This Risk Management Model Works")
    st.markdown("""
    This system provides institutional-grade risk analytics across seven disciplines:

    1. **Value at Risk (VaR)** &mdash; Historical, Parametric, and Monte Carlo methods
    2. **Expected Shortfall (CVaR)** &mdash; average loss beyond VaR
    3. **Portfolio Optimization** &mdash; Markowitz efficient frontier (Sharpe & Min Variance)
    4. **Monte Carlo Simulation** &mdash; 1,000+ paths for portfolio projections
    5. **Stress Testing** &mdash; market crash, volatility shock, and flash crash scenarios
    6. **Risk Metrics** &mdash; Sharpe, Sortino, Calmar ratios, Beta, and Maximum Drawdown
    7. **Visualizations** &mdash; performance charts, correlation heatmaps, and risk contribution
    """)
