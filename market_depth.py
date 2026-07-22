# -*- coding: utf-8 -*-
"""
Market Analysis Depth Model - Streamlit Dashboard (v2)

This is a complete, working prototype synthesizing concepts from advanced
market depth analysis systems (bybit-depth, DepthSim, and HFT research).

IMPORTANT: This is a simulation for educational purposes. It demonstrates
the core concepts of market depth analysis without connecting to live exchanges.

Target runtime: Python 3.13.x
See requirements.txt for versions verified against Python 3.13.

IMPROVEMENTS IN v2:
- Robust error handling with user-friendly feedback
- Improved caching strategy to reduce recomputation
- Volatility-aware spread and depth simulation
- Better code organization with constants and helpers
- Input validation for all user parameters
- Export functionality for analysis results
"""

import sys

if sys.version_info < (3, 13):
    raise RuntimeError(
        f"This app targets Python 3.13.x. Detected {sys.version.split()[0]}. "
        "Create a fresh virtualenv with Python 3.13 (e.g. `python3.13 -m venv .venv`) "
        "and reinstall dependencies from requirements.txt."
    )

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import warnings
import logging
from io import StringIO
import csv

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
class Config:
    """Configuration constants for the application."""
    MIN_ORDER_SIZE = 100
    DEFAULT_ORDER_SIZE = 50000
    MIN_DEPTH_LEVELS = 5
    MAX_DEPTH_LEVELS = 50
    DEFAULT_DEPTH_LEVELS = 20
    LARGE_WALL_THRESHOLD = 5000
    WARNING_WALL_THRESHOLD = 5000
    SNAPSHOT_SAMPLING_RATIO = 1 / 20  # Sample 1 in 20 data points
    DEFAULT_TRADE_COUNT = 50
    INSTITUTIONAL_TRADE_RATIO = 0.6
    VOLATILITY_SENSITIVITY = 50.0
    BASE_SPREAD_BPS = 4.0
    DEPTH_IMBALANCE_FACTOR = 0.15
    METRIC_DECIMAL_PLACES = 2
    CACHE_TTL_SECONDS = 300

# --- Setup Page Config ---
st.set_page_config(page_title="Market Depth Analysis", page_icon="📊", layout="wide")

# =============================================================================
# DESIGN SYSTEM
# -----------------------------------------------------------------------------
# Same research-desk aesthetic used across the other apps in this suite: ink
# slate + ledger ivory, deep emerald / antique gold accents, Fraunces for
# display type, Inter for body text, IBM Plex Mono for figures.
# =============================================================================

PALETTE = {
    "ink": "#2B3B50",        # soft slate navy — sidebar, headings
    "ink_2": "#374B65",      # secondary ink surface
    "paper": "#F6F4EE",      # warm ivory — page background
    "paper_2": "#EFEBE0",    # card / metric surface
    "rule": "rgba(43,59,80,0.10)",   # hairline dividers
    "text": "#33404F",       # body text on paper
    "muted": "#697787",      # secondary text
    "paper_text": "#D9D4C7", # text on ink surfaces
    "emerald": "#33604F",    # primary accent — bids, gains, confidence
    "emerald_soft": "rgba(51,96,79,0.08)",
    "gold": "#B0925F",       # secondary accent — highlights, rules
    "gold_soft": "rgba(176,146,95,0.12)",
    "burgundy": "#8A4A4A",   # asks, risk / loss accent
    "burgundy_soft": "rgba(138,74,74,0.08)",
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
            font-size: 0.92rem;
        }}

        .block-container {{
            padding-top: 1.6rem !important;
            padding-bottom: 1.5rem !important;
            max-width: 1300px;
        }}

        /* ---------- Typography ---------- */
        h1, h2, h3, h4 {{
            font-family: 'Fraunces', serif !important;
            color: {PALETTE["ink"]} !important;
            font-weight: 500 !important;
            letter-spacing: -0.01em;
        }}
        h3 {{
            border-bottom: 1px solid {PALETTE["rule"]};
            padding-bottom: 0.3rem;
            margin-top: 1.1rem !important;
            margin-bottom: 0.6rem !important;
            font-size: 1.15rem !important;
        }}
        .eyebrow {{
            display: block;
            font-family: 'Inter', sans-serif;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            color: {PALETTE["gold"]};
            margin-bottom: 0.1rem;
        }}

        /* ---------- Masthead ---------- */
        .masthead {{
            border-top: 2px solid {PALETTE["ink"]};
            border-bottom: 1px solid {PALETTE["rule"]};
            padding: 0.5rem 0 0.6rem 0;
            margin-bottom: 0.9rem;
        }}
        .masthead .eyebrow {{ margin-bottom: 0.2rem; }}
        .masthead h1 {{
            font-size: 1.55rem !important;
            margin: 0 !important;
            line-height: 1.15;
        }}
        .masthead .dek {{
            font-family: 'Inter', sans-serif;
            color: {PALETTE["muted"]};
            font-size: 0.82rem;
            margin-top: 0.2rem;
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
            font-weight: 500 !important;
            color: {PALETTE["paper_text"]} !important;
            border-bottom: 1px solid rgba(217,212,199,0.14);
            padding-bottom: 0.3rem;
            margin-top: 0.4rem !important;
            margin-bottom: 0.4rem !important;
            font-size: 1.05rem !important;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: rgba(217,212,199,0.12) !important;
            margin: 0.6rem 0 !important;
        }}
        [data-testid="stSidebar"] label {{ color: {PALETTE["paper_text"]} !important; opacity: 0.8; }}

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: {PALETTE["ink_2"]} !important;
            border: 1px solid rgba(217,212,199,0.16) !important;
            color: {PALETTE["paper_text"]} !important;
            border-radius: 4px !important;
        }}

        /* ---------- Buttons ---------- */
        .stButton > button, button[kind="primary"] {{
            background: {PALETTE["emerald"]} !important;
            color: {PALETTE["paper_text"]} !important;
            border: 1px solid {PALETTE["emerald"]} !important;
            border-radius: 4px !important;
            font-weight: 500 !important;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            font-size: 0.74rem !important;
            padding: 0.35rem 0.85rem !important;
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
            padding: 0.5rem 0.65rem 0.4rem 0.65rem;
        }}
        [data-testid="stMetricLabel"] {{
            font-family: 'Inter', sans-serif !important;
            font-size: 0.66rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {PALETTE["muted"]} !important;
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'IBM Plex Mono', monospace !important;
            color: {PALETTE["ink"]} !important;
            font-weight: 500 !important;
            font-size: 1.3rem !important;
        }}

        /* ---------- Dataframes & expanders ---------- */
        [data-testid="stDataFrame"] {{
            border: 1px solid {PALETTE["rule"]};
            border-radius: 6px;
            overflow: hidden;
            font-size: 0.85rem;
        }}
        [data-testid="stExpander"] {{
            border: 1px solid {PALETTE["rule"]} !important;
            border-radius: 6px !important;
            background: {PALETTE["paper_2"]};
        }}

        /* ---------- Rules ---------- */
        hr {{ border-color: {PALETTE["rule"]} !important; margin: 0.6rem 0 !important; }}

        /* ---------- Section header block ---------- */
        .section-head h3 {{ margin-top: 0 !important; }}

        /* ---------- General compaction ---------- */
        div[data-testid="stVerticalBlock"] {{ gap: 0.5rem; }}
        div[data-testid="stHorizontalBlock"] {{ gap: 0.6rem; }}
        .element-container {{ margin-bottom: 0.15rem !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def masthead(eyebrow, title, dek):
    st.markdown(
        f"""
        <div class="masthead">
            <span class="eyebrow">{eyebrow}</span>
            <h1>{title}</h1>
            <div class="dek">{dek}</div>
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
        font=dict(family="Inter, sans-serif", color=PALETTE["text"], size=12),
        colorway=PLOTLY_COLORWAY,
        margin=dict(t=46 if title else 24, l=10, r=10, b=10),
        legend=dict(font=dict(family="Inter, sans-serif", size=11)),
    )
    if title:
        layout_kwargs["title"] = dict(
            text=title, font=dict(family="Fraunces, serif", size=15, color=PALETTE["ink"])
        )
    if height:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor="rgba(43,59,80,0.08)", zerolinecolor="rgba(43,59,80,0.15)", linecolor=PALETTE["rule"])
    fig.update_yaxes(gridcolor="rgba(43,59,80,0.08)", zerolinecolor="rgba(43,59,80,0.15)", linecolor=PALETTE["rule"])
    return fig


inject_design_system()

masthead(
    eyebrow="Market Microstructure Desk",
    title="Market Analysis Depth Model",
    dek="Professional order book depth analysis with simulated L2/L3 market data",
)

# Try to import depthsim for enhanced features (optional)
try:
    from depthsim import DepthSimulator
    HAS_DEPTHSIM = True
except ImportError:
    HAS_DEPTHSIM = False

# --- Custom Exceptions ---
class MarketDataError(Exception):
    """Raised when market data cannot be fetched or processed."""
    pass

class SimulationError(Exception):
    """Raised when simulation fails."""
    pass

# --- Core Market Depth Simulation Engine ---
class MarketDepthAnalyzer:
    """
    Market Depth Analysis engine with professional simulation capabilities.
    
    Attributes:
        ticker (str): The stock ticker symbol
        data (pd.DataFrame): OHLCV market data
        quotes (pd.DataFrame): Simulated bid-ask quotes
        depth_snapshots (dict): L2/L3 order book snapshots at various time points
        trade_sequence (list): Simulated trade history
        depth_metrics (dict): Computed market depth metrics
    """
    
    def __init__(self, ticker: str, data: pd.DataFrame):
        """
        Initialize the analyzer.
        
        Args:
            ticker: Stock ticker symbol
            data: OHLCV DataFrame with columns: Close, Volume, High, Low
            
        Raises:
            ValueError: If data is empty or malformed
        """
        if data.empty:
            raise ValueError("Market data cannot be empty")
        if not all(col in data.columns for col in ['Close', 'Volume']):
            raise ValueError("Data must contain 'Close' and 'Volume' columns")
        
        self.ticker = ticker
        self.data = data
        self.quotes = None
        self.depth_snapshots = None
        self.trade_sequence = None
        self.depth_metrics = {}
        self.l2_snapshots = None
        
    def simulate_market_depth(self, spread_model: str = 'volatility', levels: int = 20):
        """
        Generate realistic market depth data using professional simulation.
        Uses depthsim if available; otherwise uses an enhanced built-in simulation.
        
        Args:
            spread_model: Model for bid-ask spread ('volatility', 'volume', 'constant')
            levels: Number of order book levels (5-50)
            
        Returns:
            self (for method chaining)
            
        Raises:
            SimulationError: If simulation fails
        """
        if not (Config.MIN_DEPTH_LEVELS <= levels <= Config.MAX_DEPTH_LEVELS):
            raise ValueError(f"Levels must be between {Config.MIN_DEPTH_LEVELS} and {Config.MAX_DEPTH_LEVELS}")
        
        try:
            if HAS_DEPTHSIM:
                self._simulate_with_depthsim(spread_model, levels)
            else:
                self._simulate_builtin(spread_model, levels)
            self._compute_depth_metrics()
            return self
        except Exception as e:
            logger.error(f"Simulation failed for {self.ticker}: {e}")
            raise SimulationError(f"Simulation failed: {str(e)}")
    
    def _simulate_with_depthsim(self, spread_model: str, levels: int):
        """Use the depthsim library for professional market depth generation."""
        sim = DepthSimulator(
            spread_model=spread_model,
            base_spread_bps=Config.BASE_SPREAD_BPS,
            volatility_sensitivity=Config.VOLATILITY_SENSITIVITY,
            depth_levels=levels
        )
        
        self.quotes = sim.generate_quotes(self.data)
        self.depth_snapshots = sim.generate_l2_depth_snapshots(
            self.data,
            levels=levels,
            asymmetry_factor=Config.DEPTH_IMBALANCE_FACTOR,
            size_clustering=True
        )
        self.trade_sequence = sim.generate_realistic_trade_sequence(
            self.data,
            trade_intensity=1.5,
            institutional_ratio=Config.INSTITUTIONAL_TRADE_RATIO
        )
    
    def _simulate_builtin(self, spread_model: str, levels: int):
        """
        Built-in simulation fallback when depthsim is not available.
        Uses volatility-aware spread generation for improved realism.
        """
        try:
            prices = self.data['Close'].values
            volumes = self.data['Volume'].values
            
            # Calculate volatility-aware spreads
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * 100  # Convert to percentage
            
            # Generate spread based on model
            spread_bps = self._calculate_spreads(spread_model, prices, volumes, volatility, len(prices))
            
            # Simulate bid-ask quotes
            mid_prices = prices
            bid_prices = mid_prices * (1 - spread_bps / 10000)
            ask_prices = mid_prices * (1 + spread_bps / 10000)
            
            self.quotes = pd.DataFrame({
                'mid_price': mid_prices,
                'bid_price': bid_prices,
                'ask_price': ask_prices,
                'spread_bps': spread_bps,
                'volume': volumes,
                'volatility': volatility
            }, index=self.data.index)
            
            # Generate L2 depth snapshots
            self.depth_snapshots = self._generate_l2_depth_snapshots_builtin(levels)
            self.trade_sequence = self._generate_trades_builtin()
            
        except Exception as e:
            raise SimulationError(f"Built-in simulation failed: {str(e)}")
    
    def _calculate_spreads(self, spread_model: str, prices: np.ndarray, 
                          volumes: np.ndarray, volatility: float, 
                          num_points: int) -> np.ndarray:
        """
        Calculate spread based on the selected model.
        
        Args:
            spread_model: One of 'volatility', 'volume', 'constant'
            prices: Array of prices
            volumes: Array of volumes
            volatility: Market volatility (%)
            num_points: Number of data points
            
        Returns:
            Array of spreads in basis points
        """
        base_spread = Config.BASE_SPREAD_BPS
        
        if spread_model == 'volatility':
            # Higher volatility → higher spread
            volatility_factor = 1 + (volatility / 100) * 5
            spread_bps = base_spread * volatility_factor + np.random.normal(0, 1, num_points)
        
        elif spread_model == 'volume':
            # Higher volume → lower spread (more liquid)
            normalized_vol = (volumes - volumes.min()) / (volumes.max() - volumes.min() + 1)
            volume_factor = 1 + (1 - normalized_vol) * 2
            spread_bps = base_spread * volume_factor + np.random.normal(0, 1, num_points)
        
        elif spread_model == 'constant':
            # Constant spread
            spread_bps = base_spread + np.random.normal(0, 0.5, num_points)
        
        else:
            # Default to volatility model
            volatility_factor = 1 + (volatility / 100) * 5
            spread_bps = base_spread * volatility_factor + np.random.normal(0, 1, num_points)
        
        # Ensure spreads are positive
        return np.maximum(spread_bps, 1)
    
    def _generate_l2_depth_snapshots_builtin(self, levels: int) -> dict:
        """
        Generate synthetic L2 order book data with realistic clustering.
        
        Args:
            levels: Number of order book levels to generate
            
        Returns:
            Dictionary of snapshots keyed by timestamp
        """
        snapshots = {}
        step = max(1, int(len(self.data) * Config.SNAPSHOT_SAMPLING_RATIO))
        
        for i in range(0, len(self.data), step):
            if i + 1 >= len(self.data):
                break
            
            try:
                mid = float(self.data['Close'].iloc[i])
                vol = float(self.data['Volume'].iloc[i])
                
                # Volatility-adjusted spread
                spread = self._get_spread_at_index(i)
                
                # Generate bid levels (prices decreasing from mid)
                bids = self._generate_side_levels(
                    mid, spread, levels, vol, side='bid'
                )
                
                # Generate ask levels (prices increasing from mid)
                asks = self._generate_side_levels(
                    mid, spread, levels, vol, side='ask'
                )
                
                total_bid_size = sum(b['size'] for b in bids)
                total_ask_size = sum(a['size'] for a in asks)
                
                snapshots[self.data.index[i]] = {
                    'mid_price': mid,
                    'spread_bps': spread,
                    'bids': bids,
                    'asks': asks,
                    'total_bid_size': total_bid_size,
                    'total_ask_size': total_ask_size,
                    'depth_imbalance': (total_bid_size - total_ask_size) / 
                                       (total_bid_size + total_ask_size + 1)
                }
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipped snapshot at index {i}: {e}")
                continue
        
        return snapshots
    
    def _get_spread_at_index(self, index: int) -> float:
        """Safely get the spread at a specific index."""
        if self.quotes is not None and index < len(self.quotes):
            return float(self.quotes['spread_bps'].iloc[index])
        return Config.BASE_SPREAD_BPS
    
    def _generate_side_levels(self, mid: float, spread: float, levels: int, 
                             volume: float, side: str) -> list:
        """
        Generate one side (bid/ask) of the order book with clustering.
        
        Args:
            mid: Mid price
            spread: Spread in basis points
            levels: Number of levels to generate
            volume: Base volume for this timestamp
            side: 'bid' or 'ask'
            
        Returns:
            List of order book levels, each with price, size, and order count
        """
        side_levels = []
        direction = -1 if side == 'bid' else 1
        
        for level in range(levels):
            # Price decreases/increases exponentially (more clustering near mid)
            exponential_spacing = (np.exp(level / (levels / 2)) - 1) / 100
            price = mid * (1 + direction * exponential_spacing * spread / 100)
            
            # Size clustering: larger sizes near mid-price, smaller deeper
            base_size = max(1, int(100 + 400 * np.random.rand()))
            decay = np.exp(-level / (levels / 3))  # Exponential decay
            size = max(1, int(base_size * decay))
            
            orders = max(1, int(1 + 5 * np.random.rand()))
            
            side_levels.append({
                'price': price,
                'size': size,
                'orders': orders
            })
        
        return side_levels
    
    def _generate_trades_builtin(self) -> list:
        """
        Generate synthetic trade sequence correlated with price movement.
        
        Returns:
            List of trade dictionaries with time, price, size, and side
        """
        trades = []
        num_trades = min(Config.DEFAULT_TRADE_COUNT, len(self.data))
        trade_indices = np.random.choice(
            len(self.data), 
            size=num_trades, 
            replace=False
        )
        
        for idx in sorted(trade_indices):
            try:
                ts = self.data.index[idx]
                price = float(self.data['Close'].iloc[idx])
                
                # Base size from volume
                vol = float(self.data['Volume'].iloc[idx])
                base_size = vol / num_trades / 1000
                
                # Institutional vs retail ratio
                if np.random.rand() < Config.INSTITUTIONAL_TRADE_RATIO:
                    size = int(base_size * (2 + 3 * np.random.rand()))
                else:
                    size = int(base_size * (0.5 + np.random.rand()))
                
                # Momentum bias
                if idx > 0:
                    prev_price = float(self.data['Close'].iloc[idx - 1])
                    momentum = price > prev_price
                    side = 'buy' if momentum else 'sell'
                else:
                    side = 'buy' if np.random.rand() > 0.4 else 'sell'
                
                trades.append({
                    'time': ts,
                    'price': price,
                    'size': max(1, size),
                    'side': side
                })
            except (ValueError, IndexError):
                continue
        
        return trades
    
    def _compute_depth_metrics(self):
        """Compute key market depth metrics."""
        self.depth_metrics = {}
        
        if self.quotes is not None:
            self.depth_metrics.update({
                'avg_spread_bps': float(self.quotes['spread_bps'].mean()),
                'spread_volatility': float(self.quotes['spread_bps'].std()),
                'avg_mid_price': float(self.quotes['mid_price'].mean()),
                'price_volatility': float(self.quotes['mid_price'].std())
            })
        
        if self.depth_snapshots:
            snapshots_list = list(self.depth_snapshots.values())
            imbalances = [s['depth_imbalance'] for s in snapshots_list]
            self.depth_metrics.update({
                'avg_depth_imbalance': float(np.mean(imbalances)),
                'depth_imbalance_volatility': float(np.std(imbalances)),
                'num_snapshots': len(snapshots_list)
            })
    
    def get_latest_depth(self) -> dict:
        """
        Get the most recent depth snapshot.
        
        Returns:
            Latest depth snapshot dict, or None if no snapshots exist
        """
        if not self.depth_snapshots:
            return None
        latest_key = list(self.depth_snapshots.keys())[-1]
        return self.depth_snapshots[latest_key]
    
    def calculate_order_book_analytics(self) -> dict:
        """
        Calculate advanced order book analytics (liquidity, walls, etc.).
        
        Returns:
            Dictionary of analytics keyed by timestamp
        """
        if not self.depth_snapshots:
            return {}
        
        analytics = {}
        sample_size = min(10, len(self.depth_snapshots))
        
        for key, snapshot in list(self.depth_snapshots.items())[-sample_size:]:
            bids = snapshot['bids']
            asks = snapshot['asks']
            
            top_bid_depth = sum(b['size'] for b in bids[:5])
            top_ask_depth = sum(a['size'] for a in asks[:5])
            
            bid_wall = max(b['size'] for b in bids) if bids else 0
            ask_wall = max(a['size'] for a in asks) if asks else 0
            
            analytics[key] = {
                'top_5_bid_depth': top_bid_depth,
                'top_5_ask_depth': top_ask_depth,
                'bid_wall_size': bid_wall,
                'ask_wall_size': ask_wall,
                'spread_bps': snapshot['spread_bps'],
                'depth_imbalance': snapshot['depth_imbalance']
            }
        
        return analytics
    
    def simulate_market_impact(self, order_size: float, side: str = 'buy') -> dict:
        """
        Simulate market impact for a given order size.
        
        Args:
            order_size: Size of the order
            side: 'buy' or 'sell'
            
        Returns:
            Dictionary with impact metrics (avg_price, impact_bps, etc.)
            
        Raises:
            ValueError: If order_size is invalid or side is not 'buy'/'sell'
        """
        if order_size <= 0:
            raise ValueError("Order size must be positive")
        if side not in ('buy', 'sell'):
            raise ValueError("Side must be 'buy' or 'sell'")
        
        latest_depth = self.get_latest_depth()
        if not latest_depth:
            return None
        
        levels = latest_depth['asks'] if side == 'buy' else latest_depth['bids']
        if not levels:
            return None
        
        mid_price = latest_depth['mid_price']
        
        remaining = order_size
        total_cost = 0
        levels_consumed = 0
        executed_size = 0
        
        for level in levels:
            if remaining <= 0:
                break
            level_price = level['price']
            level_size = level['size']
            
            if remaining >= level_size:
                total_cost += level_size * level_price
                remaining -= level_size
                executed_size += level_size
            else:
                total_cost += remaining * level_price
                executed_size += remaining
                remaining = 0
            levels_consumed += 1
        
        if executed_size == 0:
            return None
        
        avg_price = total_cost / executed_size
        impact_bps = (avg_price / mid_price - 1) * 10000 if side == 'buy' else (1 - avg_price / mid_price) * 10000
        
        return {
            'average_price': avg_price,
            'impact_bps': impact_bps,
            'levels_consumed': levels_consumed,
            'executed_size': executed_size,
            'fill_rate': executed_size / order_size,
            'unexecuted_size': remaining
        }

# --- Data Fetching & Processing ---
@st.cache_data(ttl=Config.CACHE_TTL_SECONDS)
def fetch_and_process_data(ticker: str, period: int) -> pd.DataFrame:
    """
    Fetch market data and prepare for depth analysis.
    
    Args:
        ticker: Stock ticker symbol
        period: Number of days of historical data
        
    Returns:
        OHLCV DataFrame, or None if fetch fails
        
    Raises:
        MarketDataError: If data fetch or processing fails
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period)
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise MarketDataError(f"No data found for ticker '{ticker}'")
        
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Validate required columns
        required_cols = ['Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            raise MarketDataError(f"Data missing required columns: {required_cols}")
        
        # Remove NaN rows
        data = data.dropna()
        
        if data.empty:
            raise MarketDataError("All data rows are NaN")
        
        logger.info(f"Fetched {len(data)} rows for {ticker}")
        return data
    
    except yf.YFinanceError as e:
        logger.error(f"yfinance error: {e}")
        raise MarketDataError(f"Failed to fetch data: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error fetching data: {e}")
        raise MarketDataError(f"Unexpected error: {str(e)}")

# --- Helper Functions ---
def export_analytics_to_csv(analytics: dict, ticker: str) -> str:
    """
    Export analytics to CSV format.
    
    Args:
        analytics: Dictionary of analytics data
        ticker: Stock ticker for filename
        
    Returns:
        CSV string ready for download
    """
    if not analytics:
        return ""
    
    output = StringIO()
    df = pd.DataFrame(analytics).T
    df.to_csv(output)
    return output.getvalue()

def format_metric(value: float, decimal_places: int = Config.METRIC_DECIMAL_PLACES) -> str:
    """Format a metric value with consistent decimals."""
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value:.{decimal_places}f}"

# --- Streamlit Dashboard ---

# Sidebar configuration
with st.sidebar:
    st.markdown('<span class="eyebrow">Desk Setup</span>', unsafe_allow_html=True)
    st.header("Configuration")
    
    ticker = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        help="Enter a valid stock ticker (e.g., AAPL, MSFT)"
    ).strip().upper()
    
    period = st.selectbox(
        "Data Period (days)",
        [30, 60, 90, 180, 365],
        index=2,
        help="How far back to fetch historical data"
    )
    
    st.markdown("---")
    st.subheader("Depth Simulation Parameters")
    
    spread_model = st.selectbox(
        "Spread Model",
        ["volatility", "volume", "constant"],
        help="Model for bid-ask spread simulation. 'volatility': spreads widen with price swings. "
             "'volume': spreads narrow with higher trading volume. 'constant': steady spreads."
    )
    
    depth_levels = st.slider(
        "Order Book Levels",
        min_value=Config.MIN_DEPTH_LEVELS,
        max_value=Config.MAX_DEPTH_LEVELS,
        value=Config.DEFAULT_DEPTH_LEVELS,
        help="Number of price levels to simulate on each side"
    )
    
    if not HAS_DEPTHSIM:
        st.info(
            "ℹ️ **Using built-in simulation.** For enhanced features, install: "
            "`pip install conflux-depthsim`"
        )
    
    if st.button("Run Depth Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Fetching {period} days of data for {ticker}..."):
            try:
                data = fetch_and_process_data(ticker, period)
                
                with st.spinner("Running market depth simulation..."):
                    analyzer = MarketDepthAnalyzer(ticker, data)
                    analyzer.simulate_market_depth(
                        spread_model=spread_model,
                        levels=depth_levels
                    )
                
                st.session_state['analyzer'] = analyzer
                st.session_state['data'] = data
                st.session_state['ticker'] = ticker
                st.session_state['analytics_cache'] = None  # Invalidate cache
                
                st.success("✅ Depth analysis complete!")
                
            except MarketDataError as e:
                st.error(f"❌ Data Error: {str(e)}")
                st.info("💡 Try another ticker or adjust the date range.")
            except SimulationError as e:
                st.error(f"❌ Simulation Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
                logger.exception("Unhandled exception in analysis")

# --- Dashboard Display ---
if 'analyzer' in st.session_state:
    analyzer = st.session_state['analyzer']
    data = st.session_state['data']
    ticker = st.session_state['ticker']
    
    section_header("Overview", f"Market Depth Analysis \u00b7 {ticker}")
    
    # Row 1: Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = analyzer.depth_metrics
    
    with col1:
        st.metric(
            "Avg Spread (bps)",
            format_metric(metrics.get('avg_spread_bps', 0))
        )
    with col2:
        st.metric(
            "Spread Vol",
            format_metric(metrics.get('spread_volatility', 0))
        )
    with col3:
        st.metric(
            "Depth Imbalance",
            format_metric(metrics.get('avg_depth_imbalance', 0), 3)
        )
    with col4:
        latest = analyzer.get_latest_depth()
        price_str = f"${latest['mid_price']:.2f}" if latest else "N/A"
        st.metric("Mid Price", price_str)
    with col5:
        st.metric(
            "Snapshots",
            metrics.get('num_snapshots', 0)
        )
    
    # Row 2: Market Impact Simulation
    section_header("Execution", "Market Impact Simulation")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        impact_size = st.number_input(
            "Order Size ($)",
            min_value=Config.MIN_ORDER_SIZE,
            value=Config.DEFAULT_ORDER_SIZE,
            step=10000,
            help="Size of hypothetical order in dollars"
        )
        
        impact_side = st.selectbox(
            "Order Side",
            ["buy", "sell"],
            help="Buy or sell order"
        )
        
        if st.button("Calculate Impact", use_container_width=True):
            try:
                impact = analyzer.simulate_market_impact(impact_size, impact_side)
                if impact:
                    st.session_state['impact_result'] = impact
                else:
                    st.warning("⚠️ Insufficient liquidity to execute this order.")
            except ValueError as e:
                st.error(f"❌ Invalid input: {str(e)}")
    
    with col2:
        if 'impact_result' in st.session_state:
            impact = st.session_state['impact_result']
            st.info(f"""
            **Market Impact Analysis**
            - **Avg Price:** ${format_metric(impact['average_price'], 4)}
            - **Impact:** {format_metric(impact['impact_bps'], 1)} bps
            - **Levels Consumed:** {impact['levels_consumed']}
            - **Executed Size:** {impact['executed_size']:,.0f}
            - **Fill Rate:** {impact['fill_rate']:.1%}
            - **Unexecuted:** {impact['unexecuted_size']:,.0f}
            """)
    
    # Row 3: L2 Depth Visualization
    section_header("Order Book", "L2 Order Book Depth")
    
    latest_depth = analyzer.get_latest_depth()
    if latest_depth:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Bid Side Depth", "Ask Side Depth"),
            shared_yaxes=False
        )
        
        bids = latest_depth['bids'][:10]
        bid_prices = [b['price'] for b in bids]
        bid_sizes = [b['size'] for b in bids]
        bid_orders = [b['orders'] for b in bids]
        
        fig.add_trace(
            go.Bar(
                x=bid_prices,
                y=bid_sizes,
                name='Bid Size',
                marker_color=PALETTE["emerald"],
                text=[f"{o} orders" for o in bid_orders],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        asks = latest_depth['asks'][:10]
        ask_prices = [a['price'] for a in asks]
        ask_sizes = [a['size'] for a in asks]
        ask_orders = [a['orders'] for a in asks]
        
        fig.add_trace(
            go.Bar(
                x=ask_prices,
                y=ask_sizes,
                name='Ask Size',
                marker_color=PALETTE["burgundy"],
                text=[f"{o} orders" for o in ask_orders],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        themed_layout(fig, height=360)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text="Price", row=1, col=1)
        fig.update_xaxes(title_text="Price", row=1, col=2)
        fig.update_yaxes(title_text="Size", row=1, col=1)
        fig.update_yaxes(title_text="Size", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Bid Depth", f"{latest_depth['total_bid_size']:,}")
        with col2:
            st.metric("Total Ask Depth", f"{latest_depth['total_ask_size']:,}")
        with col3:
            st.metric(
                "Depth Imbalance",
                format_metric(latest_depth['depth_imbalance'], 3)
            )
    
    # Row 4: Spread and Depth Trends
    section_header("Trends", "Spread & Depth Trends")
    
    if analyzer.quotes is not None:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Bid-Ask Spread Over Time", "Mid Price"),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(
                x=analyzer.quotes.index,
                y=analyzer.quotes['spread_bps'],
                mode='lines',
                name='Spread (bps)',
                line=dict(color=PALETTE["gold"])
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=analyzer.quotes.index,
                y=analyzer.quotes['mid_price'],
                mode='lines',
                name='Mid Price',
                line=dict(color=PALETTE["emerald"])
            ),
            row=2, col=1
        )
        
        themed_layout(fig, height=360)
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 5: Order Book Analytics (with caching)
    section_header("Structure", "Order Book Analytics")
    
    if st.session_state.get('analytics_cache') is None:
        st.session_state['analytics_cache'] = analyzer.calculate_order_book_analytics()
    
    analytics = st.session_state['analytics_cache']
    
    if analytics:
        analytics_df = pd.DataFrame(analytics).T
        st.dataframe(
            analytics_df.style.format({
                'top_5_bid_depth': '{:,.0f}',
                'top_5_ask_depth': '{:,.0f}',
                'bid_wall_size': '{:,.0f}',
                'ask_wall_size': '{:,.0f}',
                'spread_bps': '{:.2f}',
                'depth_imbalance': '{:+.3f}'
            }),
            use_container_width=True
        )
        
        # Warning for large walls
        if 'bid_wall_size' in analytics_df.columns and 'ask_wall_size' in analytics_df.columns:
            max_bid_wall = analytics_df['bid_wall_size'].max()
            max_ask_wall = analytics_df['ask_wall_size'].max()
            
            if max_bid_wall > Config.WARNING_WALL_THRESHOLD:
                st.warning(
                    f"⚠️ **Large Bid Wall Detected:** {max_bid_wall:,.0f} units at one level"
                )
            if max_ask_wall > Config.WARNING_WALL_THRESHOLD:
                st.warning(
                    f"⚠️ **Large Ask Wall Detected:** {max_ask_wall:,.0f} units at one level"
                )
        
        # Export functionality
        csv_data = export_analytics_to_csv(analytics, ticker)
        st.download_button(
            label="📥 Download Analytics as CSV",
            data=csv_data,
            file_name=f"{ticker}_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Row 6: Recent Trade Activity
    section_header("Tape", "Recent Trade Activity")
    if analyzer.trade_sequence:
        trades_df = pd.DataFrame(analyzer.trade_sequence)
        trades_df['side'] = trades_df['side'].apply(
            lambda x: '🟢 Buy' if x == 'buy' else '🔴 Sell'
        )
        st.dataframe(trades_df.tail(20), use_container_width=True)
        
        buy_trades = [t for t in analyzer.trade_sequence if t['side'] == 'buy']
        sell_trades = [t for t in analyzer.trade_sequence if t['side'] == 'sell']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", len(analyzer.trade_sequence))
        with col2:
            st.metric("Buy Trades", len(buy_trades))
        with col3:
            st.metric("Sell Trades", len(sell_trades))

else:
    st.info("👈 **Configure the parameters in the sidebar and click 'Run Depth Analysis' to start.**")
