# -*- coding: utf-8 -*-
"""
Market Analysis Depth Model - Streamlit Dashboard

This is a working, integrated prototype synthesizing concepts from advanced
market depth analysis systems (bybit-depth, DepthSim, and HFT research).

IMPORTANT: This is a simulation for educational purposes. It demonstrates
the core concepts of market depth analysis without connecting to live exchanges.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf

# Try to import depthsim; fallback to simulation if not installed
try:
    from depthsim import DepthSimulator
    HAS_DEPTHSIM = True
except ImportError:
    HAS_DEPTHSIM = False
    st.warning("`conflux-depthsim` not installed. Using built-in simulation engine.")

# --- 1. Configuration ---
st.set_page_config(page_title="Market Depth Analysis", layout="wide")
st.title("📊 Market Analysis Depth Model")
st.markdown("**Professional order book depth analysis with simulated L2/L3 market data**")

# --- 2. Core Market Depth Simulation Engine (using depthsim if available) ---
class MarketDepthAnalyzer:
    """Market Depth Analysis engine with professional simulation capabilities."""
    
    def __init__(self, ticker, data):
        self.ticker = ticker
        self.data = data
        self.quotes = None
        self.depth_snapshots = None
        self.trade_sequence = None
        self.depth_metrics = {}
        self.l2_snapshots = None  # For L2 depth analysis
        
    def simulate_market_depth(self, spread_model='volatility', levels=20):
        """
        Generate realistic market depth data using professional simulation.
        Uses depthsim if available; otherwise uses a simplified simulation.
        """
        if HAS_DEPTHSIM:
            return self._simulate_with_depthsim(spread_model, levels)
        else:
            return self._simulate_builtin(spread_model, levels)
    
    def _simulate_with_depthsim(self, spread_model, levels):
        """Use the depthsim library for professional market depth generation."""
        sim = DepthSimulator(
            spread_model=spread_model,
            base_spread_bps=4.0,
            volatility_sensitivity=50.0,
            depth_levels=levels
        )
        
        # Generate quotes, depth snapshots, and trade sequences
        self.quotes = sim.generate_quotes(self.data)
        self.depth_snapshots = sim.generate_l2_depth_snapshots(
            self.data,
            levels=levels,
            asymmetry_factor=0.15,
            size_clustering=True
        )
        self.trade_sequence = sim.generate_realistic_trade_sequence(
            self.data,
            trade_intensity=1.5,
            institutional_ratio=0.6
        )
        
        # Compute key metrics
        self._compute_depth_metrics()
        return self
    
    def _simulate_builtin(self, spread_model, levels):
        """Built-in simulation fallback when depthsim is not available."""
        # Simulate price data
        prices = self.data['Close'].values
        volumes = self.data['Volume'].values
        
        # Simulate bid-ask quotes
        spread_bps = 5 + 10 * np.random.rand(len(prices))
        mid_prices = prices
        bid_prices = mid_prices * (1 - spread_bps / 10000)
        ask_prices = mid_prices * (1 + spread_bps / 10000)
        
        # Create quotes DataFrame
        self.quotes = pd.DataFrame({
            'mid_price': mid_prices,
            'bid_price': bid_prices,
            'ask_price': ask_prices,
            'spread_bps': spread_bps,
            'volume': volumes
        }, index=self.data.index)
        
        # Simulate L2 depth snapshots
        self.depth_snapshots = self._generate_l2_depth_snapshots_builtin(levels)
        self.trade_sequence = self._generate_trades_builtin()
        
        self._compute_depth_metrics()
        return self
    
    def _generate_l2_depth_snapshots_builtin(self, levels):
        """Generate synthetic L2 order book data."""
        snapshots = {}
        step = max(1, len(self.data) // 20)  # Take ~20 snapshots
        
        for i in range(0, len(self.data), step):
            if i + 1 >= len(self.data):
                break
                
            mid = self.data['Close'].iloc[i]
            spread = 4 + 8 * np.random.rand()  # Random spread in bps
            tick_size = 0.01
            
            # Generate bid levels (prices decreasing from mid)
            bids = []
            for level in range(levels):
                price = mid * (1 - (level + 1) * spread / 10000)
                size = max(1, int(100 + 500 * np.random.rand()))
                orders = max(1, int(1 + 5 * np.random.rand()))
                bids.append({'price': price, 'size': size, 'orders': orders})
            
            # Generate ask levels (prices increasing from mid)
            asks = []
            for level in range(levels):
                price = mid * (1 + (level + 1) * spread / 10000)
                size = max(1, int(100 + 500 * np.random.rand()))
                orders = max(1, int(1 + 5 * np.random.rand()))
                asks.append({'price': price, 'size': size, 'orders': orders})
            
            snapshots[self.data.index[i]] = {
                'mid_price': mid,
                'spread_bps': spread,
                'bids': bids,
                'asks': asks,
                'total_bid_size': sum(b['size'] for b in bids),
                'total_ask_size': sum(a['size'] for a in asks),
                'depth_imbalance': (sum(b['size'] for b in bids) - sum(a['size'] for a in asks)) / 
                                   (sum(b['size'] for b in bids) + sum(a['size'] for a in asks) + 1)
            }
        
        return snapshots
    
    def _generate_trades_builtin(self):
        """Generate synthetic trade sequence."""
        trades = []
        base_time = self.data.index[0]
        trade_times = np.random.choice(self.data.index, size=min(50, len(self.data)), replace=False)
        
        for i, ts in enumerate(sorted(trade_times)):
            if ts not in self.data.index:
                continue
            idx = self.data.index.get_loc(ts)
            price = self.data['Close'].iloc[idx]
            size = 100 + 900 * np.random.rand()
            side = 'buy' if np.random.rand() > 0.4 else 'sell'
            trades.append({
                'time': ts,
                'price': price,
                'size': int(size),
                'side': side
            })
        
        return trades
    
    def _compute_depth_metrics(self):
        """Compute key market depth metrics."""
        if self.quotes is not None:
            self.depth_metrics = {
                'avg_spread_bps': self.quotes['spread_bps'].mean(),
                'spread_volatility': self.quotes['spread_bps'].std(),
                'avg_mid_price': self.quotes['mid_price'].mean(),
                'price_volatility': self.quotes['mid_price'].std()
            }
        
        # Add L2 metrics
        if self.depth_snapshots:
            snapshots_list = list(self.depth_snapshots.values())
            imbalances = [s['depth_imbalance'] for s in snapshots_list]
            self.depth_metrics.update({
                'avg_depth_imbalance': np.mean(imbalances),
                'depth_imbalance_volatility': np.std(imbalances),
                'num_snapshots': len(snapshots_list)
            })
    
    def get_latest_depth(self):
        """Get the most recent depth snapshot."""
        if not self.depth_snapshots:
            return None
        latest_key = list(self.depth_snapshots.keys())[-1]
        return self.depth_snapshots[latest_key]
    
    def calculate_order_book_analytics(self):
        """Calculate advanced order book analytics (liquidity, walls, etc.)."""
        if not self.depth_snapshots:
            return {}
        
        analytics = {}
        for key, snapshot in list(self.depth_snapshots.items())[:10]:  # Analyze first 10 snapshots
            bids = snapshot['bids']
            asks = snapshot['asks']
            
            # Liquidity analysis
            top_bid_depth = sum(b['size'] for b in bids[:5])
            top_ask_depth = sum(a['size'] for a in asks[:5])
            
            # Wall detection (anomalous large orders)
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
    
    def simulate_market_impact(self, order_size, side='buy'):
        """Simulate market impact for a given order size."""
        latest_depth = self.get_latest_depth()
        if not latest_depth:
            return None
        
        levels = latest_depth['asks'] if side == 'buy' else latest_depth['bids']
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
            'fill_rate': executed_size / order_size
        }

# --- 3. Data Fetching & Processing ---
@st.cache_data(ttl=300)
def fetch_and_process_data(ticker, period):
    """Fetch market data and prepare for depth analysis."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period)
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        return None
    
    # Clean and prepare data
    data = data.dropna()
    return data

# --- 4. Streamlit Dashboard ---

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker = st.text_input("Ticker Symbol", value="AAPL")
    period = st.selectbox("Data Period (days)", [30, 60, 90, 180, 365], index=2)
    
    st.markdown("---")
    st.subheader("Depth Simulation Parameters")
    
    spread_model = st.selectbox(
        "Spread Model",
        ["volatility", "volume", "constant", "volatility_volume", "time_of_day"],
        help="Model for bid-ask spread simulation"
    )
    
    depth_levels = st.slider("Order Book Levels", 5, 50, 20)
    
    if st.button("🔬 Run Depth Analysis", type="primary"):
        with st.spinner(f"Fetching {period} days of data for {ticker}..."):
            data = fetch_and_process_data(ticker, period)
            if data is None:
                st.error("No data found. Please try another ticker.")
                st.stop()
            
            # Initialize and run depth analysis
            analyzer = MarketDepthAnalyzer(ticker, data)
            analyzer.simulate_market_depth(spread_model=spread_model, levels=depth_levels)
            
            st.session_state['analyzer'] = analyzer
            st.session_state['data'] = data
            st.session_state['ticker'] = ticker
            st.success("Depth analysis complete!")

# --- 5. Dashboard Display ---
if 'analyzer' in st.session_state:
    analyzer = st.session_state['analyzer']
    data = st.session_state['data']
    ticker = st.session_state['ticker']
    
    st.subheader(f"📊 Market Depth Analysis: {ticker}")
    
    # Row 1: Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = analyzer.depth_metrics
    
    with col1:
        st.metric("Avg Spread (bps)", f"{metrics.get('avg_spread_bps', 0):.2f}")
    with col2:
        st.metric("Spread Volatility", f"{metrics.get('spread_volatility', 0):.2f}")
    with col3:
        st.metric("Depth Imbalance", f"{metrics.get('avg_depth_imbalance', 0):+.3f}")
    with col4:
        latest = analyzer.get_latest_depth()
        st.metric("Mid Price", f"${latest['mid_price']:.2f}" if latest else "N/A")
    with col5:
        st.metric("Snapshots", metrics.get('num_snapshots', 0))
    
    # Row 2: Market Impact Analysis
    st.subheader("💹 Market Impact Simulation")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        impact_size = st.number_input("Order Size ($)", min_value=1000, value=50000, step=10000)
        impact_side = st.selectbox("Order Side", ["buy", "sell"])
        
        if st.button("Calculate Impact"):
            impact = analyzer.simulate_market_impact(impact_size, impact_side)
            if impact:
                st.session_state['impact_result'] = impact
    
    with col2:
        if 'impact_result' in st.session_state:
            impact = st.session_state['impact_result']
            st.info(f"""
            **Market Impact Analysis**
            - **Avg Price:** ${impact['average_price']:.2f}
            - **Impact:** {impact['impact_bps']:.1f} bps
            - **Levels Consumed:** {impact['levels_consumed']}
            - **Executed Size:** {impact['executed_size']:,}
            - **Fill Rate:** {impact['fill_rate']:.1%}
            """)
    
    # Row 3: L2 Depth Visualization
    st.subheader("📚 L2 Order Book Depth")
    
    latest_depth = analyzer.get_latest_depth()
    if latest_depth:
        # Create depth chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Bid Side Depth", "Ask Side Depth"),
            shared_yaxes=False
        )
        
        # Bid side
        bids = latest_depth['bids'][:10]
        bid_prices = [b['price'] for b in bids]
        bid_sizes = [b['size'] for b in bids]
        bid_orders = [b['orders'] for b in bids]
        
        fig.add_trace(
            go.Bar(x=bid_prices, y=bid_sizes, name='Bid Size', marker_color='green',
                   text=[f"{o} orders" for o in bid_orders], textposition='outside'),
            row=1, col=1
        )
        
        # Ask side
        asks = latest_depth['asks'][:10]
        ask_prices = [a['price'] for a in asks]
        ask_sizes = [a['size'] for a in asks]
        ask_orders = [a['orders'] for a in asks]
        
        fig.add_trace(
            go.Bar(x=ask_prices, y=ask_sizes, name='Ask Size', marker_color='red',
                   text=[f"{o} orders" for o in ask_orders], textposition='outside'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title_text="Price", row=1, col=1)
        fig.update_xaxes(title_text="Price", row=1, col=2)
        fig.update_yaxes(title_text="Size", row=1, col=1)
        fig.update_yaxes(title_text="Size", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display depth metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Bid Depth", f"{latest_depth['total_bid_size']:,}")
        with col2:
            st.metric("Total Ask Depth", f"{latest_depth['total_ask_size']:,}")
        with col3:
            st.metric("Depth Imbalance", f"{latest_depth['depth_imbalance']:.3f}")
    
    # Row 4: Spread and Depth Trends
    st.subheader("📈 Spread & Depth Trends")
    
    # Extract spread and depth trends
    if analyzer.quotes is not None:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=("Bid-Ask Spread Over Time", "Mid Price"),
                           vertical_spacing=0.1)
        
        # Spread
        fig.add_trace(
            go.Scatter(x=analyzer.quotes.index, y=analyzer.quotes['spread_bps'],
                      mode='lines', name='Spread (bps)', line=dict(color='orange')),
            row=1, col=1
        )
        
        # Price
        fig.add_trace(
            go.Scatter(x=analyzer.quotes.index, y=analyzer.quotes['mid_price'],
                      mode='lines', name='Mid Price', line=dict(color='blue')),
            row=2, col=1
        )
        
        fig.update_layout(height=400, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 5: Order Book Analytics
    st.subheader("🔍 Order Book Analytics")
    
    analytics = analyzer.calculate_order_book_analytics()
    if analytics:
        # Convert to DataFrame for display
        analytics_df = pd.DataFrame(analytics).T
        st.dataframe(analytics_df.style.format({
            'top_5_bid_depth': '{:,.0f}',
            'top_5_ask_depth': '{:,.0f}',
            'bid_wall_size': '{:,.0f}',
            'ask_wall_size': '{:,.0f}',
            'spread_bps': '{:.2f}',
            'depth_imbalance': '{:+.3f}'
        }))
        
        # Wall detection summary
        if 'bid_wall_size' in analytics_df.columns and 'ask_wall_size' in analytics_df.columns:
            max_bid_wall = analytics_df['bid_wall_size'].max()
            max_ask_wall = analytics_df['ask_wall_size'].max()
            if max_bid_wall > 5000:
                st.warning(f"⚠️ Large Bid Wall Detected: {max_bid_wall:,.0f} units at one level")
            if max_ask_wall > 5000:
                st.warning(f"⚠️ Large Ask Wall Detected: {max_ask_wall:,.0f} units at one level")
    
    # Row 6: Recent Trade Activity
    st.subheader("🔄 Recent Trade Activity")
    if analyzer.trade_sequence:
        trades_df = pd.DataFrame(analyzer.trade_sequence)
        trades_df['side'] = trades_df['side'].apply(lambda x: '🟢 Buy' if x == 'buy' else '🔴 Sell')
        st.dataframe(trades_df.tail(20))
        
        # Trade summary
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
    st.info("👈 Configure the parameters in the sidebar and click 'Run Depth Analysis' to start.")
    st.markdown("""
    ### 🎯 How This Market Depth Model Works
    
    This integrated system combines key features from professional market depth analysis projects [citation:1][citation:2]:
    
    1.  **L2/L3 Depth Simulation**: Generates realistic order book data with bid-ask spreads, depth levels, and size clustering [citation:1].
    2.  **Spread Modeling**: Supports multiple spread models (volatility-linked, volume-sensitive, time-of-day) [citation:1].
    3.  **Order Book Analytics**: Calculates depth imbalance, liquidity at key levels, and wall detection [citation:2].
    4.  **Market Impact Simulation**: Simulates execution costs for large orders using VWAP slippage analysis [citation:1][citation:6].
    5.  **Professional Visualizations**: Displays L2 depth charts, spread trends, and trade activity.
    
    ### 🔧 Technologies Used
    
    - **Streamlit** for the interactive dashboard
    - **Plotly** for professional charts and depth visualizations
    - **conflux-depthsim** (optional) for professional market depth simulation [citation:1]
    - **yfinance** for real market data
    - **Pandas/NumPy** for data processing and analytics
    
    ### 💡 Next Steps
    
    - Install `conflux-depthsim` for full professional simulation capabilities: `pip install conflux-depthsim`
    - Connect to live exchange WebSocket feeds for real-time order book analysis [citation:2][citation:12]
    - Integrate machine learning for depth-based trading signal generation [citation:9]
    - Add historical depth persistence with SQLite [citation:2]
    """)
