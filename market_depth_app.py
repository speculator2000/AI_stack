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

# Handle plotly import with fallback
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.warning("⚠️ Plotly not installed. Install with: pip install plotly")

from datetime import datetime, timedelta
import yfinance as yf

# Try to import depthsim; fallback to simulation if not installed
try:
    from depthsim import DepthSimulator
    HAS_DEPTHSIM = True
except ImportError:
    HAS_DEPTHSIM = False
    # Don't show warning here, will show in the sidebar

# --- 1. Configuration ---
st.set_page_config(page_title="Market Depth Analysis", layout="wide")
st.title("📊 Market Analysis Depth Model")
st.markdown("**Professional order book depth analysis with simulated L2/L3 market data**")

# Check for required packages
if not HAS_PLOTLY:
    st.error("""
    ❌ **Missing Required Package: plotly**
    
    Please install it using:
    ```bash
    pip install plotly
