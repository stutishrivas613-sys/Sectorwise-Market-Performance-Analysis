import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="Sector Performance Tracker", layout="wide")

# 1. Define Stock Universe (from your project)
STOCKS = {
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS", "LTTS.NS"],
    "BANKING": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS", "GODREJCP.NS", "COLPAL.NS", "UBL.NS", "TATACONSUM.NS"],
    "PHARMA": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS", "LUPIN.NS", "AUROPHARMA.NS", "TORNTPHARM.NS", "ALKEM.NS", "GLENMARK.NS"],
    "AUTO": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "ASHOKLEY.NS", "TVSMOTOR.NS", "ESCORTS.NS", "BALKRISIND.NS"],
    "ENERGY": ["RELIANCE.NS", "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "COALINDIA.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "ADANIGREEN.NS", "ADANIPOWER.NS"],
    "METAL": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "SAIL.NS", "JINDALSTEL.NS", "NMDC.NS", "MOIL.NS", "RATNAMANI.NS", "WELCORP.NS"]
}

@st.cache_data(ttl=3600)
def load_data():
    sector_data = {}
    for sector, tickers in STOCKS.items():
        df = yf.download(tickers, period="3y", group_by="ticker", auto_adjust=True)
        # Extract Close prices
        prices = pd.DataFrame()
        for t in tickers:
            if t in df.columns.get_level_values(0):
                prices[t] = df[t]['Close']
        sector_data[sector] = prices.ffill().bfill()
    return sector_data

# Data Loading
with st.spinner("Fetching latest market data..."):
    all_data = load_data()

# Calculate Sector Indices (Mean of stocks in sector)
sector_index_df = pd.DataFrame({s: df.mean(axis=1) for s, df in all_data.items()})

# Calculate Metrics
returns = sector_index_df.pct_change().dropna()
volatility = returns.std() * np.sqrt(252)
risk_free_rate = 0.05
sharpe = (returns.mean() * 252 - risk_free_rate) / volatility
years = (sector_index_df.index[-1] - sector_index_df.index[0]).days / 365
cagr = (sector_index_df.iloc[-1] / sector_index_df.iloc[0])**(1/years) - 1

summary = pd.DataFrame({"CAGR": cagr, "Volatility": volatility, "Sharpe Ratio": sharpe}).sort_values("Sharpe Ratio", ascending=False)

# UI Logic
st.title("📈 Sector-wise Market Performance Analysis")
st.markdown("This dashboard visualizes risk and returns across different market sectors over the last 3 years.")

# Sidebar
menu = st.sidebar.selectbox("Select View", ["Overall Market Summary", "Sector Deep-Dive"])

if menu == "Overall Market Summary":
    st.header("Overall Performance Metrics")
    
    # Top Metrics Columns
    col1, col2, col3 = st.columns(3)
    best_sector = summary.index[0]
    col1.metric("Top Performing Sector", best_sector)
    col2.metric("Highest CAGR", f"{summary.loc[best_sector, 'CAGR']:.2%}")
    col3.metric("Avg Market Volatility", f"{summary['Volatility'].mean():.2%}")

    # Cumulative Returns Plot
    st.subheader("Sector Growth (Cumulative Returns)")
    cum_returns = (1 + returns).cumprod()
    fig_line = px.line(cum_returns, labels={'value': 'Growth Multiplier', 'Date': 'Timeline'})
    st.plotly_chart(fig_line, use_container_width=True)

    # Risk vs Return Scatter
    st.subheader("Risk vs. Return Profile")
    fig_scatter = px.scatter(summary.reset_index(), x="Volatility", y="CAGR", text="index", 
                             size="Sharpe Ratio", color="Sharpe Ratio",
                             title="CAGR vs Volatility (Bubble size = Sharpe Ratio)")
    st.plotly_chart(fig_scatter, use_container_width=True)

elif menu == "Sector Deep-Dive":
    selected_sector = st.sidebar.selectbox("Choose a Sector", list(STOCKS.keys()))
    st.header(f"Sector Analysis: {selected_sector}")
    
    # Sector Specific Table
    sector_prices = all_data[selected_sector]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("Summary Statistics")
        st.dataframe(summary.loc[[selected_sector]].style.format("{:.2%}"))
        
    with col2:
        # Comparison of stocks within that sector
        stock_cum_returns = (sector_prices / sector_prices.iloc[0])
        fig_sector = px.line(stock_cum_returns, title=f"Stock Performance within {selected_sector}")
        st.plotly_chart(fig_sector, use_container_width=True)

    st.subheader("Raw Data Preview")
    st.dataframe(sector_prices.tail(10))