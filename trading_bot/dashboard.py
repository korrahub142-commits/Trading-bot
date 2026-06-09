import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os
from alpaca.trading.client import TradingClient

# Explicitly load .env from the current directory
load_dotenv(dotenv_path='.env')

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

# If keys are missing, show error and stop
if not API_KEY or not SECRET_KEY:
    st.error("Missing API keys. Please check your .env file.")
    st.stop()

try:
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
except Exception as e:
    st.error(f"Failed to connect to Alpaca: {e}")
    st.stop()

st.set_page_config(page_title="Live Trading Bot Dashboard", layout="wide")
st.title("🤖 Live Trading Bot Dashboard")

def get_account_data():
    acc = trading_client.get_account()
    return {
        "portfolio_value": float(acc.portfolio_value),
        "buying_power": float(acc.buying_power),
        "cash": float(acc.cash)
    }

def get_positions():
    positions = trading_client.get_all_positions()
    if not positions:
        return pd.DataFrame()
    data = []
    for p in positions:
        data.append({
            "Symbol": p.symbol,
            "Quantity": float(p.qty),
            "Market Value": float(p.market_value),
            "Unrealized P/L": float(p.unrealized_pl),
            "Current Price": float(p.current_price)
        })
    return pd.DataFrame(data)

def get_recent_orders():
    # Fetch all orders (no arguments to avoid version issues)
    orders = trading_client.get_orders()
    if not orders:
        return pd.DataFrame()
    # Take last 10, newest first
    orders_list = list(orders)[-10:][::-1]
    filled_orders = [o for o in orders_list if o.status == "filled"]
    if not filled_orders:
        return pd.DataFrame()
    data = []
    for o in filled_orders:
        fill_time = o.filled_at_qty[0].filled_at if o.filled_at_qty else o.submitted_at
        data.append({
            "Time": fill_time,
            "Symbol": o.symbol,
            "Side": o.side.value,
            "Quantity": float(o.filled_qty),
            "Price": float(o.filled_avg_price)
        })
    return pd.DataFrame(data)

st.sidebar.header("Auto-refresh")
if st.sidebar.button("Refresh Now"):
    st.rerun()

account = get_account_data()
col1, col2, col3 = st.columns(3)
col1.metric("Portfolio Value", f"${account['portfolio_value']:,.2f}")
col2.metric("Buying Power", f"${account['buying_power']:,.2f}")
col3.metric("Cash", f"${account['cash']:,.2f}")

st.subheader("Current Positions")
pos_df = get_positions()
if not pos_df.empty:
    st.dataframe(pos_df, use_container_width=True)
else:
    st.info("No open positions.")

st.subheader("Recent Filled Orders")
orders_df = get_recent_orders()
if not orders_df.empty:
    st.dataframe(orders_df, use_container_width=True)
else:
    st.info("No recent orders.")

st.subheader("Live Logs")
st.code("Run main_bot.py in a separate terminal to see real-time decisions.")
