# main_bot.py – 8‑indicator strategy on 1‑hour bars (Yahoo Finance data)
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
import yfinance as yf

from alpaca.trading.client import TradingClient
from allocation import PositionAllocator
from safety import SafetyNet
from broker import BrokerConnection

# ------------------------------
# Telegram alerts (replace with your token and chat ID)
# ------------------------------
def send_telegram_message(message):
    bot_token = "8954699344:AAG_d5zazERDqhmq-j4CqseYMrYSX8G3__s"
    chat_id = "1217871917"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code != 200:
            print(f"Telegram error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")

# ------------------------------
# Keep‑alive web server (for Render)
# ------------------------------
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Bot is alive')

def run_health_server():
    port = int(os.environ.get('PORT', 8000))
    httpd = HTTPServer(('0.0.0.0', port), HealthHandler)
    httpd.serve_forever()

Thread(target=run_health_server, daemon=True).start()
print(f"Health check server running on port {os.environ.get('PORT', 8000)}")

# ------------------------------
# Environment variables
# ------------------------------
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing API keys. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY.")

# ------------------------------
# Alpaca trading client (only for orders)
# ------------------------------
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
broker = BrokerConnection(API_KEY, SECRET_KEY, is_paper=True)

# ------------------------------
# Pure pandas indicator functions (no 'ta')
# ------------------------------
def rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def bollinger_bands(close, window=20, num_std=2):
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    lower_band = rolling_mean - (rolling_std * num_std)
    return lower_band

def aroon(high, low, window=25):
    aroon_up = 100 * high.rolling(window=window+1).apply(lambda x: x.argmax()) / window
    aroon_down = 100 * low.rolling(window=window+1).apply(lambda x: x.argmin()) / window
    return aroon_up, aroon_down

def stoch_rsi(close, period=14, smooth=3):
    rsi_vals = rsi(close, period)
    stochrsi = (rsi_vals - rsi_vals.rolling(period).min()) / (rsi_vals.rolling(period).max() - rsi_vals.rolling(period).min())
    stochrsi_k = stochrsi.rolling(smooth).mean()
    stochrsi_d = stochrsi_k.rolling(smooth).mean()
    return stochrsi_d

def ema(close, period):
    return close.ewm(span=period, adjust=False).mean()

def obv(close, volume):
    obv_vals = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv_vals

def tdi(close, rsi_period=13, green_period=2, red_period=7):
    rsi_vals = rsi(close, rsi_period)
    tdi_green = rsi_vals.rolling(green_period).mean()
    tdi_red = tdi_green.rolling(red_period).mean()
    return tdi_green, tdi_red

def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# ------------------------------
# Compute all 8 signals
# ------------------------------
def compute_signals(df):
    """Adds boolean signal columns to df (True = buy)."""
    df = df.copy()
    # 1. RSI < 35
    df['rsi'] = rsi(df['close'], 14)
    df['sig_rsi'] = df['rsi'] < 35

    # 2. MACD bullish: MACD line > signal line
    macd_line, signal_line = macd(df['close'])
    df['sig_macd'] = macd_line > signal_line

    # 3. Bollinger lower band touch: close < lower band
    lower_band = bollinger_bands(df['close'], 20, 2)
    df['sig_bb'] = df['close'] < lower_band

    # 4. AROON: up > 70 and down < 30
    aroon_up, aroon_down = aroon(df['high'], df['low'], 25)
    df['sig_aroon'] = (aroon_up > 70) & (aroon_down < 30)

    # 5. Stochastic RSI < 20
    stoch_d = stoch_rsi(df['close'], 14, 3)
    df['sig_stoch'] = stoch_d < 20

    # 6. EMA 9 > EMA 21
    ema9 = ema(df['close'], 9)
    ema21 = ema(df['close'], 21)
    df['sig_ema'] = ema9 > ema21

    # 7. OBV rising: current OBV > OBV 5 bars ago
    obv_vals = obv(df['close'], df['volume'])
    df['sig_obv'] = obv_vals > obv_vals.shift(5)

    # 8. TDI simplified: green > red
    tdi_green, tdi_red = tdi(df['close'], 13, 2, 7)
    df['sig_tdi'] = tdi_green > tdi_red

    return df

# ------------------------------
# Fetch 1‑hour data from Yahoo Finance
# ------------------------------
def get_1h_bars_yf(symbol, period="60d"):
    """Fetch 1‑hour bars from Yahoo Finance (max 60 days)."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="1h")
    if df.empty:
        raise ValueError(f"No 1‑hour data for {symbol}")
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    return df

print("Fetching 1‑hour SPY data from Yahoo Finance...")
df_1h = get_1h_bars_yf('SPY', period="60d")
print(f"Loaded {len(df_1h)} 1‑hour bars")

# Compute indicators and signals
df_1h = compute_signals(df_1h)
df_1h['atr'] = atr(df_1h['high'], df_1h['low'], df_1h['close'], 14)
df_1h = df_1h.dropna().copy()

# ------------------------------
# Initialize bot components
# ------------------------------
print("Initializing Trading Bot (8‑indicator, 1‑hour, Yahoo Finance data)...")
account = trading_client.get_account()
initial_value = float(account.portfolio_value)
print(f"Initial portfolio value: ${initial_value:,.2f}")

safety_net = SafetyNet(initial_portfolio_value=initial_value)

send_telegram_message("🤖 8‑indicator bot (1‑hour, Yahoo) started. Monitoring SPY.")

# Trade state
trade_state = {}

# ------------------------------
# Main loop – check for new 1‑hour bars every 60 seconds
# ------------------------------
print("Starting main loop (checks every 60 seconds).\n")
last_datetime = df_1h.index[-1]

while True:
    try:
        # Fetch latest 1‑hour bars (last 5 days)
        new_data = get_1h_bars_yf('SPY', period="5d")
        if new_data.empty:
            print("No new data yet.")
            time.sleep(60)
            continue

        latest_dt = new_data.index[-1]
        if latest_dt > last_datetime:
            # New bar detected
            print(f"New 1‑hour bar detected: {latest_dt}")
            new_row = new_data.iloc[-1:].copy()
            df_1h = pd.concat([df_1h, new_row])
            # Recompute signals for the whole frame
            df_1h = compute_signals(df_1h)
            df_1h['atr'] = atr(df_1h['high'], df_1h['low'], df_1h['close'], 14)
            df_1h = df_1h.dropna().copy()
            last_datetime = latest_dt

        # Get latest bar values
        last = df_1h.iloc[-1]
        close = last['close']
        atr_val = last['atr']
        if pd.isna(atr_val):
            atr_val = 1.0

        # Count signals (threshold=3)
        signal_count = 0
        signal_names = []
        for col in ['sig_rsi', 'sig_macd', 'sig_bb', 'sig_aroon', 'sig_stoch', 'sig_ema', 'sig_obv', 'sig_tdi']:
            if col in df_1h.columns and last[col]:
                signal_count += 1
                signal_names.append(col[4:])
        print(f"{datetime.now().strftime('%H:%M:%S')} | Price: {close:.2f} | Signals: {signal_count}/8 | {signal_names}")

        # Get current SPY position from Alpaca
        positions = trading_client.get_all_positions()
        current_spy_shares = 0
        entry_price = None
        for pos in positions:
            if pos.symbol == "SPY":
                current_spy_shares = float(pos.qty)
                entry_price = float(pos.avg_entry_price)
                break

        # --- Entry logic (no position) ---
        if current_spy_shares == 0:
            if signal_count >= 3:
                stop_dist = 2 * atr_val
                if stop_dist > 0:
                    risk_percent = 0.01
                    equity = float(trading_client.get_account().equity)
                    size = max(1, int((equity * risk_percent) / stop_dist))
                else:
                    size = 1
                print(f"BUY signal: placing order for {size} shares of SPY")
                broker.submit_order("SPY", size, "buy")
                send_telegram_message(f"🚀 8‑indicator BUY {size} SPY @ {close:.2f}\nSignals: {signal_count}/8 ({', '.join(signal_names)})")
            else:
                print(f"No buy. Only {signal_count}/8 signals.")
        else:
            # --- Manage existing position (same as before) ---
            if 'SPY' not in trade_state and entry_price:
                trade_state['SPY'] = {
                    'entry_price': entry_price,
                    'initial_shares': current_spy_shares,
                    'tp1_hit': False,
                    'stop_price': entry_price - 2 * atr_val,
                    'highest_close': close,
                    'breakeven_activated': False
                }
                print(f"Initial stop set at ${trade_state['SPY']['stop_price']:.2f}")
                print(f"TP1 level: ${entry_price + 1.5 * atr_val:.2f}")
                send_telegram_message(f"📈 Trade opened: {current_spy_shares} SPY @ {entry_price:.2f}\nStop: ${trade_state['SPY']['stop_price']:.2f}")

            if 'SPY' in trade_state:
                ts = trade_state['SPY']
                if close > ts['highest_close']:
                    ts['highest_close'] = close
                    print(f"New highest close: ${close:.2f}")

                # Breakeven
                if not ts.get('breakeven_activated', False) and close >= ts['entry_price'] + atr_val:
                    ts['stop_price'] = ts['entry_price']
                    ts['breakeven_activated'] = True
                    print(f"Breakeven stop activated at ${ts['stop_price']:.2f}")
                    send_telegram_message(f"🔒 Breakeven stop for SPY (entry ${ts['entry_price']:.2f})")

                # Trailing
                if ts.get('breakeven_activated', False):
                    new_stop = ts['highest_close'] - 2 * atr_val
                    if new_stop > ts['stop_price']:
                        ts['stop_price'] = new_stop
                        print(f"Trailing stop raised to ${ts['stop_price']:.2f}")

                # Take profit 1
                tp1_price = ts['entry_price'] + 1.5 * atr_val
                if not ts['tp1_hit'] and close >= tp1_price:
                    shares_to_sell = max(1, int(current_spy_shares / 2))
                    if shares_to_sell > 0:
                        print(f"Take profit hit at ${close:.2f} – selling {shares_to_sell} shares")
                        broker.submit_order("SPY", shares_to_sell, "sell")
                        ts['tp1_hit'] = True
                        send_telegram_message(f"🎯 TP1 at ${close:.2f}. Sold {shares_to_sell} SPY. Remaining {current_spy_shares - shares_to_sell}.")

                # Stop loss
                if close <= ts['stop_price']:
                    print(f"Stop loss hit at ${close:.2f} – selling {current_spy_shares} shares")
                    broker.submit_order("SPY", current_spy_shares, "sell")
                    send_telegram_message(f"🛑 Stop loss at ${close:.2f}. Closed {current_spy_shares} SPY.")
                    del trade_state['SPY']
                    time.sleep(60)
                    continue

        # Update safety net
        account = trading_client.get_account()
        portfolio_value = float(account.portfolio_value)
        if safety_net.update_portfolio_value(portfolio_value):
            send_telegram_message("⚠️ Circuit breaker triggered. Trading halted.")
            break

        # Print holding status
        if current_spy_shares > 0 and 'SPY' in trade_state:
            ts = trade_state['SPY']
            print(f"Holding {current_spy_shares} shares. Stop: ${ts['stop_price']:.2f} | TP1: {'Hit' if ts['tp1_hit'] else 'Not hit'}")
        elif current_spy_shares > 0:
            print(f"Holding {current_spy_shares} shares (initial stop not set yet).")

        print("-" * 50)
        time.sleep(60)

    except KeyboardInterrupt:
        print("Bot stopped by user.")
        break
    except Exception as e:
        error_msg = f"Error: {e}"
        print(error_msg)
        send_telegram_message(f"⚠️ Bot error: {error_msg}")
        time.sleep(60)
