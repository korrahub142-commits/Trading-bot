# main_bot.py – 8‑ETF multi‑symbol, only one active trade at a time
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
# Keep‑alive web server
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

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
broker = BrokerConnection(API_KEY, SECRET_KEY, is_paper=True)

# ------------------------------
# Pure pandas indicator functions
# ------------------------------
def rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def bollinger_bands(close, window=20, num_std=2):
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    return rolling_mean - (rolling_std * num_std)

def aroon(high, low, window=25):
    aroon_up = 100 * high.rolling(window=window+1).apply(lambda x: x.argmax()) / window
    aroon_down = 100 * low.rolling(window=window+1).apply(lambda x: x.argmin()) / window
    return aroon_up, aroon_down

def stoch_rsi(close, period=14, smooth=3):
    rsi_vals = rsi(close, period)
    stochrsi = (rsi_vals - rsi_vals.rolling(period).min()) / (rsi_vals.rolling(period).max() - rsi_vals.rolling(period).min())
    return stochrsi.rolling(smooth).mean().rolling(smooth).mean()

def ema(close, period):
    return close.ewm(span=period, adjust=False).mean()

def obv(close, volume):
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

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
    return tr.rolling(period).mean()

# ------------------------------
# Compute all 8 signals
# ------------------------------
def compute_signals(df):
    df = df.copy()
    df['rsi'] = rsi(df['close'], 14)
    df['sig_rsi'] = df['rsi'] < 35
    macd_line, signal_line = macd(df['close'])
    df['sig_macd'] = macd_line > signal_line
    lower_band = bollinger_bands(df['close'], 20, 2)
    df['sig_bb'] = df['close'] < lower_band
    aroon_up, aroon_down = aroon(df['high'], df['low'], 25)
    df['sig_aroon'] = (aroon_up > 70) & (aroon_down < 30)
    stoch_d = stoch_rsi(df['close'], 14, 3)
    df['sig_stoch'] = stoch_d < 20
    ema9 = ema(df['close'], 9)
    ema21 = ema(df['close'], 21)
    df['sig_ema'] = ema9 > ema21
    obv_vals = obv(df['close'], df['volume'])
    df['sig_obv'] = obv_vals > obv_vals.shift(5)
    tdi_green, tdi_red = tdi(df['close'], 13, 2, 7)
    df['sig_tdi'] = tdi_green > tdi_red
    return df

# ------------------------------
# Fetch 1‑hour data from Yahoo Finance
# ------------------------------
def get_1h_bars_yf(symbol, period="60d"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="1h")
    if df.empty:
        raise ValueError(f"No 1‑hour data for {symbol}")
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    return df

# ------------------------------
# List of 8 ETFs
# ------------------------------
SYMBOLS = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLE', 'SMH']

# Load initial data for all symbols
dataframes = {}
print("Loading 1‑hour data for 8 ETFs...")
for sym in SYMBOLS:
    print(f"  {sym}...")
    df = get_1h_bars_yf(sym, period="60d")
    df = compute_signals(df)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df = df.dropna().copy()
    dataframes[sym] = df
    print(f"    {len(df)} bars")

# ------------------------------
# Initialize bot components
# ------------------------------
account = trading_client.get_account()
initial_value = float(account.portfolio_value)
print(f"Initial portfolio value: ${initial_value:,.2f}")
safety_net = SafetyNet(initial_portfolio_value=initial_value)

send_telegram_message(f"🤖 8‑ETF bot started (one active trade max). TP2 enabled. Symbols: {', '.join(SYMBOLS)}")

# Trade state per symbol (only one will be active at a time)
trade_state = {sym: {} for sym in SYMBOLS}
last_timestamp = {sym: dataframes[sym].index[-1] for sym in SYMBOLS}

# Global flag to ensure only one trade at a time
has_active_position = False
active_symbol = None

# ------------------------------
# Main loop – check all symbols every 60 seconds
# ------------------------------
print("Starting main loop (checks every 60 seconds). Only one trade at a time.\n")
while True:
    try:
        # First, update global position status from Alpaca
        positions = trading_client.get_all_positions()
        current_positions = [p.symbol for p in positions if float(p.qty) != 0]
        if current_positions:
            has_active_position = True
            active_symbol = current_positions[0]  # only one allowed
        else:
            has_active_position = False
            active_symbol = None
            # Also reset trade_state for all symbols that are not in a trade (clean up stale data)
            for sym in SYMBOLS:
                if trade_state[sym]:
                    trade_state[sym] = {}

        for sym in SYMBOLS:
            # Skip if we already have an active position and it's not this symbol (we only manage the active one)
            if has_active_position and active_symbol != sym:
                # For the active symbol, we still need to manage its exit
                if sym == active_symbol:
                    pass
                else:
                    continue

            # Fetch latest 1‑hour bars
            new_data = get_1h_bars_yf(sym, period="5d")
            if new_data.empty:
                continue
            latest_dt = new_data.index[-1]
            if latest_dt > last_timestamp[sym]:
                print(f"New 1‑hour bar for {sym}: {latest_dt}")
                new_row = new_data.iloc[-1:].copy()
                df = dataframes[sym]
                df = pd.concat([df, new_row])
                df = compute_signals(df)
                df['atr'] = atr(df['high'], df['low'], df['close'], 14)
                df = df.dropna().copy()
                dataframes[sym] = df
                last_timestamp[sym] = latest_dt

            # Get latest bar values
            last = dataframes[sym].iloc[-1]
            close = last['close']
            atr_val = last['atr']
            if pd.isna(atr_val):
                atr_val = 1.0

            # Count signals
            signal_count = 0
            signal_names = []
            for col in ['sig_rsi', 'sig_macd', 'sig_bb', 'sig_aroon', 'sig_stoch', 'sig_ema', 'sig_obv', 'sig_tdi']:
                if last[col]:
                    signal_count += 1
                    signal_names.append(col[4:])

            # Get current position details for this symbol (from Alpaca)
            pos_shares = 0
            pos_entry = None
            for p in positions:
                if p.symbol == sym:
                    pos_shares = float(p.qty)
                    pos_entry = float(p.avg_entry_price)
                    break

            # --- Manage existing position (if this symbol is the active one) ---
            if pos_shares > 0 and sym == active_symbol:
                # Initialize trade state if first time
                if not trade_state[sym]:
                    trade_state[sym] = {
                        'entry_price': pos_entry,
                        'initial_shares': pos_shares,
                        'tp1_hit': False,
                        'tp2_hit': False,
                        'shares_after_tp1': pos_shares,
                        'stop_price': pos_entry - 2 * atr_val,
                        'highest_close': close,
                        'breakeven_activated': False
                    }
                    print(f"{sym}: Initial stop set at ${trade_state[sym]['stop_price']:.2f}")
                    print(f"{sym}: TP1 level: ${pos_entry + 1.5 * atr_val:.2f}")
                    print(f"{sym}: TP2 level: ${pos_entry + 3 * atr_val:.2f}")
                    send_telegram_message(f"📈 Trade opened: {pos_shares} {sym} @ {pos_entry:.2f}\nStop: ${trade_state[sym]['stop_price']:.2f}\nTP1: ${pos_entry + 1.5 * atr_val:.2f}\nTP2: ${pos_entry + 3 * atr_val:.2f}")

                ts = trade_state[sym]
                # Update highest close
                if close > ts['highest_close']:
                    ts['highest_close'] = close
                    print(f"{sym}: New highest close: ${close:.2f}")

                # Breakeven
                if not ts.get('breakeven_activated', False) and close >= ts['entry_price'] + atr_val:
                    ts['stop_price'] = ts['entry_price']
                    ts['breakeven_activated'] = True
                    print(f"{sym}: Breakeven stop activated at ${ts['stop_price']:.2f}")
                    send_telegram_message(f"🔒 {sym}: Breakeven stop (entry ${ts['entry_price']:.2f})")

                # Trailing
                if ts.get('breakeven_activated', False):
                    new_stop = ts['highest_close'] - 2 * atr_val
                    if new_stop > ts['stop_price']:
                        ts['stop_price'] = new_stop
                        print(f"{sym}: Trailing stop raised to ${ts['stop_price']:.2f}")

                # TP1
                tp1_price = ts['entry_price'] + 1.5 * atr_val
                if not ts.get('tp1_hit', False) and close >= tp1_price:
                    shares_to_sell = max(1, int(pos_shares / 2))
                    if shares_to_sell > 0:
                        print(f"{sym}: TP1 hit at ${close:.2f} – selling {shares_to_sell} shares")
                        broker.submit_order(sym, shares_to_sell, "sell")
                        ts['tp1_hit'] = True
                        ts['shares_after_tp1'] = pos_shares - shares_to_sell
                        send_telegram_message(f"🎯 {sym} TP1 at ${close:.2f}. Sold {shares_to_sell}. Remaining {ts['shares_after_tp1']}.")

                # TP2
                tp2_price = ts['entry_price'] + 3 * atr_val
                if ts.get('tp1_hit', False) and not ts.get('tp2_hit', False) and close >= tp2_price:
                    remaining = ts.get('shares_after_tp1', pos_shares)
                    shares_to_sell_2 = max(1, int(remaining / 2))
                    if shares_to_sell_2 > 0 and remaining > 0:
                        print(f"{sym}: TP2 hit at ${close:.2f} – selling {shares_to_sell_2} shares")
                        broker.submit_order(sym, shares_to_sell_2, "sell")
                        ts['tp2_hit'] = True
                        new_remaining = remaining - shares_to_sell_2
                        send_telegram_message(f"🎯🎯 {sym} TP2 at ${close:.2f}. Sold {shares_to_sell_2}. Remaining {new_remaining}.")

                # Stop loss
                if close <= ts['stop_price']:
                    print(f"{sym}: Stop loss hit at ${close:.2f} – selling {pos_shares} shares")
                    broker.submit_order(sym, pos_shares, "sell")
                    send_telegram_message(f"🛑 {sym} Stop loss at ${close:.2f}. Closed {pos_shares}.")
                    trade_state[sym] = {}
                    has_active_position = False
                    active_symbol = None
                    continue

            # --- Entry logic (only if no active position and signal_count >= 3) ---
            if not has_active_position and signal_count >= 3:
                stop_dist = 2 * atr_val
                if stop_dist > 0:
                    risk_percent = 0.01
                    equity = float(trading_client.get_account().equity)
                    size = max(1, int((equity * risk_percent) / stop_dist))
                else:
                    size = 1
                print(f"BUY signal for {sym}: placing order for {size} shares")
                broker.submit_order(sym, size, "buy")
                send_telegram_message(f"🚀 BUY {size} {sym} @ {close:.2f}\nSignals: {signal_count}/8 ({', '.join(signal_names)})")
                has_active_position = True
                active_symbol = sym
                # Trade state will be initialized on next loop when the position appears
                # Wait a moment before checking more symbols this iteration
                time.sleep(5)
                continue

            # Optional: print signal counts for debugging (verbose, uncomment if needed)
            # if signal_count > 0:
            #     print(f"{sym}: {signal_count}/8 signals {signal_names}")

        # Update safety net
        account = trading_client.get_account()
        portfolio_value = float(account.portfolio_value)
        if safety_net.update_portfolio_value(portfolio_value):
            send_telegram_message("⚠️ Circuit breaker triggered. Trading halted.")
            break

        if has_active_position and active_symbol:
            print(f"Active position: {active_symbol} (only one allowed)")
        else:
            print("No active position – scanning all 8 ETFs for new signals.")
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
