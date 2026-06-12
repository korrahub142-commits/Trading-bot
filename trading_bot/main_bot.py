# main_bot.py – 8‑indicator strategy on 1‑hour bars (Alpaca data)
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
import ta

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

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
# Alpaca clients
# ------------------------------
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# ------------------------------
# 8‑indicator calculation function (same as before)
# ------------------------------
def compute_indicators(df):
    """Returns a DataFrame with 8 boolean signals (True = buy)."""
    df = df.copy()
    # 1. RSI < 35
    rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    rsi_sig = rsi < 35
    # 2. MACD bullish crossover
    macd = ta.trend.MACD(df['close'])
    macd_sig = macd.macd() > macd.macd_signal()
    # 3. Bollinger lower band touch
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    bb_sig = df['close'] < bb.bollinger_lband()
    # 4. AROON up >70 and down <30
    aroon = ta.trend.AroonIndicator(df['high'], df['low'], window=25)
    aroon_sig = (aroon.aroon_up() > 70) & (aroon.aroon_down() < 30)
    # 5. Stochastic RSI <20
    stoch = ta.momentum.StochRSIIndicator(df['close'], window=14)
    stoch_sig = stoch.stochrsi_d() < 20
    # 6. EMA 9 > EMA 21
    ema9 = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    ema_sig = ema9 > ema21
    # 7. OBV rising (5‑bar lookback)
    obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    obv_sig = obv > obv.shift(5)
    # 8. TDI simplified
    rsi13 = ta.momentum.RSIIndicator(df['close'], window=13).rsi()
    tdi_green = rsi13.rolling(2).mean()
    tdi_red = tdi_green.rolling(7).mean()
    tdi_sig = tdi_green > tdi_red

    signals = pd.DataFrame({
        'rsi': rsi_sig,
        'macd': macd_sig,
        'bb': bb_sig,
        'aroon': aroon_sig,
        'stoch': stoch_sig,
        'ema': ema_sig,
        'obv': obv_sig,
        'tdi': tdi_sig
    })
    return signals

# ------------------------------
# Load initial 1‑hour historical data from Alpaca
# ------------------------------
def get_1h_bars(symbol, days_back=30):
    end = datetime.now()
    start = end - timedelta(days=days_back)
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Hour,
        start=start,
        end=end,
        limit=10000
    )
    bars = data_client.get_stock_bars(request)
    df = bars.df
    if df.empty:
        raise ValueError(f"No 1‑hour data for {symbol}")
    df = df.reset_index()
    df = df.rename(columns={'timestamp': 'datetime', 'open': 'open', 'high': 'high',
                            'low': 'low', 'close': 'close', 'volume': 'volume'})
    df.set_index('datetime', inplace=True)
    return df

print("Fetching 1‑hour SPY data from Alpaca...")
df_1h = get_1h_bars('SPY', days_back=90)  # ~90 days of 1‑hour bars
print(f"Loaded {len(df_1h)} 1‑hour bars")

# Pre‑compute indicators on the full DataFrame
signals = compute_indicators(df_1h)
for col in signals.columns:
    df_1h[f'sig_{col}'] = signals[col]

# Compute ATR (14 periods)
atr_ind = ta.volatility.AverageTrueRange(high=df_1h['high'], low=df_1h['low'],
                                         close=df_1h['close'], window=14)
df_1h['atr'] = atr_ind.average_true_range()
df_1h = df_1h.dropna().copy()

# ------------------------------
# Initialize bot components
# ------------------------------
print("Initializing Trading Bot (8‑indicator, 1‑hour)...")
broker = BrokerConnection(API_KEY, SECRET_KEY, is_paper=True)
account = trading_client.get_account()
initial_value = float(account.portfolio_value)
print(f"Initial portfolio value: ${initial_value:,.2f}")

safety_net = SafetyNet(initial_portfolio_value=initial_value)

send_telegram_message("🤖 8‑indicator bot (1‑hour) started. Monitoring SPY.")

# Trade state
trade_state = {}

# ------------------------------
# Main loop – check for new 1‑hour bars every 60 seconds
# ------------------------------
print("Starting main loop (checks every 60 seconds).\n")
last_datetime = df_1h.index[-1]

while True:
    try:
        # Fetch latest 1‑hour bars (last 5 days to catch any new bar)
        new_bars = get_1h_bars('SPY', days_back=5)
        if new_bars.empty:
            print("No new data yet.")
            time.sleep(60)
            continue

        latest_dt = new_bars.index[-1]
        if latest_dt > last_datetime:
            # New bar appeared – add it to our DataFrame
            print(f"New 1‑hour bar detected: {latest_dt}")
            new_row = new_bars.iloc[-1:].copy()
            df_1h = pd.concat([df_1h, new_row])
            # Recompute indicators for the new row (simpler: recompute whole frame)
            signals_new = compute_indicators(df_1h)
            for col in signals_new.columns:
                df_1h[f'sig_{col}'] = signals_new[col]
            atr_new = ta.volatility.AverageTrueRange(high=df_1h['high'], low=df_1h['low'],
                                                     close=df_1h['close'], window=14).average_true_range()
            df_1h['atr'] = atr_new
            df_1h = df_1h.dropna().copy()
            last_datetime = latest_dt

        # Get latest bar values
        last = df_1h.iloc[-1]
        close = last['close']
        high = last['high']
        low = last['low']
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
            if signal_count >= 3:  # threshold = 3
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
                # trade_state will be initialised on next loop when position appears
            else:
                print(f"No buy. Only {signal_count}/8 signals.")
        else:
            # --- Manage existing position ---
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

                # Take profit 1 (sell half at entry + 1.5*ATR)
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
