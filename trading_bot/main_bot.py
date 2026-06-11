# main_bot.py
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import yfinance as yf
import requests

from market_brain import MarketBrain
from allocation import PositionAllocator
from safety import SafetyNet
from broker import BrokerConnection

# --- Telegram alert function ---
def send_telegram_message(message):
    if not message:
        print("WARNING: Attempted to send empty Telegram message")
        return
    bot_token = "8954699344:AAG_d5zazERDqhmq-j4CqseYMrYSX8G3__s"
    chat_id = "993606490"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code != 200:
            print(f"Telegram error {response.status_code}: {response.text}")
        else:
            print("Telegram message sent successfully")
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")

# --- Keep-alive web server ---
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

# Get API keys
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing API keys. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY as environment variables.")

print("Initializing Trading Bot...")
broker = BrokerConnection(API_KEY, SECRET_KEY, is_paper=True)
account = broker.get_account_info()
initial_value = float(account.portfolio_value)
print(f"Initial portfolio value: ${initial_value:,.2f}")

brain = MarketBrain(n_states=5)
safety_net = SafetyNet(initial_portfolio_value=initial_value)

# Fetch live historical data
print("Fetching live historical data (SPY daily bars from Yahoo Finance)...")
end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)
spy = yf.Ticker("SPY")
historical_df = spy.history(start=start_date, end=end_date, interval="1d")
if historical_df.empty:
    raise ValueError("No data received from Yahoo Finance")
historical_df = historical_df[['Close', 'High', 'Low']].rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low'})
print(f"Training data: {len(historical_df)} days")

brain.train(historical_df)

# Send startup alert
send_telegram_message("🤖 Trading bot started. Monitoring market...")

# Store trade state per symbol
trade_state = {}

print("Starting main trading loop. Will check every 60 seconds.\n")
while True:
    try:
        # Update with latest daily bar
        new_data = spy.history(period="5d", interval="1d")
        if not new_data.empty:
            latest_close = new_data['Close'].iloc[-1]
            latest_high = new_data['High'].iloc[-1]
            latest_low = new_data['Low'].iloc[-1]
            latest_date = new_data.index[-1]
            if latest_date not in historical_df.index:
                new_row = pd.DataFrame({'close': [latest_close], 'high': [latest_high], 'low': [latest_low]}, index=[latest_date])
                historical_df = pd.concat([historical_df, new_row])
                print(f"Added new data point: {latest_date.date()}")

        # Market regime & RSI
        current_market_state = brain.predict_current_market(historical_df)
        print(f"Market is currently: {current_market_state}")
        rsi_value = brain.get_rsi(historical_df)
        print(f"RSI(14): {rsi_value:.2f}")

        # Current ATR
        atr = brain.get_atr(historical_df)
        print(f"ATR(14): {atr:.2f}")

        # Account value
        account = broker.get_account_info()
        portfolio_value = float(account.portfolio_value)
        print(f"Portfolio value: ${portfolio_value:,.2f}")

        # Safety net
        if safety_net.update_portfolio_value(portfolio_value):
            send_telegram_message("⚠️ Circuit breaker triggered. Trading halted.")
            break

        # Get current SPY position
        positions = broker.trading_client.get_all_positions()
        current_spy_shares = 0
        entry_price = None
        for pos in positions:
            if pos.symbol == "SPY":
                current_spy_shares = float(pos.qty)
                entry_price = float(pos.avg_entry_price)
                break

        # --- Trailing stop & take profit management (if position exists) ---
        if current_spy_shares > 0 and entry_price:
            current_price = historical_df['close'].iloc[-1]
            # Initialize trade state if not present
            if 'SPY' not in trade_state:
                trade_state['SPY'] = {
                    'entry_price': entry_price,
                    'initial_shares': current_spy_shares,
                    'tp1_hit': False,
                    'stop_price': entry_price - 2 * atr,
                    'highest_close': current_price,
                    'breakeven_activated': False
                }
                print(f"Initial stop set at ${trade_state['SPY']['stop_price']:.2f}")
                print(f"Take profit level set at ${entry_price + 1.5 * atr:.2f} (1.5×ATR)")
                send_telegram_message(f"📈 New trade: bought {current_spy_shares} SPY @ ${entry_price:.2f}\nStop: ${trade_state['SPY']['stop_price']:.2f}\nTP1: ${entry_price + 1.5 * atr:.2f}")

            ts = trade_state['SPY']
            # Update highest close
            if current_price > ts['highest_close']:
                ts['highest_close'] = current_price
                print(f"New highest close: ${current_price:.2f}")

            # Breakeven
            if not ts.get('breakeven_activated', False) and current_price >= ts['entry_price'] + atr:
                ts['stop_price'] = ts['entry_price']
                ts['breakeven_activated'] = True
                print(f"Breakeven stop activated at ${ts['stop_price']:.2f}")
                send_telegram_message(f"🔒 Breakeven stop activated for SPY (entry: ${ts['entry_price']:.2f})")

            # Trailing
            if ts.get('breakeven_activated', False):
                new_stop = ts['highest_close'] - 2 * atr
                if new_stop > ts['stop_price']:
                    ts['stop_price'] = new_stop
                    print(f"Trailing stop raised to ${ts['stop_price']:.2f}")

            # Take profit
            tp1_price = ts['entry_price'] + 1.5 * atr
            if not ts['tp1_hit'] and current_price >= tp1_price:
                shares_to_sell = int(current_spy_shares / 2)
                if shares_to_sell > 0:
                    print(f"Take profit hit at ${current_price:.2f} – selling {shares_to_sell} shares (half position)")
                    broker.submit_order("SPY", shares_to_sell, "sell")
                    ts['tp1_hit'] = True
                    send_telegram_message(f"🎯 Take profit 1 hit at ${current_price:.2f}\nSold {shares_to_sell} SPY. Remaining {current_spy_shares - shares_to_sell} shares.")
                    current_spy_shares -= shares_to_sell

            # Stop loss
            if current_price <= ts['stop_price']:
                print(f"Stop loss hit at ${current_price:.2f} – selling {current_spy_shares} shares")
                broker.submit_order("SPY", current_spy_shares, "sell")
                send_telegram_message(f"🛑 Stop loss hit at ${current_price:.2f}\nSold {current_spy_shares} SPY. Trade closed.")
                del trade_state['SPY']
                time.sleep(60)
                continue

        # --- Entry logic (only if no position) ---
        if current_spy_shares == 0:
            close_prices = historical_df['close'].dropna()
            if len(close_prices) > 20:
                returns = np.log(close_prices / close_prices.shift(1)).dropna()
                volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            else:
                volatility = 0.02
            allocator = PositionAllocator(portfolio_value=portfolio_value)
            target_shares = int(allocator.calculate_position_size(current_market_state, volatility))
            print(f"Target shares to buy: {target_shares}")

            if current_market_state in ["Bull", "Euphoria"] and target_shares > 0 and rsi_value < 70:
                print(f"RSI {rsi_value:.2f} is below 70 – Placing BUY order for {target_shares} shares of SPY")
                broker.submit_order("SPY", target_shares, "buy")
                send_telegram_message(f"🚀 Buying {target_shares} SPY\nMarket regime: {current_market_state}\nRSI: {rsi_value:.2f}\nATR: {atr:.2f}")
            else:
                print(f"No buy. RSI = {rsi_value:.2f}")
        else:
            ts = trade_state.get('SPY', {})
            stop_price = ts.get('stop_price', 'N/A')
            tp1_status = "Hit" if ts.get('tp1_hit', False) else "Not hit"
            print(f"Holding {current_spy_shares} shares. Stop: ${stop_price} | TP1: {tp1_status}")

        print("-" * 50)
        time.sleep(60)

    except KeyboardInterrupt:
        print("Shutting down bot.")
        break
    except Exception as e:
        error_msg = f"Error: {e}"
        print(error_msg)
        send_telegram_message(f"⚠️ Bot error: {error_msg}")
        time.sleep(60)
