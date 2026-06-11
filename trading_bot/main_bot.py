# main_bot.py
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import yfinance as yf

from market_brain import MarketBrain
from allocation import PositionAllocator
from safety import SafetyNet
from broker import BrokerConnection

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

# Store trailing stop info per symbol (simple dictionary)
trailing_stops = {}  # { 'SPY': { 'active': bool, 'stop_price': float, 'highest_close': float, 'entry_price': float } }

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
            print("Circuit breaker triggered. Bot halted.")
            break

        # Check current SPY position
        positions = broker.trading_client.get_all_positions()
        current_spy_shares = 0
        for pos in positions:
            if pos.symbol == "SPY":
                current_spy_shares = float(pos.qty)
                break

        # --- Trailing stop management (if position exists) ---
        if current_spy_shares > 0:
            # Get the current price (use close)
            current_price = historical_df['close'].iloc[-1]
            # Get or create trailing stop record
            if 'SPY' not in trailing_stops:
                # First time we see a position – initialise trailing stop
                # We need the entry price. For simplicity, get the average entry price from Alpaca.
                entry_price = None
                for pos in positions:
                    if pos.symbol == "SPY":
                        entry_price = float(pos.avg_entry_price)
                        break
                if entry_price:
                    trailing_stops['SPY'] = {
                        'active': False,
                        'stop_price': entry_price - 2 * atr,  # initial stop
                        'highest_close': current_price,
                        'entry_price': entry_price,
                        'breakeven_activated': False
                    }
                    print(f"Initial stop set at ${trailing_stops['SPY']['stop_price']:.2f}")
            else:
                ts = trailing_stops['SPY']
                # Update highest close since entry
                if current_price > ts['highest_close']:
                    ts['highest_close'] = current_price
                    print(f"New highest close: ${current_price:.2f}")

                # Breakeven: move stop to entry price once price >= entry + 1*ATR
                if not ts.get('breakeven_activated', False) and current_price >= ts['entry_price'] + atr:
                    ts['stop_price'] = ts['entry_price']
                    ts['breakeven_activated'] = True
                    print(f"Breakeven stop activated at ${ts['stop_price']:.2f}")

                # Trailing: after breakeven, keep stop 2*ATR below highest close
                if ts.get('breakeven_activated', False):
                    new_stop = ts['highest_close'] - 2 * atr
                    if new_stop > ts['stop_price']:
                        ts['stop_price'] = new_stop
                        print(f"Trailing stop raised to ${ts['stop_price']:.2f}")

                # Check if stop loss is hit
                if current_price <= ts['stop_price']:
                    print(f"Stop loss hit at ${current_price:.2f} – selling {current_spy_shares} shares")
                    broker.submit_order("SPY", current_spy_shares, "sell")
                    del trailing_stops['SPY']  # clear trailing data
                    # Skip the rest of the loop after selling
                    time.sleep(60)
                    continue

        # --- Entry logic (only if no position) ---
        if current_spy_shares == 0:
            # Position sizing
            close_prices = historical_df['close'].dropna()
            if len(close_prices) > 20:
                returns = np.log(close_prices / close_prices.shift(1)).dropna()
                volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            else:
                volatility = 0.02
            allocator = PositionAllocator(portfolio_value=portfolio_value)
            target_shares = int(allocator.calculate_position_size(current_market_state, volatility))
            print(f"Target shares to buy: {target_shares}")

            # Buy condition: Bull/Euphoria, RSI<70, and no position
            if current_market_state in ["Bull", "Euphoria"] and target_shares > 0 and rsi_value < 70:
                print(f"RSI {rsi_value:.2f} is below 70 – Placing BUY order for {target_shares} shares of SPY")
                broker.submit_order("SPY", target_shares, "buy")
                # Trailing stop will be initialised on next cycle when position is detected
            else:
                print(f"No buy. RSI = {rsi_value:.2f}")
        else:
            # Already have a position – just print status
            print(f"Holding {current_spy_shares} shares. Trailing stop active: {trailing_stops.get('SPY', {}).get('stop_price', 'N/A')}")

        print("-" * 50)
        time.sleep(60)

    except KeyboardInterrupt:
        print("Shutting down bot.")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)
