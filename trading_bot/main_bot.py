# main_bot.py
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta# --- Keep-alive web server (for Render) ---
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler
import os

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
# -----------------------------------------
from dotenv import load_dotenv
import os
import yfinance as yf

from market_brain import MarketBrain
from allocation import PositionAllocator
from safety import SafetyNet
from broker import BrokerConnection

# Load environment variables
load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing API keys in .env file")

# Initialize components
print("Initializing Trading Bot...")
broker = BrokerConnection(API_KEY, SECRET_KEY, is_paper=True)

# Get initial portfolio value
account = broker.get_account_info()
initial_value = float(account.portfolio_value)
print(f"Initial portfolio value: ${initial_value:,.2f}")

brain = MarketBrain(n_states=5)
safety_net = SafetyNet(initial_portfolio_value=initial_value)

# Fetch live historical data using yfinance (no API key needed)
print("Fetching live historical data (SPY daily bars from Yahoo Finance)...")
end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)

spy = yf.Ticker("SPY")
historical_df = spy.history(start=start_date, end=end_date, interval="1d")
if historical_df.empty:
    raise ValueError("No data received from Yahoo Finance")
historical_df = historical_df[['Close']].rename(columns={'Close': 'close'})
print(f"Training data: {len(historical_df)} days")

# Train the brain
brain.train(historical_df)

# Main trading loop
print("Starting main trading loop. Will check every 60 seconds.\n")
while True:
    try:
        print(f"DEBUG: Loop running at {datetime.now().strftime('%H:%M:%S')}")
        # Fetch latest daily bar (to update the dataframe)
        new_data = spy.history(period="5d", interval="1d")
        if not new_data.empty:
            latest_close = new_data['Close'].iloc[-1]
            latest_date = new_data.index[-1]
            # Avoid duplicate
            if latest_date not in historical_df.index:
                new_row = pd.DataFrame({'close': [latest_close]}, index=[latest_date])
                historical_df = pd.concat([historical_df, new_row])
                print(f"Added new data point: {latest_date.date()}")
        
        # Predict current market state
        market_state = brain.predict_current_market(historical_df)
        print(f"{datetime.now().strftime('%H:%M:%S')} Market regime: {market_state}")
        
        # Get current account value from Alpaca
        account = broker.get_account_info()
        portfolio_value = float(account.portfolio_value)
        print(f"Portfolio value: ${portfolio_value:,.2f}")
        
        # Safety check
        if safety_net.update_portfolio_value(portfolio_value):
            print("Circuit breaker triggered. Bot halted.")
            break
        
        # Position sizing using recent volatility
        close_prices = historical_df['close'].dropna()
        if len(close_prices) > 20:
            returns = np.log(close_prices / close_prices.shift(1)).dropna()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        else:
            volatility = 0.02
        allocator = PositionAllocator(portfolio_value=portfolio_value)
        target_shares = int(allocator.calculate_position_size(market_state, volatility))
        print(f"Target shares to buy/sell: {target_shares}")
        
        # Trading decision
        if market_state in ["Bull", "Euphoria"] and target_shares > 0:
            # Check existing SPY position
            positions = broker.trading_client.get_all_positions()
            current_spy_shares = 0
            for pos in positions:
                if pos.symbol == "SPY":
                    current_spy_shares = float(pos.qty)
                    break
            if current_spy_shares == 0:
                print(f"Placing BUY order for {target_shares} shares of SPY")
                broker.submit_order("SPY", target_shares, "buy")
            else:
                print(f"Already hold {current_spy_shares} shares. Skipping buy.")
        elif market_state in ["Bear", "Crash"]:
            # Close all SPY positions
            positions = broker.trading_client.get_all_positions()
            for pos in positions:
                if pos.symbol == "SPY":
                    qty = float(pos.qty)
                    if qty > 0:
                        print(f"Closing position: selling {qty} shares of SPY")
                        broker.submit_order("SPY", qty, "sell")
        else:
            print("No action.")
        
        print("-" * 50)
        time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down bot.")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)

# --- Keep-alive web server (for Render) ---
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler
import os

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

# --- Keep-alive web server (for Render) ---
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler
import os

class HealthHandler(BaseHTTPRequestHandler):
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
