# test_connection.py
from dotenv import load_dotenv
import os
from pathlib import Path
from alpaca.trading.client import TradingClient

# Get the current directory and explicitly load .env from there
env_path = Path.cwd() / '.env'
print(f"Looking for .env at: {env_path}")
print(f"File exists: {env_path.exists()}")

load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

print(f"API Key found: {'Yes' if API_KEY else 'No'}")
print(f"Secret Key found: {'Yes' if SECRET_KEY else 'No'}")

if not API_KEY or not SECRET_KEY:
    print("ERROR: Missing API keys. Check your .env file.")
else:
    try:
        print("Connecting to Alpaca paper trading...")
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        account = trading_client.get_account()
        print(f"✅ Success! Portfolio value: ${float(account.portfolio_value):,.2f}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
