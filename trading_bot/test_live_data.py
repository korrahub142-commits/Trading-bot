import os
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

print(f"API Key loaded: {'Yes' if API_KEY else 'No'}")
print(f"Secret Key loaded: {'Yes' if SECRET_KEY else 'No'}")

if not API_KEY or not SECRET_KEY:
    print("Missing credentials. Check .env file.")
else:
    try:
        client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        request = StockBarsRequest(
            symbol_or_symbols=["SPY"],
            timeframe=TimeFrame.Day,
            start=pd.Timestamp("2025-01-01"),
            end=pd.Timestamp("2026-06-09")
        )
        data = client.get_stock_bars(request)
        bars = data.data.get("SPY", [])
        print(f"Live data fetched, bars: {len(bars)}")
        if bars:
            print("First bar:", bars[0])
    except Exception as e:
        print(f"Error: {e}")
