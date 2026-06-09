# broker.py

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class BrokerConnection:
    """
    Handles communication with the Alpaca brokerage.
    """
    def __init__(self, api_key, api_secret, is_paper=True):
        self.trading_client = TradingClient(api_key, api_secret, paper=is_paper)

    def get_account_info(self):
        """Get current account balance."""
        return self.trading_client.get_account()

    def get_current_price(self, symbol):
        """Get latest price for a stock (placeholder)."""
        # In a real bot, you would use Alpaca's data API.
        # For now, return a dummy price.
        return 100.00

    def submit_order(self, symbol, qty, side):
        """Submit a market order to buy or sell."""
        if qty <= 0:
            print("Order quantity must be positive.")
            return None

        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )

        try:
            order = self.trading_client.submit_order(order_data)
            print(f"Order placed: {order}")
            return order
        except Exception as e:
            print(f"Order failed: {e}")
            return None
