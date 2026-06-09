# allocation.py

class PositionAllocator:
    """
    Decides how much of the portfolio to risk based on the market regime.
    """
    def __init__(self, portfolio_value):
        self.portfolio_value = portfolio_value

    def calculate_position_size(self, market_state, volatility):
        """
        Calculates the position size (how many shares to buy) based on market state.
        """
        # Base risk per trade (e.g., 2% of the portfolio)
        base_risk_pct = 0.02

        # Adjust the position size based on the market state (the Brain's output)
        state_multipliers = {
            "Crash": 0.0,     # Risk nothing in a crash
            "Bear": 0.25,     # Risk 25% of the base amount
            "Neutral": 0.5,   # Risk 50% of the base amount
            "Bull": 1.0,      # Risk the full base amount
            "Euphoria": 0.8,  # Reduce a bit in euphoric markets to be cautious
        }

        multiplier = state_multipliers.get(market_state, 0.5)
        risk_adjusted_pct = base_risk_pct * multiplier
        dollar_risk = self.portfolio_value * risk_adjusted_pct

        # For simplicity, assume we're buying a stock at $100. In real bot, get live price.
        stock_price = 100
        position_size = dollar_risk / stock_price

        return max(0, position_size)  # Never return negative
