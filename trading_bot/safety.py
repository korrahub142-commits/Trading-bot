# safety.py

class SafetyNet:
    """
    The Safety Net. Monitors portfolio and can halt trading to prevent large losses.
    """
    def __init__(self, initial_portfolio_value, daily_loss_limit_pct=0.03, total_drawdown_limit_pct=0.08):
        self.initial_portfolio_value = initial_portfolio_value
        self.peak_portfolio_value = initial_portfolio_value
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.total_drawdown_limit_pct = total_drawdown_limit_pct
        self.daily_starting_value = initial_portfolio_value
        self.is_circuit_broken = False

    def update_portfolio_value(self, current_portfolio_value):
        """
        Updates portfolio value and checks circuit breakers.
        Returns True if circuit breaker is active (trading halted).
        """
        if self.is_circuit_broken:
            print("Circuit Breaker is active. Trading is halted.")
            return True

        # Daily Loss Check
        daily_loss = (self.daily_starting_value - current_portfolio_value) / self.daily_starting_value
        if daily_loss >= self.daily_loss_limit_pct:
            print(f"Daily loss limit hit: {daily_loss:.2%} > {self.daily_loss_limit_pct:.2%}. Halting trading for the day.")
            self.is_circuit_broken = True
            return True

        # Total Drawdown Check
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value

        drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
        if drawdown >= self.total_drawdown_limit_pct:
            print(f"Total drawdown limit hit: {drawdown:.2%} > {self.total_drawdown_limit_pct:.2%}. Halting all trading.")
            self.is_circuit_broken = True
            return True

        return False

    def reset_daily(self):
        """Call at the start of each new trading day."""
        self.daily_starting_value = self.peak_portfolio_value
        self.is_circuit_broken = False
