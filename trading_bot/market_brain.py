# market_brain.py

import numpy as np
import pandas as pd
from hmmlearn import hmm

class MarketBrain:
    """
    The "Brain" of your trading bot.
    Uses a Hidden Markov Model (HMM) to classify the market as:
    Crash, Bear, Neutral, Bull, or Euphoria.
    """
    def __init__(self, n_states=5):
        self.n_states = n_states
        self.model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
        self.is_fitted = False
        self.state_labels = {}

    def _prepare_features(self, df):
        """Convert price data into returns and volatility for the HMM."""
        df = df.copy()
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        features = df[['returns', 'volatility']].dropna()
        return features.values

    def train(self, df):
        """Train the HMM on historical price data."""
        print("Training the Brain on historical data...")
        features = self._prepare_features(df)
        self.model.fit(features)
        self.is_fitted = True

        # Predict states for the training data to label them
        states = self.model.predict(features)
        self._label_states(features, states)
        print("Training complete!")

    def _label_states(self, features, states):
        """Give meaningful names (Crash, Bear, etc.) to each hidden state."""
        state_means = {}
        for state in range(self.n_states):
            state_returns = features[states == state, 0]
            state_means[state] = np.mean(state_returns) if len(state_returns) > 0 else 0

        sorted_states = sorted(state_means, key=state_means.get)

        self.state_labels = {
            sorted_states[0]: "Crash",
            sorted_states[1]: "Bear",
            sorted_states[2]: "Neutral",
            sorted_states[3]: "Bull",
            sorted_states[4]: "Euphoria",
        }

    def predict_current_market(self, df):
        """Predict the current market state using the trained model."""
        if not self.is_fitted:
            raise Exception("Brain hasn't been trained yet! Run 'train' first.")

        features = self._prepare_features(df)
        state = self.model.predict(features[-1].reshape(1, -1))[0]
        return self.state_labels[state]

    def get_rsi(self, df, period=14):
        """
        Calculate RSI using pure pandas (no extra library).
        Returns the latest RSI value.
        """
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def get_atr(self, df, period=14):
        """
        Calculate ATR using pure pandas (no extra library).
        Returns the latest ATR value.
        """
        # Use high/low if available, otherwise fallback to close
        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1]
