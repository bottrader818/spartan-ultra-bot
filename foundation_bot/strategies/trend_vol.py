from __future__ import annotations
import numpy as np
import pandas as pd

class TrendVolStrategy:
    """
    Simple, testable baseline:
      - Prepare: compute returns, EWMA trend, EWMA vol
      - Signals: long if trend > 0, short if trend < 0
      - Position sizing: inverse volatility targeting
      - Risk layer: simple stop/take-profit (optional, off by default)
    """
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.fast = int(self.config.get("fast", 20))
        self.slow = int(self.config.get("slow", 100))
        self.vol_window = int(self.config.get("vol_window", 20))
        self.annualization = float(self.config.get("annualization", 252))
        self.target_vol = float(self.config.get("target_vol", 0.10))  # 10% annual
        self.max_leverage = float(self.config.get("max_leverage", 1.0))
        self.stop_loss = float(self.config.get("stop_loss", 0.0))     # e.g. 0.05 for 5%
        self.take_profit = float(self.config.get("take_profit", 0.0)) # e.g. 0.10 for 10%

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expect df with at least a 'close' column, sorted by time index.
        Returns df with added columns: 'ret', 'trend', 'vol'
        """
        df = df.copy()
        if "close" not in df.columns:
            raise ValueError("DataFrame must have a 'close' column")
        df = df.sort_index()

        # daily returns
        df["ret"] = df["close"].pct_change().fillna(0.0)

        # trend as fast EWMA - slow EWMA (momentum-like)
        fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        slow = df["close"].ewm(span=self.slow, adjust=False).mean()
        df["trend"] = (fast - slow) / slow

        # vol as EWMA of returns, annualized
        vol_daily = df["ret"].ewm(span=self.vol_window, adjust=False).std().fillna(0.0)
        df["vol"] = vol_daily * np.sqrt(self.annualization)

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        signal: +1 if trend>0, -1 if trend<0, else 0
        """
        df = df.copy()
        df["signal"] = 0.0
        df.loc[df["trend"] > 0, "signal"] = 1.0
        df.loc[df["trend"] < 0, "signal"] = -1.0
        return df

    def position_sizer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Size = signal * min(max_leverage, target_vol / max(vol, eps))
        """
        df = df.copy()
        eps = 1e-8
        vol = df["vol"].replace(0, eps)
        raw_size = self.target_vol / vol
        raw_size = raw_size.clip(upper=self.max_leverage)
        df["position"] = df["signal"] * raw_size
        df["position"] = df["position"].fillna(0.0)
        return df

    def risk_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optional stop loss / take profit applied to the *next* day's position.
        For simplicity, we just zero the position when extreme moves occur.
        """
        df = df.copy()
        if self.stop_loss <= 0 and self.take_profit <= 0:
            return df

        # crude flags based on daily return threshold
        stop = (self.stop_loss > 0) & (df["ret"] <= -abs(self.stop_loss))
        take = (self.take_profit > 0) & (df["ret"] >= abs(self.take_profit))
        reset = stop | take

        # zero next day's position when triggered
        reset_idx = df.index[reset]
        next_pos_idx = df.index.intersection(df.index[df.index.isin(reset_idx)].shift(1, freq=None))
        # simpler: shift the mask forward by one row
        mask_next = reset.shift(1).fillna(False)
        df.loc[mask_next, "position"] = 0.0
        return df
