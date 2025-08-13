import numpy as np
import pandas as pd

def rolling_vol(series: pd.Series, window: int=20) -> pd.Series:
    returns = series.pct_change()
    return returns.rolling(window).std() * np.sqrt(252)

def atr(df: pd.DataFrame, period: int=14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = np.maximum(high - low, np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
    return pd.Series(tr).rolling(period).mean()
