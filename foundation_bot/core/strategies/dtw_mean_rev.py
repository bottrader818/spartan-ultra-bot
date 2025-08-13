from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class DTWMeanRevConfig:
    lookback: int = 50
    z_entry: float = 1.0
    z_exit: float = 0.2
    max_abs: float = 0.8

class DTWMeanReversion:
    def __init__(self, cfg: DTWMeanRevConfig | None = None):
        self.cfg = cfg or DTWMeanRevConfig()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ret"] = out["close"].pct_change()
        out["z"] = (out["close"] - out["close"].rolling(self.cfg.lookback).mean()) / (out["close"].rolling(self.cfg.lookback).std() + 1e-9)
        out["z"] = out["z"].fillna(0.0)
        pos = np.where(out["z"] > self.cfg.z_entry, -1.0,
              np.where(out["z"] < -self.cfg.z_entry, 1.0,
              np.where(out["z"].abs() < self.cfg.z_exit, 0.0, np.nan)))
        # forward fill within regimes
        out["pos"] = pd.Series(pos, index=out.index).ffill().fillna(0.0)
        out["pos"] = out["pos"].clip(-self.cfg.max_abs, self.cfg.max_abs)
        return out

    def generate_orders(self, prepared: pd.DataFrame) -> pd.Series:
        return prepared["pos"].fillna(0.0)
