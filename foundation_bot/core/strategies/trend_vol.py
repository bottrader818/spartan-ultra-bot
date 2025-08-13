from dataclasses import dataclass
import pandas as pd
import numpy as np
from foundation_bot.core.utils.math_tools import rolling_vol, atr
from foundation_bot.risk.position_sizing import target_position, kelly_fraction

@dataclass
class TrendVolConfig:
    fast: int = 20
    slow: int = 50
    atr: int = 14
    vol_target: float = 0.20
    kelly_cap: float = 0.5

class TrendVolStrategy:
    def __init__(self, cfg: TrendVolConfig | None = None):
        self.cfg = cfg or TrendVolConfig()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ma_fast"] = out["close"].rolling(self.cfg.fast).mean()
        out["ma_slow"] = out["close"].rolling(self.cfg.slow).mean()
        out["trend"] = np.sign(out["ma_fast"] - out["ma_slow"])
        out["vol"] = rolling_vol(out["close"], window=max(20, self.cfg.slow))
        out["atr"] = atr(out, period=self.cfg.atr)
        out["signal_raw"] = out["trend"]
        out["pos"] = target_position(out["signal_raw"], out["vol"], self.cfg.vol_target, max_abs=1.0)
        # Apply a light Kelly scaling based on trend stability
        edge = (out["ma_fast"] - out["ma_slow"]).abs() / (out["close"] + 1e-9)
        edge = edge.fillna(0.0).clip(0, 0.1)
        out["kelly"] = edge.apply(lambda e: kelly_fraction(e, cap=self.cfg.kelly_cap))
        out["pos"] = (out["pos"] * (0.5 + 0.5 * out["kelly"])).clip(-1, 1)
        return out

    def generate_orders(self, prepared: pd.DataFrame) -> pd.Series:
        # Position is the order target weight
        return prepared["pos"].fillna(0.0)
