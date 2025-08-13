from dataclasses import dataclass
import numpy as np
import pandas as pd
import yaml

from foundation_bot.data.loader import load_ohlcv
from foundation_bot.core.strategies.trend_vol import TrendVolStrategy, TrendVolConfig
from foundation_bot.core.strategies.dtw_mean_rev import DTWMeanReversion, DTWMeanRevConfig
from foundation_bot.risk.drawdown import max_drawdown

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    metrics: dict

class BacktestEngine:
    def __init__(self, config_path: str = "foundation_bot/config/config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def _build_strategy(self):
        sc = self.cfg["strategy"]
        if sc["type"] == "trend_vol":
            return TrendVolStrategy(TrendVolConfig(
                fast=sc.get("fast",20),
                slow=sc.get("slow",50),
                atr=sc.get("atr",14),
                vol_target=sc.get("vol_target",0.2),
                kelly_cap=sc.get("kelly_cap",0.5),
            ))
        elif sc["type"] == "dtw_mean_rev":
            return DTWMeanReversion(DTWMeanRevConfig(
                lookback=sc.get("lookback",50),
                z_entry=sc.get("z_entry",1.0),
                z_exit=sc.get("z_exit",0.2),
                max_abs=sc.get("max_abs",0.8),
            ))
        else:
            raise ValueError(f"Unknown strategy.type: {sc['type']}")

    def run(self, df: pd.DataFrame) -> BacktestResult:
        strat = self._build_strategy()
        data = strat.prepare(df)
        weights = strat.generate_orders(data)

        # Simple portfolio: daily rebalancing to weight, 100k starting, close-to-close fills
        equity = pd.Series(index=data.index, dtype=float)
        equity.iloc[0] = 100_000.0
        pos_shares = 0.0

        for i in range(1, len(data)):
            px_prev = float(data["close"].iloc[i-1])
            px_now = float(data["close"].iloc[i])
            wt = float(weights.iloc[i-1])  # rebalance at previous close for next period

            # rebalance position to target weight at px_prev
            port_val = float(equity.iloc[i-1])
            target_shares = (wt * port_val) / max(px_prev, 1e-9)
            pos_shares = target_shares

            # mark-to-market to px_now
            port_val = port_val + pos_shares * (px_now - px_prev)
            equity.iloc[i] = port_val

        ret = equity.pct_change().fillna(0.0)
        sharpe = float(np.sqrt(252) * ret.mean() / (ret.std() + 1e-12))
        mdd = max_drawdown(equity)
        metrics = {"sharpe": sharpe, "max_drawdown": mdd, "final_equity": float(equity.iloc[-1])}
        return BacktestResult(equity_curve=equity, metrics=metrics)
