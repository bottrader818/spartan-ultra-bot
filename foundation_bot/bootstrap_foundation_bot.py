# bootstrap_foundation_bot.py
import os, textwrap

def write(path, content):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(textwrap.dedent(content).lstrip())

write("requirements.txt", """
numpy
pandas
yfinance
python-dotenv
alpaca-trade-api>=3.1
""")

write(".env.example", """
ALPACA_API_KEY_ID=your_key
ALPACA_API_SECRET_KEY=your_secret
ALPACA_PAPER_BASE=https://paper-api.alpaca.markets
""")

write("core/strategies/trend_vol.py", """
import numpy as np
import pandas as pd
from dataclasses import dataclass

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h,l,c = df['High'], df['Low'], df['Close']
    tr = np.maximum(h-l, np.maximum((h-c.shift()).abs(), (l-c.shift()).abs()))
    return pd.Series(tr).rolling(n, min_periods=n).mean()

@dataclass
class TrendVolStrategy:
    fast: int = 20
    slow: int = 50
    breakout_lookback: int = 55
    vol_target: float = 0.15  # annualized target vol (15%)
    atr_stop_R: float = 1.0
    atr_tp_R: float = 2.0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["fast"] = ema(df["Close"], self.fast)
        df["slow"] = ema(df["Close"], self.slow)
        df["trend"] = (df["fast"] > df["slow"]).astype(int)
        df["breakout"] = (df["Close"] > df["High"].rolling(self.breakout_lookback, min_periods=self.breakout_lookback).max().shift()).astype(int)
        df["ATR"] = atr(df, 14)
        # Daily realized vol proxy
        ret = df["Close"].pct_change()
        df["daily_vol"] = ret.rolling(20, min_periods=20).std()
        return df

    def signal(self, row: pd.Series) -> float:
        # Long-only: require trend and breakout confirmation
        return 1.0 if (row["trend"] == 1 and row["breakout"] == 1) else 0.0

    def size(self, equity: float, price: float, daily_vol: float) -> float:
        # Target daily vol ≈ annual vol / sqrt(252)
        if not np.isfinite(daily_vol) or daily_vol <= 0:
            return 0.0
        target_daily = self.vol_target / np.sqrt(252.0)
        # dollar exposure = equity * target_daily / daily_vol
        exposure = equity * (target_daily / daily_vol)
        qty = max(0.0, exposure / price)
        return float(qty)

    def stops(self, entry: float, atr_val: float):
        sl = entry - self.atr_stop_R * atr_val
        tp = entry + self.atr_tp_R * atr_val
        return sl, tp
""")

write("core/risk/risk_manager.py", """
from dataclasses import dataclass

@dataclass
class RiskManager:
    max_drawdown: float = 0.25  # 25% hard stop

    def breach(self, peak_equity: float, equity: float) -> bool:
        if peak_equity <= 0: return False
        dd = 1.0 - (equity / peak_equity)
        return dd >= self.max_drawdown
""")

write("core/execution/paper_executor.py", """
from dataclasses import dataclass

@dataclass
class FillReport:
    filled_qty: float
    avg_price: float
    status: str  # 'filled' | 'flat'

class PaperExecutor:
    def submit(self, side: str, qty: float, price: float) -> FillReport:
        if qty <= 0:
            return FillReport(0.0, price, "flat")
        # Naive: assume immediate fill at given price
        return FillReport(qty, price, "filled")
""")

write("backtesting/engine.py", """
import pandas as pd
from dataclasses import dataclass
from core.strategies.trend_vol import TrendVolStrategy
from core.risk.risk_manager import RiskManager
from core.execution.paper_executor import PaperExecutor

@dataclass
class BacktestResult:
    equity_curve: pd.Series

@dataclass
class BacktestEngine:
    start_equity: float = 100_000.0

    def run(self, df: pd.DataFrame) -> BacktestResult:
        strat = TrendVolStrategy()
        risk = RiskManager()
        execu = PaperExecutor()

        data = strat.prepare(df)
        equity = self.start_equity
        peak = equity
        pos = 0.0
        entry = 0.0
        sl = None; tp = None

        records = []

        for i in range(len(data)):
            row = data.iloc[i]
            price = float(row["Close"])

            # Drawdown hard-stop check
            if risk.breach(peak, equity):
                # force flat
                if pos != 0:
                    equity += (price - entry) * pos
                    pos = 0.0; entry = 0.0; sl = tp = None
                records.append({"equity": equity})
                continue

            # manage trailing stops while in position
            if pos > 0 and row["ATR"] == row["ATR"]:  # ATR not NaN
                # trailing SL at 2*ATR behind price (simple)
                trail_sl = price - 2.0 * float(row["ATR"])
                sl = max(sl or trail_sl, trail_sl)

            # exit conditions
            if pos > 0:
                if (sl is not None and price <= sl) or (tp is not None and price >= tp):
                    # exit
                    equity += (price - entry) * pos
                    pos = 0.0; entry = 0.0; sl = tp = None

            # entry (long only)
            sig = strat.signal(row)
            if pos == 0 and sig > 0 and row["daily_vol"] == row["daily_vol"]:
                target_qty = strat.size(equity, price, float(row["daily_vol"]))
                if target_qty > 0:
                    fill = execu.submit("buy", target_qty, price)
                    if fill.status == "filled" and fill.filled_qty > 0:
                        pos = fill.filled_qty
                        entry = fill.avg_price
                        sl, tp = strat.stops(entry, float(row["ATR"]))

            # mark-to-market
            if pos > 0:
                mtm = (price - entry) * pos
                records.append({"equity": equity + mtm})
            else:
                records.append({"equity": equity})

            # track peak for DD
            if records[-1]["equity"] > peak:
                peak = records[-1]["equity"]

        eq = pd.Series([r["equity"] for r in records], index=df.index[:len(records)])
        return BacktestResult(equity_curve=eq)
""")

write("run_backtest.py", """
import argparse, pandas as pd, numpy as np
import yfinance as yf

def load(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
    return df.dropna()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end",   default="2024-12-31")
    args = ap.parse_args()

    df = load(args.symbol, args.start, args.end)
    from backtesting.engine import BacktestEngine
    res = BacktestEngine().run(df)
    eq = res.equity_curve
    ret = (eq.iloc[-1]/eq.iloc[0]-1)*100
    dd = (1 - eq/eq.cummax()).max()*100
    print(eq.tail())
    print(f"Return: {ret:.2f}%   MaxDD: {dd:.2f}%   CAGR-ish: {(1+ret/100)**(252/len(eq))-1:.2%}")
""")

print("✅ Bootstrapped. Next:")
print("  1) python -m pip install -U pip && pip install -r requirements.txt")
print("  2) python run_backtest.py --symbol SPY --start 2019-01-01 --end 2024-12-31")
