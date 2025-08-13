import argparse
import sys
from typing import Any
import pandas as pd

from backtesting.engine import BacktestEngine
from data.loader import fetch_prices


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Price DataFrame is empty.")

    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        for candidate in ("date", "timestamp", "time"):
            if candidate in df.columns:
                df[candidate] = pd.to_datetime(df[candidate], errors="coerce")
                df = df.set_index(candidate)
                break
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
            except Exception:
                pass

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Could not coerce index to DatetimeIndex.")

    # Ensure 'close'
    if "close" not in df.columns:
        for alt in ("adj_close", "adjusted_close", "price", "last", "close_price"):
            if alt in df.columns:
                df["close"] = df[alt]
                break

    if "close" not in df.columns:
        raise ValueError(f"No suitable close column found. Columns: {list(df.columns)}")

    df = df.sort_index()
    df = df.dropna(subset=["close"])
    return df


def print_result(res: Any) -> None:
    def pct(x: float) -> str:
        try:
            return f"{x*100:.2f}%"
        except Exception:
            return str(x)

    if isinstance(res, dict):
        print("\n=== Backtest Summary ===")
        for k in ("symbol", "start", "end", "trades", "win_rate", "cagr", "sharpe", "max_drawdown", "final_value"):
            if k in res:
                v = res[k]
                if k in ("win_rate", "cagr", "max_drawdown"):
                    v = pct(float(v))
                print(f"{k:>14}: {v}")
        extras = {k: v for k, v in res.items() if k not in ("symbol","start","end","trades","win_rate","cagr","sharpe","max_drawdown","final_value")}
        if extras:
            print("\n--- Additional Metrics ---")
            for k, v in extras.items():
                print(f"{k:>14}: {v}")
        print()
    else:
        print(res)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run backtest for a strategy.")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--start", default="2019-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--cash", type=float, default=100000.0, help="Initial cash (default: 100000)")
    parser.add_argument("--csv", default=None, help="Optional CSV path instead of loader")
    args = parser.parse_args(argv)

    # Load data
    if args.csv:
        df = pd.read_csv(args.csv)
        for candidate in ("date", "timestamp", "time"):
            if candidate in df.columns:
                df[candidate] = pd.to_datetime(df[candidate], errors="coerce")
                df = df.set_index(candidate)
                break
    else:
        df = fetch_prices(args.symbol, args.start, args.end)

    # Normalize and validate
    df = normalize_price_frame(df)

    # Run backtest
    engine = BacktestEngine(initial_cash=args.cash)
    res = engine.run(df)

    # Pretty print
    print_result(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
