import argparse
import os
from foundation_bot.data.loader import load_ohlcv
from foundation_bot.backtesting.engine import BacktestEngine

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--interval", default="1d")
    args = p.parse_args()

    # allow running from repo root
    os.environ.setdefault("PYTHONPATH", os.getcwd())
    df = load_ohlcv(args.symbol, args.start, args.end, args.interval)
    res = BacktestEngine().run(df)
    print("=== Results ===")
    for k, v in res.metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
