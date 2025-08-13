from __future__ import annotations
import pandas as pd

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase & flatten columns regardless of MultiIndex/ticker-label weirdness."""
    if isinstance(df.columns, pd.MultiIndex):
        flat = []
        for col in df.columns.values:
            if isinstance(col, tuple):
                # pick the last non-empty element
                parts = [p for p in col if p not in (None, "", " ")]
                flat.append(str(parts[-1]).lower() if parts else "value")
            else:
                flat.append(str(col).lower())
        df.columns = flat
    else:
        df.columns = [str(c).lower() for c in df.columns]
    return df

def _pick(df: pd.DataFrame, name: str) -> str | None:
    """Find a column by fuzzy name (e.g., 'close' matches 'adj close', 'nvda close')."""
    name = name.lower()
    for c in df.columns:
        lc = c.lower()
        if lc == name:
            return c
    # fuzzy: contains name
    candidates = [c for c in df.columns if name in c.lower()]
    return candidates[0] if candidates else None

def _download_yf(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    # primary attempt
    df = yf.download(
        symbol, start=start, end=end, interval=interval,
        auto_adjust=True, group_by="column", progress=False, threads=False
    )
    # edge-case: sometimes returns (df, meta)
    if isinstance(df, tuple):
        df = df[0]
    if df is None or len(df) == 0:
        # fallback: history API
        df = yf.Ticker(symbol).history(start=start, end=end, interval=interval, auto_adjust=True)
    return df

def load_ohlcv(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    try:
        df = _download_yf(symbol, start, end, interval)
    except Exception as e:
        raise RuntimeError(f"yfinance download failed for {symbol}: {e}")

    if df is None or len(df) == 0:
        raise ValueError(f"No data returned for {symbol} ({start}..{end}, {interval}).")

    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = _normalize_columns(df).copy()
    df.index.name = "date"

    # Try to map OHLCV with flexible matching
    open_c  = _pick(df, "open")
    high_c  = _pick(df, "high")
    low_c   = _pick(df, "low")
    close_c = _pick(df, "close") or _pick(df, "adj close")
    vol_c   = _pick(df, "volume") or _pick(df, "vol")

    # As a last resort for crypto/odd feeds: build close-only with synthetic volume
    if close_c and not vol_c:
        df["volume"] = 0
        vol_c = "volume"

    required = {"open": open_c, "high": high_c, "low": low_c, "close": close_c, "volume": vol_c}
    missing_names = [k for k, v in required.items() if v is None]

    if len(missing_names) == 5:
        # extreme fallback: try again without auto_adjust (some intervals behave differently)
        try:
            import yfinance as yf
            df2 = yf.download(symbol, start=start, end=end, interval=interval,
                              auto_adjust=False, group_by=None, progress=False, threads=False)
            if isinstance(df2, tuple): df2 = df2[0]
            if df2 is not None and len(df2) > 0:
                df2 = _normalize_columns(df2)
                # retry mapping
                open_c  = _pick(df2, "open")
                high_c  = _pick(df2, "high")
                low_c   = _pick(df2, "low")
                close_c = _pick(df2, "close") or _pick(df2, "adj close")
                vol_c   = _pick(df2, "volume") or _pick(df2, "vol")
                if close_c and not vol_c:
                    df2["volume"] = 0; vol_c = "volume"
                required = {"open": open_c, "high": high_c, "low": low_c, "close": close_c, "volume": vol_c}
                df = df2
                missing_names = [k for k, v in required.items() if v is None]
        except Exception:
            pass

    if missing_names:
        # Print helpful diagnostics
        cols_preview = list(df.columns)
        raise ValueError(
            f"Missing required columns: {missing_names}. "
            f"Got columns: {cols_preview[:10]}{'...' if len(cols_preview) > 10 else ''}"
        )

    out = pd.DataFrame(index=df.index)
    out["open"]   = pd.to_numeric(df[required["open"]], errors="coerce")
    out["high"]   = pd.to_numeric(df[required["high"]], errors="coerce")
    out["low"]    = pd.to_numeric(df[required["low"]], errors="coerce")
    out["close"]  = pd.to_numeric(df[required["close"]], errors="coerce")
    out["volume"] = pd.to_numeric(df[required["volume"]], errors="coerce").fillna(0)

    out = out.dropna()
    if out.empty:
        raise ValueError(f"All rows NA after cleaning for {symbol}.")
    return out
