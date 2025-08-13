#!/usr/bin/env python3
import re, sys, time
from pathlib import Path

# Edit these as needed for your project
IMPORT_MAP = {
    # old path                          -> new path
    "core.strategies.dtw_mean_reversion_strategy": "core.strategies.traditional.mean_reversion.dtw_mean_reversion_strategy",
    "core.strategies.volatility_breakout_strategy": "core.strategies.traditional.volatility.volatility_breakout_strategy",
    "core.strategies.dtw_neural": "core.strategies.neural.dtw_neural",
    "core.strategies.ai_signal_fusion_strategy": "core.strategies.neural.ai_signal_fusion_strategy",
}

BACKUP_DIR = Path(".autobackups")

def rewrite(code: str) -> str:
    for old, new in IMPORT_MAP.items():
        code = re.sub(rf'\bfrom {re.escape(old)}\b', f'from {new}', code)
        code = re.sub(rf'\bimport {re.escape(old)}\b', f'import {new}', code)
    return code

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    BACKUP_DIR.mkdir(exist_ok=True)
    changed = 0
    for path in root.rglob("*.py"):
        # skip venv, caches, zips
        p = str(path.as_posix())
        if any(seg in p for seg in ("/venv/", "__pycache__", "/.autobackups/")):
            continue
        original = path.read_text(encoding="utf-8", errors="ignore")
        updated = rewrite(original)
        if updated != original:
            ts = int(time.time())
            backup = BACKUP_DIR / f"{path.name}.bak_{ts}"
            backup.write_text(original, encoding="utf-8")
            path.write_text(updated, encoding="utf-8")
            print(f"fixed imports: {path} (backup: {backup.name})")
            changed += 1
    print(f"Done. Files changed: {changed}")

if __name__ == "__main__":
    main()
