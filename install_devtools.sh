# 1) Create folder + files
mkdir -p scripts/devtools .autobackups

# --- auto_correct.py ---
cat > scripts/devtools/auto_correct.py <<'PY'
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
PY

# --- auto_save.py ---
cat > scripts/devtools/auto_save.py <<'PY'
#!/usr/bin/env python3
import time, shutil
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

BACKUP_DIR = Path(".autobackups")
BACKUP_DIR.mkdir(exist_ok=True)

class Saver(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory: return
        p = Path(event.src_path)
        if p.suffix != ".py": return
        if any(s in p.as_posix() for s in ("/venv/", "__pycache__", "/.autobackups/")):
            return
        ts = int(time.time())
        backup = BACKUP_DIR / f"{p.name}.bak_{ts}"
        try:
            shutil.copy2(p, backup)
            print(f"[auto-save] backup -> {backup}")
        except Exception as e:
            print(f"[auto-save] failed: {e}")

if __name__ == "__main__":
    path = "."
    handler = Saver()
    obs = Observer()
    obs.schedule(handler, path=path, recursive=True)
    obs.start()
    print("[auto-save] watching for .py changes…")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()
PY

# --- auto_test.py ---
cat > scripts/devtools/auto_test.py <<'PY'
#!/usr/bin/env python3
import subprocess, sys, time
from pathlib import Path
from threading import Timer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

DEBOUNCE = 0.4  # seconds
proc = None
timer = None

def run_tests():
    global proc
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("\n[auto-test] running pytest…")
    proc = subprocess.Popen(["pytest", "-q"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        line = proc.stdout.readline()
        if not line: break
        print(line, end="")
    proc.wait()

class Runner(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory: return
        p = Path(event.src_path)
        if p.suffix != ".py": return
        if any(s in p.as_posix() for s in ("/venv/", "__pycache__", "/.autobackups/")): return
        self._debounced()

    def _debounced(self):
        global timer
        if timer: timer.cancel()
        timer = Timer(DEBOUNCE, run_tests)
        timer.start()

if __name__ == "__main__":
    obs = Observer()
    obs.schedule(Runner(), path=".", recursive=True)
    obs.start()
    print("[auto-test] watching for changes…")
    run_tests()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()
PY

# --- auto_debug.sh ---
cat > scripts/devtools/auto_debug.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

TARGETS=("core" "infrastructure" "strategies" "trading_system")
echo "[auto-debug] starting loop (Ctrl+C to stop)"
while true; do
  echo "---- lint (errors only) ----"
  pylint --errors-only "${TARGETS[@]}" || true
  echo "---- quick tests ----"
  pytest -q || true
  sleep 10
done
SH

# --- quant_watch.sh ---
cat > scripts/devtools/quant_watch.sh <<'SH'
#!/usr/bin/env bash
set -e

# Ensure executables
chmod +x scripts/devtools/auto_debug.sh || true

have_tmux=$(command -v tmux || true)
if [ -n "$have_tmux" ]; then
  session="quant_devtools"
  tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session" || true
  tmux new-session -d -s "$session" 'python3 scripts/devtools/auto_save.py'
  tmux split-window -v 'python3 scripts/devtools/auto_test.py'
  tmux split-window -h 'bash scripts/devtools/auto_debug.sh'
  tmux select-layout tiled
  tmux attach -t "$session"
else
  echo "[quant-watch] tmux not found. Running sequentially in this terminal."
  echo "Press Ctrl+C to stop."
  (python3 scripts/devtools/auto_save.py &) 
  (python3 scripts/devtools/auto_test.py &)
  bash scripts/devtools/auto_debug.sh
fi
SH

# --- requirements-dev.txt ---
cat > requirements-dev.txt <<'REQ'
watchdog
pytest
pytest-xdist
pylint
libcst
memory_profiler
REQ

# 2) Make executables
chmod +x scripts/devtools/*.sh
chmod +x scripts/devtools/*.py

echo "Devtools installed in scripts/devtools/"
