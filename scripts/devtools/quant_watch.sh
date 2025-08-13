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
