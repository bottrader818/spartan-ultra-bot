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
