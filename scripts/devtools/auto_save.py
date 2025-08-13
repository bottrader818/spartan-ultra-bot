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
