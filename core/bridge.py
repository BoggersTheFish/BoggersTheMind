"""
Vault Bridge — watches the Obsidian vault and syncs changes into the living graph.
"""

import hashlib
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent


VAULT_PATH = Path("obsidian/TS-Knowledge-Vault")


class VaultHandler(FileSystemEventHandler):
    """Handles file events in the Obsidian vault."""

    def __init__(self, on_change: Callable[[str, str], None]):
        super().__init__()
        self.on_change = on_change
        self._seen: dict[str, str] = {}  # path -> content hash

    def _process(self, path: str) -> None:
        p = Path(path)
        if p.suffix.lower() != ".md":
            return
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            h = hashlib.sha256(content.encode()).hexdigest()
            if self._seen.get(path) != h:
                self._seen[path] = h
                self.on_change(path, content)
        except Exception:
            pass

    def on_modified(self, event: FileModifiedEvent) -> None:
        if event.is_directory:
            return
        self._process(event.src_path)

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        self._process(event.src_path)


class VaultWatcher:
    """Watches the vault and invokes callback on .md changes."""

    def __init__(self, on_change: Callable[[str, str], None]):
        self.on_change = on_change
        self.observer: Optional[Observer] = None

    def start(self) -> None:
        """Start watching the vault."""
        VAULT_PATH.mkdir(parents=True, exist_ok=True)
        handler = VaultHandler(self.on_change)
        self.observer = Observer()
        self.observer.schedule(handler, str(VAULT_PATH), recursive=True)
        self.observer.start()

    def stop(self) -> None:
        """Stop watching."""
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=2)
