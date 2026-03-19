"""
Mode Manager — handles AUTO ↔ USER mode with safe handoff.
When user types: finish current autonomous cycle, switch to User Mode, answer, return to Auto.
"""

import threading
from enum import Enum


class Mode(Enum):
    AUTO = "auto"
    USER = "user"


class ModeManager:
    """
    Thread-safe mode manager. Autonomous explorer checks user_requested at cycle start.
    If set, finishes current step, signals handoff complete, blocks until return to auto.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._mode = Mode.AUTO
        self._user_requested = False
        self._handoff_complete = threading.Event()  # Set when explorer has yielded
        self._return_to_auto = threading.Event()   # Set when user is done, explorer can resume

    @property
    def mode(self) -> Mode:
        with self._lock:
            return self._mode

    @property
    def is_auto(self) -> bool:
        return self.mode == Mode.AUTO

    @property
    def is_user(self) -> bool:
        return self.mode == Mode.USER

    @property
    def user_requested(self) -> bool:
        with self._lock:
            return self._user_requested

    def request_user_mode(self) -> None:
        """User wants to chat. Signal explorer to finish current cycle."""
        with self._lock:
            self._user_requested = True
        self._handoff_complete.clear()
        self._return_to_auto.clear()

    def wait_for_safe_handoff(self, timeout: float = 120.0) -> bool:
        """Block until autonomous explorer has finished its current cycle. Returns True if handoff completed."""
        return self._handoff_complete.wait(timeout=timeout)

    def notify_handoff_complete(self) -> None:
        """Called by autonomous explorer when it has yielded."""
        with self._lock:
            self._mode = Mode.USER
            self._user_requested = False
        self._handoff_complete.set()

    def return_to_auto_mode(self) -> None:
        """Called after user gets response. Explorer can resume."""
        with self._lock:
            self._mode = Mode.AUTO
        self._return_to_auto.set()

    def wait_until_auto(self, timeout: float = 300.0) -> bool:
        """Block until we're back in auto mode. Called by explorer when yielding."""
        return self._return_to_auto.wait(timeout=timeout)
