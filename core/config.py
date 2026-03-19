"""
Config — runtime flags for BoggersTheMind.
Environment-based for simplicity and reliability (e.g. GitHub Actions).
"""

import os


def is_fast_mode() -> bool:
    """True when BOGGERS_FAST_MODE=1 (disables throttle on GitHub runners)."""
    return os.environ.get("BOGGERS_FAST_MODE") == "1"
