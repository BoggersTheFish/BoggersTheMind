"""
Hypothesis Queue — shared deque for entities to push hypotheses into the autonomous explorer.
No dependencies on entities; breaks circular imports.
"""

from collections import deque

HYPOTHESIS_QUEUE: deque = deque(maxlen=30)


def push_hypothesis(hypothesis: str) -> None:
    """Entities call this to feed new hypotheses into the explorer."""
    if hypothesis and hypothesis.strip():
        HYPOTHESIS_QUEUE.append(hypothesis.strip()[:200])
