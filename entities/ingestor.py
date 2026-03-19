"""
Wikipedia Ingestor — pulls concepts from Wikipedia and adds them to the living graph.
"""

import re
import time
from typing import Optional

import wikipedia

from core.graph import UniversalLivingGraph


def slugify(text: str) -> str:
    """Create a safe node ID from text."""
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s_-]+", "_", s).strip("_")
    return s[:64] if s else f"node_{int(time.time())}"


def ingest_wikipedia(
    graph: UniversalLivingGraph,
    topic: str,
    sentences: int = 3,
) -> Optional[str]:
    """
    Fetch Wikipedia summary for topic, add as node, return node_id.
    """
    try:
        summary = wikipedia.summary(topic, sentences=sentences, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        summary = wikipedia.summary(e.options[0], sentences=sentences)
    except wikipedia.exceptions.PageError:
        return None

    node_id = slugify(topic)
    graph.add_node(
        node_id,
        topics=[topic],
        label=topic,
        content=summary,
        strength=1.0,
        source="wikipedia",
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    return node_id
