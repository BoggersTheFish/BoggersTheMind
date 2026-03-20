"""
Ollama Inference — reasoning grounded in the living graph. Throttled to 1 call per 60s.
"""

import os
import time
from typing import Optional

import ollama

from core.graph import UniversalLivingGraph
from core.wave import run_wave

# Throttle: max 1 Ollama call per 60 seconds (laptop-friendly)
LAST_CALL_TIME: float = 0
MIN_INTERVAL = 60.0
MODEL = "llama3.2"


def _client() -> ollama.Client:
    """HTTP client with bounded wait (httpx timeout); avoids indefinite hang when Ollama is down."""
    sec = float(os.environ.get("OLLAMA_TIMEOUT", "120"))
    return ollama.Client(timeout=sec)


def _throttle() -> bool:
    """Return True if we can make a call, False if we must wait."""
    # Fast mode disables throttle on GitHub runners (fast CPUs, no laptop limits)
    if os.environ.get("BOGGERS_FAST_MODE") == "1":
        return True
    global LAST_CALL_TIME
    now = time.time()
    if now - LAST_CALL_TIME >= MIN_INTERVAL:
        LAST_CALL_TIME = now
        return True
    return False


def infer(
    graph: UniversalLivingGraph,
    prompt: str,
    use_graph: bool = True,
) -> Optional[str]:
    """
    Run Ollama with prompt, optionally grounded in graph context.
    Returns response text or None if throttled/failed.
    """
    if not _throttle():
        return None

    context = ""
    if use_graph and graph.node_count() > 0:
        surfaced = run_wave(graph)
        context = graph.get_context(surfaced, max_chars=1500)
        if context:
            context = f"Relevant knowledge:\n{context}\n\n"

    system = (
        "You are BoggersTheMind, a thoughtful cognitive assistant. "
        "Answer concisely and insightfully. Use the provided knowledge when relevant."
    )
    user_msg = f"{context}User: {prompt}"

    try:
        response = _client().chat(model=MODEL, messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ])
        return response["message"]["content"].strip()
    except Exception:
        return None


def synthesize(
    graph: UniversalLivingGraph,
    query: str,
    context: str,
) -> Optional[str]:
    """
    Synthesize final answer grounded in graph context. Uses single Ollama call.
    Strengthened prompt for coherent, insightful synthesis.
    """
    if not _throttle():
        return None

    system = (
        "You are BoggersTheMind. Synthesize a clear, insightful answer to the user's question "
        "using ONLY the provided knowledge context. Be concise but thorough. "
        "If the context is insufficient, say so briefly and offer what you can. "
        "Do not invent facts. Ground every claim in the context."
    )
    user_msg = f"Knowledge context:\n{context}\n\nUser question: {query}\n\nAnswer:"

    try:
        response = _client().chat(model=MODEL, messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ])
        result = response["message"]["content"].strip()
        # Data Factory instrumentation: record synthesis output
        try:
            from core.tracer import record_synthesis_output
            record_synthesis_output(result or "")
        except ImportError:
            pass
        return result
    except Exception:
        return None
