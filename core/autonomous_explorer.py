"""
Autonomous Explorer — generates hypotheses from current insights/tensions and explores them.
Uses the exact same query pipeline: topic extraction → indexed search → sufficiency → research → synthesis → consolidation.
Feeds from hypothesis queue (insight, consolidation, simulation push) or generates from graph.
"""

import random
import time
from typing import Optional, Any

from core.graph import UniversalLivingGraph
from core.wave import run_wave
from core.mode_manager import ModeManager
from core.hypothesis_queue import HYPOTHESIS_QUEUE
from core.query_processor import process_query
from entities.consolidation import consolidate
from entities.insight import write_insight

# Throttling
EXPLORER_CYCLE_INTERVAL = 45  # seconds
DEFAULT_TOPICS = ["Cognitive science", "Memory", "Knowledge graph", "Artificial intelligence"]


def push_hypothesis(hypothesis: str) -> None:
    """Entities call this to feed new hypotheses into the explorer."""
    if hypothesis and hypothesis.strip():
        HYPOTHESIS_QUEUE.append(hypothesis.strip()[:200])


def _generate_hypothesis_from_graph(graph: UniversalLivingGraph) -> Optional[str]:
    """Generate a fresh hypothesis from surfaced nodes and tensions."""
    if graph.node_count() == 0:
        return None

    surfaced = run_wave(graph)
    if not surfaced:
        return None

    # Build labels from top nodes
    labels = []
    for nid in surfaced[:5]:
        node = graph.get_node(nid)
        if node and not nid.startswith("topic_"):
            label = node.get("label", nid)
            if label:
                labels.append(label)

    if len(labels) < 2:
        return f"What is {labels[0]}?" if labels else None

    # Templates for hypothesis generation
    templates = [
        "What is the relationship between {a} and {b}?",
        "How does {a} connect to {b}?",
        "What tensions exist between {a} and {b}?",
        "How might {a} inform {b}?",
    ]
    a, b = random.sample(labels[:4], 2)
    return random.choice(templates).format(a=a, b=b)


def run_autonomous_explorer(
    graph: UniversalLivingGraph,
    mode_manager: ModeManager,
    state: Optional[Any] = None,
    headless: bool = False,
) -> None:
    """
    Main autonomous loop. Boots in Auto mode.
    Each cycle: check user_requested → if so, yield safely → else generate hypothesis → process → consolidate.
    """
    topic_index = 0

    while True:
        # Safe handoff: if user wants to chat, finish this cycle and yield
        if mode_manager.user_requested:
            mode_manager.notify_handoff_complete()
            mode_manager.wait_until_auto()
            continue

        try:
            # Get hypothesis: queue first, else generate from graph
            hypothesis = None
            if HYPOTHESIS_QUEUE:
                hypothesis = HYPOTHESIS_QUEUE.popleft()

            if not hypothesis:
                hypothesis = _generate_hypothesis_from_graph(graph)
                if not hypothesis:
                    # Fallback: ingest a default topic to seed the graph
                    from entities.ingestor import ingest_wikipedia
                    topic = DEFAULT_TOPICS[topic_index % len(DEFAULT_TOPICS)]
                    topic_index += 1
                    node_id = ingest_wikipedia(graph, topic)
                    if node_id and state:
                        state.feed_items.append(f"[dim]Ingested: {topic}[/dim]")
                        state.insights_generated += 1
                        state.last_action = f"Ingested {topic}"
                    time.sleep(EXPLORER_CYCLE_INTERVAL)
                    continue

            if state:
                state.last_action = f"Exploring: {hypothesis[:40]}..."
                state.feed_items.append(f"[dim]Hypothesis: {hypothesis}[/dim]")

            # Same pipeline as user query: process_query does it all
            result = process_query(graph, hypothesis, state=state)

            if result and state:
                state.feed_items.append(f"[bold]Insight:[/bold] {result[:150]}{'...' if len(result) > 150 else ''}")
                state.insights_generated += 1

            # Consolidate
            merged = consolidate(graph)
            if merged and state:
                for kept, m in merged:
                    state.feed_items.append(f"[dim]Merged: {m} → {kept}[/dim]")
                state.last_action = f"Merged {len(merged)} pairs"

            # Write top insight to vault
            surfaced = run_wave(graph)
            if surfaced:
                top = surfaced[0]
                path = write_insight(graph, top)
                if path and state:
                    state.feed_items.append(f"[dim]Wrote: {path.name}[/dim]")
                state.last_action = "Explored"

        except Exception as e:
            if state:
                state.last_action = f"Error: {str(e)[:30]}"

        time.sleep(EXPLORER_CYCLE_INTERVAL)
