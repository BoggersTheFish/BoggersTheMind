"""
Autonomous Explorer — generates hypotheses from current insights/tensions and explores them.
Uses the exact same query pipeline: topic extraction → indexed search → sufficiency → research → synthesis → consolidation.
Feeds from hypothesis queue (insight, consolidation, simulation push) or generates from graph.
"""

import random
import threading
import time
from contextlib import nullcontext
from typing import Callable, Optional, Any

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

# Data Factory: cycle counter for trace cycle_id
_cycle_id = 0


def push_hypothesis(hypothesis: str) -> None:
    """Entities call this to feed new hypotheses into the explorer."""
    if hypothesis and hypothesis.strip():
        HYPOTHESIS_QUEUE.append(hypothesis.strip()[:200])


def _detect_tensions(graph: UniversalLivingGraph, surfaced: list[str], limit: int = 5) -> list[dict]:
    """Minimal tension detection for Data Factory trace. Pairs of surfaced nodes as potential contrasts."""
    tensions = []
    for i, a in enumerate(surfaced[:limit]):
        for b in surfaced[i + 1 : limit]:
            if a == b or a.startswith("topic_") or b.startswith("topic_"):
                continue
            node_a = graph.get_node(a)
            node_b = graph.get_node(b)
            label_a = (node_a or {}).get("label", a)
            label_b = (node_b or {}).get("label", b)
            tensions.append({
                "node_a": a,
                "node_b": b,
                "score": 0.5,
                "description": f"Potential contrast between {label_a} and {label_b}",
            })
    return tensions[:5]


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
    max_cycles: Optional[int] = None,
    start_cycle: Optional[int] = None,
    on_cycle_complete: Optional[Callable[[int, int], None]] = None,
    shutdown_event: Optional[threading.Event] = None,
) -> None:
    """
    Main autonomous loop. Boots in Auto mode.
    Each cycle: check user_requested → if so, yield safely → else generate hypothesis → process → consolidate.
    Data Factory: max_cycles stops after N cycles. start_cycle sets initial cycle_id (for parallel chunking).
    If shutdown_event is set (e.g. SIGINT/SIGTERM on the data factory), exit cleanly after the current cycle.
    """
    global _cycle_id
    if start_cycle is not None:
        _cycle_id = start_cycle - 1  # First increment gives start_cycle
    topic_index = 0

    while True:
        # Graceful stop: finish the previous full cycle, then exit before starting a new one
        if shutdown_event is not None and shutdown_event.is_set():
            return

        # Safe handoff: if user wants to chat, finish this cycle and yield
        if mode_manager.user_requested:
            mode_manager.notify_handoff_complete()
            mode_manager.wait_until_auto()
            continue

        try:
            # Data Factory: wrap cycle with ThinkingTrace when tracing enabled
            from core.tracer import (
                is_tracing_enabled,
                ThinkingTrace,
                record_graph_state_before,
                record_strongest_node,
                record_tensions_detected,
                record_hypothesis_queue,
                record_selected_hypothesis,
                record_graph_state_after,
                record_final_obsidian_note_path,
                record_metadata,
            )

            _cycle_id += 1
            cycle_id = _cycle_id
            trace_ctx = ThinkingTrace(cycle_id) if is_tracing_enabled() else nullcontext()
            with trace_ctx:
                nodes_before = graph.node_count()
                edges_before = graph.edge_count()
                record_graph_state_before(graph.export_snapshot())
                queue_snapshot = [{"text": h, "score": 1.0} for h in list(HYPOTHESIS_QUEUE)]

                # Get hypothesis: queue first, else generate from graph
                hypothesis = None
                if HYPOTHESIS_QUEUE:
                    hypothesis = HYPOTHESIS_QUEUE.popleft()

                seed_only = False
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
                        seed_only = True
                        record_final_obsidian_note_path("")
                        record_metadata({"new_insights": 1 if graph.node_count() > nodes_before else 0})

                if not seed_only:
                    # Data Factory: record trace data (wave, strongest node, tensions, queue snapshot)
                    surfaced = run_wave(graph)
                    sn = graph.get_strongest_node_with_reason()
                    if sn:
                        record_strongest_node(sn)
                    record_tensions_detected(_detect_tensions(graph, surfaced))
                    record_hypothesis_queue(queue_snapshot)
                    record_selected_hypothesis(hypothesis)

                    if state:
                        state.last_action = f"Exploring: {hypothesis[:40]}..."
                        state.feed_items.append(f"[dim]Hypothesis: {hypothesis}[/dim]")

                    # Same pipeline as user query: process_query does it all
                    result = process_query(graph, hypothesis, state=state)

                    new_insights = 1 if result else 0
                    if result and state:
                        state.feed_items.append(f"[bold]Insight:[/bold] {result[:150]}{'...' if len(result) > 150 else ''}")
                        state.insights_generated += 1

                    # Consolidate
                    merged = consolidate(graph)
                    if merged and state:
                        for kept, m in merged:
                            state.feed_items.append(f"[dim]Merged: {m} → {kept}[/dim]")
                        state.last_action = f"Merged {len(merged)} pairs"

                    # Data Factory: graph state after
                    record_graph_state_after({
                        "new_nodes": graph.node_count() - nodes_before,
                        "new_edges": graph.edge_count() - edges_before,
                        "strength_changes": {},
                    })

                    # Write top insight to vault
                    surfaced = run_wave(graph)
                    final_path = ""
                    if surfaced:
                        top = surfaced[0]
                        path = write_insight(graph, top)
                        if path:
                            final_path = str(path)
                            if state:
                                state.feed_items.append(f"[dim]Wrote: {path.name}[/dim]")
                        if state:
                            state.last_action = "Explored"

                    record_final_obsidian_note_path(final_path)
                    record_metadata({"new_insights": new_insights})

        except Exception as e:
            if state:
                state.last_action = f"Error: {str(e)[:30]}"

        if shutdown_event is not None and shutdown_event.is_set():
            return

        time.sleep(EXPLORER_CYCLE_INTERVAL)

        # Progress tracker: notify after each cycle (for parallel job visibility)
        if on_cycle_complete and max_cycles is not None:
            if start_cycle is not None:
                completed = _cycle_id - start_cycle + 1
            else:
                completed = _cycle_id
            on_cycle_complete(completed, max_cycles)

        # Stop: after max_cycles cycles. In chunk mode, stop when _cycle_id >= start_cycle + max_cycles - 1
        if max_cycles is not None:
            if start_cycle is not None:
                if _cycle_id >= start_cycle + max_cycles - 1:
                    return
            elif _cycle_id >= max_cycles:
                return

        if shutdown_event is not None and shutdown_event.is_set():
            return
