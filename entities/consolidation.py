"""
Consolidation — merges similar insights in the graph.
Feeds new hypotheses from merged content back into the autonomous explorer.
"""

import re
from typing import Optional

from core.graph import UniversalLivingGraph
from core.hypothesis_queue import push_hypothesis


def _normalize(text: str) -> str:
    """Normalize text for similarity comparison."""
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t[:100]


def _similarity(a: str, b: str) -> float:
    """Simple Jaccard-like similarity on words."""
    wa = set(_normalize(a).split())
    wb = set(_normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def consolidate(graph: UniversalLivingGraph, threshold: float = 0.6) -> list[tuple[str, str]]:
    """
    Find pairs of similar nodes and merge them (keep stronger, merge content).
    Returns list of (kept_id, merged_id) pairs.
    """
    nodes_before = graph.node_count()
    merged_pairs = []
    nodes = list(graph.G.nodes())
    seen = set()

    for i, nid_a in enumerate(nodes):
        if nid_a in seen:
            continue
        node_a = graph.get_node(nid_a)
        if not node_a:
            continue
        content_a = node_a.get("content", "") or node_a.get("label", "")

        for nid_b in nodes[i + 1 :]:
            if nid_b in seen or nid_a == nid_b:
                continue
            node_b = graph.get_node(nid_b)
            if not node_b:
                continue
            content_b = node_b.get("content", "") or node_b.get("label", "")

            if _similarity(content_a, content_b) >= threshold:
                # Merge: keep stronger node, absorb content
                s_a = graph.get_strength(nid_a)
                s_b = graph.get_strength(nid_b)
                keeper, merged = (nid_a, nid_b) if s_a >= s_b else (nid_b, nid_a)
                kept_node = graph.get_node(keeper)
                merged_node = graph.get_node(merged)
                new_content = (
                    (kept_node.get("content", "") or "")
                    + "\n\n[Merged] "
                    + (merged_node.get("content", "") or merged_node.get("label", ""))
                )
                graph.add_node(keeper, content=new_content.strip())
                # Redirect edges from merged to keeper
                for pred in list(graph.G.predecessors(merged)):
                    if pred != keeper:
                        graph.add_edge(pred, keeper)
                for succ in list(graph.G.successors(merged)):
                    if succ != keeper:
                        graph.add_edge(keeper, succ)
                graph.remove_node(merged)
                merged_pairs.append((keeper, merged))
                seen.add(merged)
                # Feed hypothesis: merged concepts may suggest new connections
                push_hypothesis(f"How do {merged_node.get('label', merged)} and {kept_node.get('label', keeper)} relate?")

    # Data Factory instrumentation: record consolidation score (0–1, higher = more consolidation)
    try:
        from core.tracer import record_metadata, get_current_trace
        trace = get_current_trace()
        if trace is not None:
            meta = trace.get("metadata") or {}
            if isinstance(meta, dict):
                score = min(1.0, len(merged_pairs) * 0.25) if nodes_before > 0 else 0.0
                record_metadata({**meta, "consolidation_score": round(score, 2)})
    except ImportError:
        pass

    return merged_pairs
