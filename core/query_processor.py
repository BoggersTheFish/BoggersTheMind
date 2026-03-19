"""
Query Processor — the intelligent query handling pipeline.
Topic breakdown → topic-indexed graph search → sufficiency check → conditional research → synthesis.
"""

import re
from typing import Optional

from core.graph import UniversalLivingGraph
from entities.ingestor import ingest_wikipedia
from entities.inference import synthesize

# Sufficiency: need at least this many relevant nodes, or score from activation+recency
SUFFICIENCY_NODE_MIN = 2
SUFFICIENCY_SCORE_THRESHOLD = 3.0
MAX_RESEARCH_ROUNDS = 3


def _simple_topic_extract(query: str) -> list[str]:
    """Extract 2–5 clean topics from query without Ollama. Fast, no throttle."""
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "what", "how", "why", "when",
        "where", "which", "who", "and", "or", "but", "between", "about", "into",
    }
    words = re.findall(r"\w+", query.lower())
    topics = [w for w in words if len(w) > 2 and w not in STOPWORDS]
    # Dedupe preserving order
    seen = set()
    unique = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    result = unique[:5]
    if not result:
        result = [query.strip()[:50] or "general"]
    return result


def _score_sufficiency(graph: UniversalLivingGraph, node_ids: list[str]) -> float:
    """Score knowledge sufficiency: node count + activation (strength) + recency."""
    if not node_ids:
        return 0.0
    count_score = min(len(node_ids) * 0.5, 5.0)  # cap at 5
    strength_sum = sum(graph.get_strength(n) for n in node_ids[:15])
    activation_score = min(strength_sum * 0.2, 3.0)
    # Recency: nodes with created_at get a small boost (assume recent = better)
    recency_score = 0.5  # baseline
    for nid in node_ids[:5]:
        node = graph.get_node(nid)
        if node and node.get("created_at"):
            recency_score += 0.2
            break
    return count_score + activation_score + recency_score


def process_query(
    graph: UniversalLivingGraph,
    query: str,
    state=None,
) -> Optional[str]:
    """
    Full intelligent query flow:
    1. Break query into 2–5 topics (simple extraction, no Ollama)
    2. Topic-indexed graph search
    3. Score sufficiency
    4. If insufficient → research (Wikipedia) until sufficient
    5. Synthesize answer grounded in graph + any new research
    """
    # Step 1: Topic breakdown
    topics = _simple_topic_extract(query)
    if state:
        state.last_action = f"Topics: {', '.join(topics[:5])}"

    # Step 2: Fast topic-indexed graph search
    relevant_nodes = graph.get_nodes_by_topics(topics, limit=30)
    # Fallback: text search if topic index returns little
    if len(relevant_nodes) < 3:
        for t in topics[:3]:
            hits = graph.search_nodes(t, limit=5)
            relevant_nodes = list(set(relevant_nodes) | set(hits))[:30]

    # Step 3: Score sufficiency
    score = _score_sufficiency(graph, relevant_nodes)
    sufficient = score >= SUFFICIENCY_SCORE_THRESHOLD or len(relevant_nodes) >= SUFFICIENCY_NODE_MIN

    # Step 4: Research fallback if insufficient (Data Factory: track wikipedia_ingest)
    research_done = []
    for _ in range(MAX_RESEARCH_ROUNDS):
        if sufficient:
            break
        for topic in topics:
            if topic in research_done:
                continue
            node_id = ingest_wikipedia(graph, topic, sentences=4)
            if node_id:
                research_done.append(topic)
                relevant_nodes = list(set(relevant_nodes) | {node_id})
                if state:
                    state.feed_items.append(f"[dim]Researched: {topic}[/dim]")
        score = _score_sufficiency(graph, relevant_nodes)
        sufficient = score >= SUFFICIENCY_SCORE_THRESHOLD or len(relevant_nodes) >= SUFFICIENCY_NODE_MIN

    # Data Factory instrumentation: record query pipeline decisions
    try:
        from core.tracer import record_query_pipeline_decisions
        record_query_pipeline_decisions({
            "wikipedia_ingest": len(research_done) > 0,
            "sufficiency_score": score,
            "topics": topics[:5],
            "relevant_nodes_count": len(relevant_nodes),
        })
    except ImportError:
        pass

    # Step 5: Synthesize
    context = graph.get_context(relevant_nodes, max_chars=2000)
    if not context.strip():
        context = "(No relevant knowledge in graph yet. Answer from general knowledge if possible.)"
    return synthesize(graph, query, context)
