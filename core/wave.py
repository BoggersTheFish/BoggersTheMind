"""
Strongest Node Wave — a clean 5-step propagation through the living graph.
Selects the strongest node, propagates influence, and surfaces insights.
Supports optional topic clustering: restrict to nodes under given topics.
"""

from typing import Optional

from .graph import UniversalLivingGraph

# Decay factor for Data Factory trace (wave propagation step metadata)
WAVE_DECAY = 0.78


def run_wave(
    graph: UniversalLivingGraph,
    seed_node_id: Optional[str] = None,
    topic_filter: Optional[list[str]] = None,
) -> list[str]:
    """
    Execute the 5-step Strongest Node wave:
    1. Select — pick strongest node (or seed)
    2. Propagate — spread influence to neighbors
    3. Amplify — boost nodes touched by propagation
    4. Surface — collect top nodes as insights
    5. Return — ordered list of node IDs for downstream use
    """
    if graph.node_count() == 0:
        return []

    # Optional: restrict candidate nodes to topic cluster
    candidate_nodes = None
    if topic_filter:
        candidate_nodes = set(graph.get_nodes_by_topics(topic_filter, limit=50))
        if not candidate_nodes:
            candidate_nodes = None

    # Step 1: Select
    if seed_node_id and graph.G.has_node(seed_node_id):
        current = seed_node_id
    else:
        top = graph.nodes_by_strength(limit=20)
        if candidate_nodes:
            top = [(n, s) for n, s in top if n in candidate_nodes]
        current = top[0][0] if top else None

    if not current:
        return []

    # Step 2: Propagate — boost neighbors (Data Factory: record propagation steps)
    propagation_steps = []
    for neighbor in graph.G.successors(current):
        old_s = graph.get_strength(neighbor)
        energy = 0.3
        graph.set_strength(neighbor, old_s + energy)
        propagation_steps.append({"node": neighbor, "energy": energy, "decay": WAVE_DECAY})

    for predecessor in graph.G.predecessors(current):
        old_s = graph.get_strength(predecessor)
        energy = 0.2
        graph.set_strength(predecessor, old_s + energy)
        propagation_steps.append({"node": predecessor, "energy": energy, "decay": WAVE_DECAY})

    # Step 3: Amplify — boost current node
    old_s = graph.get_strength(current)
    energy = 0.5
    graph.set_strength(current, old_s + energy)
    propagation_steps.append({"node": current, "energy": energy, "decay": WAVE_DECAY})

    try:
        from .tracer import is_tracing_enabled, record_wave_propagation_steps
        if is_tracing_enabled():
            record_wave_propagation_steps(propagation_steps)
    except ImportError:
        pass

    # Step 4: Surface — collect top nodes (optionally within topic cluster)
    surfaced = graph.nodes_by_strength(limit=10)
    if candidate_nodes:
        surfaced = [(n, s) for n, s in surfaced if n in candidate_nodes][:5]
    else:
        surfaced = surfaced[:5]

    # Step 5: Return
    return [nid for nid, _ in surfaced]
