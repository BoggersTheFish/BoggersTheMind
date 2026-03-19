"""
Hypothesis Simulation — explores what-if scenarios using the graph.
Feeds follow-up hypotheses from simulation results back into the autonomous explorer.
"""

from typing import Optional

from core.graph import UniversalLivingGraph
from core.wave import run_wave
from core.hypothesis_queue import push_hypothesis


def simulate(
    graph: UniversalLivingGraph,
    hypothesis: str,
) -> Optional[str]:
    """
    Simulate a hypothesis: add it as a temporary node, run wave, return insight.
    Does not persist the hypothesis node permanently.
    """
    if graph.node_count() == 0:
        return f"Hypothesis: {hypothesis}\n(No graph context yet — add more knowledge.)"

    # Create temporary node for the hypothesis
    import time
    node_id = f"hyp_{int(time.time())}"
    graph.add_node(
        node_id,
        label=f"Hypothesis: {hypothesis}",
        content=hypothesis,
        strength=2.0,  # Give it initial boost
        source="simulation",
    )

    # Run wave from this node
    surfaced = run_wave(graph, seed_node_id=node_id)

    # Build insight from surfaced nodes
    context = graph.get_context(surfaced, max_chars=1200)

    # Remove temporary node
    graph.remove_node(node_id)

    # Feed follow-up hypothesis from simulation
    if context:
        push_hypothesis(f"What if we extend this: {hypothesis}?")

    return f"Hypothesis: {hypothesis}\n\nRelated context:\n{context}"
