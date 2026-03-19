"""
Insight Writer — writes clean .md notes to the Obsidian vault.
Feeds follow-up hypotheses back into the autonomous explorer.
"""

from pathlib import Path
from typing import Optional

from core.graph import UniversalLivingGraph
from core.bridge import VAULT_PATH
from core.hypothesis_queue import push_hypothesis


def write_insight(
    graph: UniversalLivingGraph,
    node_id: str,
    title: Optional[str] = None,
) -> Optional[Path]:
    """
    Write a node as a clean markdown note in the vault.
    Returns path to written file or None.
    """
    node = graph.get_node(node_id)
    if not node:
        return None

    label = node.get("label", node_id)
    content = node.get("content", "")
    source = node.get("source", "graph")

    VAULT_PATH.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in label)
    safe_name = safe_name[:80].strip() or node_id
    filepath = VAULT_PATH / f"{safe_name}.md"

    body = f"# {label}\n\n{content}\n\n---\n*Source: {source} | Node: {node_id}*\n"
    filepath.write_text(body, encoding="utf-8")

    # Feed follow-up hypothesis for autonomous exploration
    if content and len(content) > 20:
        push_hypothesis(f"What deeper implications does {label} suggest?")

    return filepath
