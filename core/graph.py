"""
UniversalLivingGraph — the persistent knowledge backbone of BoggersTheMind.
Nodes are concepts/insights; edges are relationships. Topic filing system for fast indexed lookup.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

import networkx as nx

# Topic cluster prefix — nodes with this prefix are index nodes, not content
TOPIC_PREFIX = "topic_"


def _slugify_topic(text: str) -> str:
    """Normalize topic for cluster ID."""
    s = re.sub(r"[^\w\s-]", "", text.lower().strip())
    s = re.sub(r"[\s_-]+", "_", s).strip("_")
    return s[:48] if s else "general"


class UniversalLivingGraph:
    """Directed graph of concepts and their relationships. Persists to JSON.
    Topic filing: auto-creates topic cluster nodes and maintains fast lookup index.
    """

    def __init__(self, persist_path: str = "data/graph.json"):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.G = nx.DiGraph()
        self._topic_index: dict[str, set[str]] = {}  # topic_slug -> set of node_ids
        self._load()

    def _load(self) -> None:
        """Load graph and topic index from disk."""
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text(encoding="utf-8"))
                self.G = nx.node_link_graph(data)
            except (json.JSONDecodeError, nx.NetworkXError):
                self.G = nx.DiGraph()

        self._rebuild_topic_index()

    def _rebuild_topic_index(self) -> None:
        """Rebuild topic index from graph structure (topic clusters -> content nodes)."""
        self._topic_index = {}
        for nid in self.G.nodes():
            if nid.startswith(TOPIC_PREFIX):
                self._topic_index[nid] = set(self.G.successors(nid))

    def _save(self) -> None:
        """Persist graph. Topic index is rebuilt from graph on load."""
        data = nx.node_link_data(self.G)
        self.persist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_node(self, node_id: str, topics: Optional[list[str]] = None, **attrs) -> None:
        """Add or update a node. Attrs: label, content, strength, created_at, source.
        If topics provided, auto-creates topic clusters and updates fast lookup index.
        """
        existing = self.G.nodes.get(node_id, {})
        merged = {**existing, **attrs}
        self.G.add_node(node_id, **merged)

        # Topic filing: link node to topic clusters and update index
        if topics:
            for t in topics[:10]:  # cap at 10 topics per node
                slug = _slugify_topic(t)
                if not slug:
                    continue
                cluster_id = TOPIC_PREFIX + slug
                # Create topic cluster node if needed
                if not self.G.has_node(cluster_id):
                    self.G.add_node(
                        cluster_id,
                        label=f"Topic: {t}",
                        type="topic_cluster",
                        strength=0.5,
                    )
                self.G.add_edge(cluster_id, node_id, weight=1.0)
                if cluster_id not in self._topic_index:
                    self._topic_index[cluster_id] = set()
                self._topic_index[cluster_id].add(node_id)

        self._save()

    def add_edge(self, u: str, v: str, weight: float = 1.0, **attrs) -> None:
        """Add or update an edge between nodes."""
        self.G.add_edge(u, v, weight=weight, **attrs)
        self._save()

    def get_node(self, node_id: str) -> Optional[dict]:
        """Get node attributes or None."""
        return dict(self.G.nodes.get(node_id, {})) or None

    def get_strength(self, node_id: str) -> float:
        """Node strength (default 1.0). Higher = more central."""
        return self.G.nodes.get(node_id, {}).get("strength", 1.0)

    def set_strength(self, node_id: str, strength: float) -> None:
        """Update node strength."""
        self.G.nodes[node_id]["strength"] = strength
        self._save()

    def nodes_by_strength(self, limit: int = 10) -> list[tuple[str, float]]:
        """Return top nodes by strength, descending."""
        items = [(n, self.get_strength(n)) for n in self.G.nodes()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:limit]

    def get_nodes_by_topics(self, topics: list[str], limit: int = 30) -> list[str]:
        """Fast lookup: return all content nodes under the given topics.
        Uses topic index. Excludes topic cluster nodes themselves.
        """
        seen = set()
        for t in topics:
            slug = _slugify_topic(t)
            cluster_id = TOPIC_PREFIX + slug
            node_ids = self._topic_index.get(cluster_id, set())
            for nid in node_ids:
                if not nid.startswith(TOPIC_PREFIX) and self.G.has_node(nid):
                    seen.add(nid)
        return list(seen)[:limit]

    def get_context(self, node_ids: list[str], max_chars: int = 2000) -> str:
        """Build context string from selected nodes for LLM grounding."""
        parts = []
        for nid in node_ids[:20]:
            node = self.get_node(nid)
            if node and not nid.startswith(TOPIC_PREFIX):
                label = node.get("label", nid)
                content = node.get("content", "")
                parts.append(f"- {label}: {content[:200]}")
        text = "\n".join(parts)
        return text[:max_chars] if len(text) > max_chars else text

    def search_nodes(self, query: str, limit: int = 5) -> list[str]:
        """Simple text search over node labels and content."""
        query_lower = query.lower()
        matches = []
        for nid, data in self.G.nodes(data=True):
            if nid.startswith(TOPIC_PREFIX):
                continue
            label = data.get("label", "").lower()
            content = data.get("content", "").lower()
            if query_lower in label or query_lower in content:
                matches.append(nid)
        return matches[:limit]

    def node_count(self) -> int:
        return self.G.number_of_nodes()

    def edge_count(self) -> int:
        return self.G.number_of_edges()

    def save(self) -> None:
        """Public save for external modifications (e.g. consolidation)."""
        self._save()

    def remove_node(self, node_id: str) -> None:
        """Remove node and clean topic index. Use instead of G.remove_node."""
        for cluster_id, nodes in list(self._topic_index.items()):
            nodes.discard(node_id)
        self.G.remove_node(node_id)
        self._save()

    def export_snapshot(self) -> dict:
        """Export graph state for Data Factory trace. Non-destructive snapshot."""
        top = self.nodes_by_strength(limit=10)
        top_strongest = [{"id": nid, "strength": s} for nid, s in top]
        return {"node_count": self.node_count(), "top_strongest": top_strongest}

    def get_strongest_node_with_reason(self) -> Optional[dict]:
        """Return strongest node and reason for Data Factory trace."""
        top = self.nodes_by_strength(limit=1)
        if not top:
            return None
        nid, strength = top[0]
        return {"id": nid, "reason": f"highest activation after wave (strength={strength})"}
