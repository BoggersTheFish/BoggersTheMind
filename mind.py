"""
BoggersTheMind — main entry. Autonomous explorer + mode manager + unified TUI.
Boots in Autonomous Mode. User types → finish cycle → User Mode → answer → return to Auto.
"""

import threading
from pathlib import Path

from core.graph import UniversalLivingGraph
from core.bridge import VaultWatcher, VAULT_PATH
from core.mode_manager import ModeManager, Mode
from core.autonomous_explorer import run_autonomous_explorer
from core.query_processor import process_query
from interface.tui import TUIState, run_unified_tui


def heartbeat(state: TUIState, graph: UniversalLivingGraph) -> None:
    """Background heartbeat: check connectivity, update status."""
    import time
    while True:
        try:
            import requests
            requests.get("https://en.wikipedia.org", timeout=3)
            state.connected = True
        except Exception:
            state.connected = False

        try:
            import ollama
            ollama.Client(timeout=5.0).list()
            state.ollama_ready = True
        except Exception:
            state.ollama_ready = False

        time.sleep(15)


def on_vault_change(path: str, content: str, graph: UniversalLivingGraph, state: TUIState) -> None:
    """When vault file changes, add/update node in graph."""
    import time
    try:
        name = Path(path).stem
        node_id = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower())[:64]
        if not node_id:
            node_id = f"vault_{int(time.time())}"
        graph.add_node(node_id, label=name, content=content[:500], source="vault", strength=1.0)
        state.feed_items.append(f"[dim]Vault: {name}[/dim]")
    except Exception:
        pass


def on_user_message(
    user_msg: str,
    graph: UniversalLivingGraph,
    state: TUIState,
    mode_manager: ModeManager,
):
    """
    Handle user chat: request user mode, wait for safe handoff, process query, return to auto.
    """
    mode_manager.request_user_mode()
    mode_manager.wait_for_safe_handoff()

    state.mode = Mode.USER
    result = process_query(graph, user_msg, state=state)
    mode_manager.return_to_auto_mode()
    state.mode = Mode.AUTO

    return result


def _sync_vault_to_graph(graph: UniversalLivingGraph) -> None:
    """Load existing vault .md files into graph (for headless cold start)."""
    import time
    if not VAULT_PATH.exists():
        return
    for p in VAULT_PATH.glob("*.md"):
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            name = p.stem
            node_id = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower())[:64]
            if not node_id:
                node_id = f"vault_{int(time.time())}"
            graph.add_node(node_id, label=name, content=content[:500], source="vault", strength=1.0)
        except Exception:
            pass


def main(headless: bool = False) -> None:
    """Main entry. headless=True for GitHub Actions (no TUI)."""
    Path("data").mkdir(exist_ok=True)
    VAULT_PATH.mkdir(parents=True, exist_ok=True)

    graph = UniversalLivingGraph()
    if headless:
        _sync_vault_to_graph(graph)
    state = TUIState()
    mode_manager = ModeManager()

    # Vault watcher (skip in headless to avoid file watcher issues)
    watcher = None
    if not headless:
        def vault_cb(path: str, content: str) -> None:
            on_vault_change(path, content, graph, state)

        watcher = VaultWatcher(vault_cb)
        watcher.start()

    # Background threads
    t_heartbeat = threading.Thread(target=heartbeat, args=(state, graph), daemon=True)
    t_explorer = threading.Thread(
        target=run_autonomous_explorer,
        args=(graph, mode_manager),
        kwargs={"state": state, "headless": headless},
        daemon=True,
    )
    t_heartbeat.start()
    t_explorer.start()

    if headless:
        # GitHub Actions: run for ~3 minutes (explorer does cycles), then exit
        import time
        for _ in range(4):
            time.sleep(45)
    else:
        # Unified TUI — blocks until user exits
        state.mode = Mode.AUTO

        def on_input(msg: str):
            return on_user_message(msg, graph, state, mode_manager)

        run_unified_tui(state, on_input)

    if watcher:
        watcher.stop()


if __name__ == "__main__":
    import sys
    # Data Factory: --trace enables TS trace generation for future model training
    if "--trace" in sys.argv:
        from core.tracer import set_tracing_enabled
        set_tracing_enabled(True)
    main(headless="--headless" in sys.argv)
