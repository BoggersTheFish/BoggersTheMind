#!/usr/bin/env python3
"""
TS Data Factory — generates structured reasoning traces for future LoRA fine-tuning.
Runs the autonomous explorer with tracing enabled. Saves to data/training/raw/.
"""

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TS reasoning traces for model training")
    parser.add_argument(
        "--mode",
        choices=["continuous", "batch", "github"],
        default="continuous",
        help="continuous=run forever; batch=run N cycles; github=Actions-optimised",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1000,
        help="For batch mode: number of cycles to run (default 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/raw"),
        help="Output directory for trace JSONL files",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        default=None,
        help="Parallel job ID (0-based). Enables trace_job{N}_{cycle}.jsonl filenames.",
    )
    parser.add_argument(
        "--start-cycle",
        type=int,
        default=None,
        help="Starting cycle_id for this job (parallel chunk offset).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Disable 60s throttle (for GitHub runners; set BOGGERS_FAST_MODE=1)",
    )
    args = parser.parse_args()

    if args.fast:
        os.environ["BOGGERS_FAST_MODE"] = "1"

    # Enable tracing before any imports that use it
    from core.tracer import set_tracing_enabled, set_trace_job_id
    set_tracing_enabled(True, output_dir=args.output_dir)
    if args.job_id is not None:
        set_trace_job_id(args.job_id)

    from core.graph import UniversalLivingGraph
    from core.bridge import VAULT_PATH
    from core.mode_manager import ModeManager
    from core.autonomous_explorer import run_autonomous_explorer
    from interface.tui import TUIState

    from rich.console import Console

    # Legacy Windows consoles (cp1252) cannot render Rich spinners (Braille); avoid Unicode live widgets.
    console = Console(legacy_windows=False) if sys.platform == "win32" else Console()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    graph = UniversalLivingGraph()
    # Sync vault for headless cold start (same as mind._sync_vault_to_graph)
    if VAULT_PATH.exists():
        import time
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

    state = TUIState()
    mode_manager = ModeManager()

    max_cycles = None
    start_cycle = args.start_cycle
    if args.mode == "batch":
        max_cycles = args.cycles
    elif args.mode == "github":
        max_cycles = 4  # ~3 min at 45s/cycle, fits Actions timeout

    shutdown_event = threading.Event()

    def on_shutdown_signal(*_):
        shutdown_event.set()
        console.print("\n[dim]Graceful shutdown requested. Finishing current cycle...[/dim]")

    signal.signal(signal.SIGINT, on_shutdown_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_shutdown_signal)

    console.print(f"[bold]TS Data Factory[/bold] - mode={args.mode}, output={args.output_dir}")
    if args.fast:
        console.print("[yellow]Fast mode enabled (throttle disabled for GitHub)[/yellow]")
    if args.job_id is not None:
        console.print(f"Job [cyan]{args.job_id}[/cyan] starting at cycle [cyan]{start_cycle or 0}[/cyan] ([cyan]{max_cycles or '?'}[/cyan] cycles)")
    elif max_cycles:
        console.print(f"Will run up to [cyan]{max_cycles}[/cyan] cycles.")
    else:
        console.print("Running continuously. Ctrl+C to stop gracefully.")

    # Progress tracker: cycle X/Y (Z%) per job for visibility in logs
    job_id = args.job_id
    def on_cycle(completed: int, total: int) -> None:
        pct = int(100 * completed / total) if total else 0
        prefix = f"[Job {job_id}] " if job_id is not None else ""
        console.print(f"{prefix}Cycle {completed}/{total} completed ({pct}%)")

    on_cycle_cb = on_cycle if max_cycles else None

    console.print("[dim]Generating traces...[/dim]")
    try:
        run_autonomous_explorer(
            graph,
            mode_manager,
            state=state,
            headless=True,
            max_cycles=max_cycles,
            start_cycle=start_cycle,
            on_cycle_complete=on_cycle_cb,
            shutdown_event=shutdown_event,
        )
    except KeyboardInterrupt:
        shutdown_event.set()

    if job_id is not None and max_cycles and not shutdown_event.is_set():
        console.print(f"[Job {job_id}] FINISHED chunk - {max_cycles} cycles completed")

    trace_count = len(list(args.output_dir.glob("trace_*.jsonl")))
    if shutdown_event.is_set():
        console.print(f"\n[yellow]Stopped after graceful shutdown.[/yellow] Traces in [cyan]{args.output_dir}[/cyan]: [cyan]{trace_count}[/cyan]")
    else:
        console.print(f"\n[green]Done.[/green] Traces saved: [cyan]{trace_count}[/cyan] in {args.output_dir}")


if __name__ == "__main__":
    main()
