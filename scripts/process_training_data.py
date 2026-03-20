#!/usr/bin/env python3
"""
Process raw TS traces into ShareGPT-format training data for LoRA fine-tuning.
Filters low-quality traces, converts to conversation format, outputs ready-to-train JSONL.
Exits with code 1 if zero rows pass filters (no empty JSONL left behind).

This data is uniquely powerful: each trace captures the full internal cognitive process
of one autonomous cycle — strongest-node selection, wave propagation, tension detection,
hypothesis generation, and consolidation — exactly as the Thinking System architecture thinks.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Default quality thresholds (Data Factory: filter low-value traces)
DEFAULT_MIN_DURATION_SECONDS = 10.0
MIN_NEW_INSIGHTS = 0  # Keep traces with 0 if synthesis is non-empty (fallback ingest cycles)
MIN_SYNTHESIS_LENGTH = 20
MIN_GRAPH_NODES = 1

SYSTEM_PROMPT = """You are BoggersTheMind, a native Thinking System / Thinking Wave cognitive architecture.

You think via:
1. **Strongest Node** — Select the highest-activation concept in your knowledge graph
2. **Wave Propagation** — Spread influence to neighbors (energy + decay), amplify the seed
3. **Tension Detection** — Identify contrasts and unresolved connections between concepts
4. **Hypothesis Queue** — Generate explorable questions from surfaced nodes and tensions
5. **Consolidation** — Merge similar insights, strengthen the graph

Given graph state, wave steps, tensions, and a selected hypothesis, synthesize a clear, insightful answer grounded in the context. Be concise but thorough. Do not invent facts."""


def _format_user_message(trace: dict) -> str:
    """Build the user-facing context from a trace. Teaches the model the input structure."""
    gs_before = trace.get("graph_state_before") or {}
    strongest = trace.get("strongest_node") or {}
    wave_steps = trace.get("wave_propagation_steps") or []
    tensions = trace.get("tensions_detected") or []
    queue = trace.get("hypothesis_queue") or []
    selected = trace.get("selected_hypothesis", "")
    pipeline = trace.get("query_pipeline_decisions") or {}

    parts = [
        "## Graph State (before cycle)",
        f"Nodes: {gs_before.get('node_count', 0)}",
        f"Top by strength: {[n.get('id') for n in gs_before.get('top_strongest', [])[:5]]}",
        "",
        "## Strongest Node",
        f"ID: {strongest.get('id', '')} — {strongest.get('reason', '')}",
        "",
        "## Wave Propagation",
    ]
    for s in wave_steps[:10]:
        parts.append(f"  {s.get('node', '')}: energy={s.get('energy', 0)}, decay={s.get('decay', 0)}")
    if not wave_steps:
        parts.append("  (none this cycle)")

    parts.extend(["", "## Tensions"])
    for t in tensions[:5]:
        parts.append(f"  {t.get('node_a', '')} ↔ {t.get('node_b', '')}: {t.get('description', '')}")
    if not tensions:
        parts.append("  (none detected)")

    parts.extend(["", "## Hypothesis Queue"])
    for q in queue[:5]:
        parts.append(f"  - {q.get('text', '')}")
    if not queue:
        parts.append("  (empty)")

    parts.extend([
        "",
        "## Query Pipeline",
        f"Wikipedia ingest: {pipeline.get('wikipedia_ingest', False)}",
        f"Sufficiency score: {pipeline.get('sufficiency_score', 0)}",
        "",
        "## Selected Hypothesis",
        selected or "(fallback ingest)",
    ])
    return "\n".join(parts)


def _is_quality_trace(trace: dict, min_duration: float = DEFAULT_MIN_DURATION_SECONDS) -> tuple[bool, str]:
    """Return (keep, reason). Filter low-quality traces."""
    meta = trace.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    duration = meta.get("duration_seconds", 0) or 0
    new_insights = meta.get("new_insights", 0) or 0
    synthesis = (trace.get("synthesis_output") or "").strip()
    gs = trace.get("graph_state_before") or {}
    node_count = gs.get("node_count", 0) or 0

    if duration < min_duration:
        return False, f"duration {duration:.1f}s < {min_duration}s"
    if node_count < MIN_GRAPH_NODES:
        return False, f"graph empty ({node_count} nodes)"
    # Allow traces with no new_insights if synthesis is substantial (e.g. throttled Ollama)
    if len(synthesis) < MIN_SYNTHESIS_LENGTH and new_insights < 1:
        return False, f"synthesis too short ({len(synthesis)} chars) and no new insights"

    return True, "ok"


def process_traces(
    raw_dir: Path,
    output_dir: Path,
    max_traces: int | None = None,
    skip_existing: bool = True,
    min_duration: float = DEFAULT_MIN_DURATION_SECONDS,
) -> dict:
    """
    Read raw traces, filter, convert to ShareGPT, write output.
    Idempotent: uses timestamp in output filename. Resumable: can skip already-processed.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all raw trace files (resumable: process in deterministic order)
    trace_files = sorted(raw_dir.glob("trace_*.jsonl"))
    total = len(trace_files)

    kept = 0
    dropped = 0
    drop_reasons: dict[str, int] = {}
    lengths = []

    # Output filename with timestamp (idempotent: each run produces one file)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = output_dir / f"boggersmind-ts-traces-{timestamp}.jsonl"

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, fp in enumerate(trace_files):
            if max_traces is not None and kept >= max_traces:
                break

            try:
                line = fp.read_text(encoding="utf-8").strip()
                if not line:
                    continue
                trace = json.loads(line)
            except (json.JSONDecodeError, OSError) as e:
                dropped += 1
                drop_reasons["parse_error"] = drop_reasons.get("parse_error", 0) + 1
                continue

            ok, reason = _is_quality_trace(trace, min_duration=min_duration)
            if not ok:
                dropped += 1
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                continue

            # Convert to ShareGPT format
            user_msg = _format_user_message(trace)
            synthesis = (trace.get("synthesis_output") or "").strip()
            if not synthesis:
                synthesis = "(No synthesis output — cycle may have been throttled or failed.)"

            sharegpt = {
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "user", "value": user_msg},
                    {"from": "assistant", "value": synthesis},
                ],
            }
            # Preserve trace metadata for debugging
            sharegpt["_meta"] = {
                "cycle_id": trace.get("cycle_id"),
                "timestamp": trace.get("timestamp"),
                "duration_seconds": (trace.get("metadata") or {}).get("duration_seconds"),
            }

            fout.write(json.dumps(sharegpt, ensure_ascii=False) + "\n")
            kept += 1
            lengths.append(len(user_msg) + len(synthesis))

    avg_len = sum(lengths) / len(lengths) if lengths else 0
    return {
        "total_files": total,
        "kept": kept,
        "dropped": dropped,
        "drop_reasons": drop_reasons,
        "avg_length": avg_len,
        "output_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process raw TS traces into ShareGPT training data for LoRA fine-tuning.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/training/raw"),
        help="Directory containing trace_*.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/final"),
        help="Output directory for processed JSONL",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Limit number of traces to process (for testing)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Process all files (default: process all; skip_existing not used for raw)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION_SECONDS,
        help=f"Minimum cycle duration in seconds (default {DEFAULT_MIN_DURATION_SECONDS})",
    )
    args = parser.parse_args()

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print("[bold]TS Training Data Processor[/bold] - raw to ShareGPT")
    console.print(f"Raw dir: [cyan]{args.raw_dir}[/cyan]")
    console.print(f"Output dir: [cyan]{args.output_dir}[/cyan]")
    if args.max_traces:
        console.print(f"Max traces: [yellow]{args.max_traces}[/yellow] (test mode)")

    stats = process_traces(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        max_traces=args.max_traces,
        skip_existing=not args.no_skip,
        min_duration=args.min_duration,
    )

    table = Table(title="Processing Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total raw files", str(stats["total_files"]))
    table.add_row("Kept", str(stats["kept"]))
    table.add_row("Dropped", str(stats["dropped"]))
    table.add_row("Avg length (chars)", f"{stats['avg_length']:.0f}")
    table.add_row("Output", stats["output_path"])
    console.print(table)

    if stats["drop_reasons"]:
        console.print("\n[dim]Drop reasons:[/dim]")
        for reason, count in sorted(stats["drop_reasons"].items(), key=lambda x: -x[1]):
            console.print(f"  {reason}: {count}")

    if stats["kept"] == 0:
        out_path = Path(stats["output_path"])
        if out_path.exists() and out_path.stat().st_size == 0:
            out_path.unlink()
        console.print()
        console.print(Panel.fit(
            "[bold red]ZERO ROWS KEPT[/bold red]\n\n"
            "No traces passed quality filters (synthesis length, graph state, duration, etc.). "
            "Downstream training would be empty - fix raw traces or relax filters.\n\n"
            "[dim]Exiting with code 1 so CI / full_cloud_train fails fast.[/dim]",
            title="process_training_data.py",
            border_style="red",
        ))
        sys.exit(1)

    console.print(f"\n[green]Done.[/green] Ready-to-train data: [cyan]{stats['output_path']}[/cyan]")


if __name__ == "__main__":
    main()
