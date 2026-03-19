"""
ThinkingTrace — Data Factory instrumentation for future model training.
Captures the full internal cognitive process of one autonomous cycle.
Non-intrusive, thread-safe via contextvars. Only active when tracing enabled.
"""

import json
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Thread-safe: each thread/context has its own trace state
_current_trace: ContextVar[Optional[dict]] = ContextVar("thinking_trace", default=None)
_tracing_enabled: ContextVar[bool] = ContextVar("tracing_enabled", default=False)

# Module-level flag for headless/CLI (no contextvars in fresh threads)
_TRACING_ENABLED_GLOBAL: bool = False
_TRACE_OUTPUT_DIR: Path = Path("data/training/raw")
_TRACE_JOB_ID: Optional[int] = None  # When set, filenames use trace_job{N}_{cycle}.jsonl (parallel mode)


def set_tracing_enabled(enabled: bool, output_dir: Optional[Path] = None, job_id: Optional[int] = None) -> None:
    """Enable or disable tracing globally. Data Factory instrumentation for future model training."""
    global _TRACING_ENABLED_GLOBAL, _TRACE_OUTPUT_DIR, _TRACE_JOB_ID
    _TRACING_ENABLED_GLOBAL = enabled
    if output_dir is not None:
        _TRACE_OUTPUT_DIR = Path(output_dir)
    _TRACE_JOB_ID = job_id


def set_trace_job_id(job_id: Optional[int] = None) -> None:
    """Set job_id for parallel trace filenames (trace_job{N}_{cycle}.jsonl). Prevents collisions."""
    global _TRACE_JOB_ID
    _TRACE_JOB_ID = job_id


def is_tracing_enabled() -> bool:
    """Check if tracing is active (global or context)."""
    try:
        return _tracing_enabled.get() or _TRACING_ENABLED_GLOBAL
    except LookupError:
        return _TRACING_ENABLED_GLOBAL


def get_current_trace() -> Optional[dict]:
    """Get the active trace dict for this context, or None if not tracing."""
    try:
        return _current_trace.get()
    except LookupError:
        return None


def _ensure_trace() -> dict:
    """Get or create trace. Call only when tracing is enabled."""
    t = get_current_trace()
    if t is None:
        t = {}
        try:
            _current_trace.set(t)
        except LookupError:
            pass
    return t


def record_graph_state_before(snapshot: dict) -> None:
    """Record graph state at cycle start. Data Factory instrumentation."""
    if not is_tracing_enabled():
        return
    _ensure_trace()["graph_state_before"] = snapshot


def record_strongest_node(node: dict) -> None:
    """Record strongest node with reason. Data Factory instrumentation."""
    if not is_tracing_enabled():
        return
    _ensure_trace()["strongest_node"] = node


def record_wave_propagation_steps(steps: list[dict]) -> None:
    """Record wave propagation steps (node, energy, decay). Data Factory instrumentation."""
    if not is_tracing_enabled():
        return
    _ensure_trace()["wave_propagation_steps"] = steps


def record_tensions_detected(tensions: list[dict]) -> None:
    """Record detected tensions. Data Factory instrumentation."""
    if not is_tracing_enabled():
        return
    _ensure_trace()["tensions_detected"] = tensions


def record_hypothesis_queue(queue: list[dict]) -> None:
    """Record hypothesis queue (text, score). Data Factory instrumentation."""
    if not is_tracing_enabled():
        return
    _ensure_trace()["hypothesis_queue"] = queue


def record_selected_hypothesis(text: str) -> None:
    """Record the hypothesis selected for this cycle. Data Factory instrumentation."""
    if not is_tracing_enabled():
        return
    _ensure_trace()["selected_hypothesis"] = text


def record_query_pipeline_decisions(decisions: dict) -> None:
    """Record query pipeline decisions (wikipedia_ingest, sufficiency_score, etc). Data Factory instrumentation."""
    if not is_tracing_enabled():
        return
    _ensure_trace()["query_pipeline_decisions"] = decisions


def record_synthesis_output(text: str) -> None:
    """Record synthesis output from inference. Data Factory instrumentation."""
    if not is_tracing_enabled():
        return
    _ensure_trace()["synthesis_output"] = text


def record_graph_state_after(snapshot: dict) -> None:
    """Record graph state at cycle end. Data Factory instrumentation."""
    if not is_tracing_enabled():
        return
    _ensure_trace()["graph_state_after"] = snapshot


def record_final_obsidian_note_path(path: str) -> None:
    """Record final Obsidian note path. Data Factory instrumentation."""
    if not is_tracing_enabled():
        return
    _ensure_trace()["final_obsidian_note_path"] = path


def record_metadata(meta: dict) -> None:
    """Record cycle metadata (new_insights, duration_seconds, consolidation_score). Merges with existing. Data Factory instrumentation."""
    if not is_tracing_enabled():
        return
    t = _ensure_trace()
    existing = t.get("metadata") or {}
    t["metadata"] = {**existing, **meta} if isinstance(existing, dict) else meta


def _defaults_for_schema() -> dict:
    """Ensure every required schema field has a value."""
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cycle_id": 0,
        "graph_state_before": {"node_count": 0, "top_strongest": []},
        "strongest_node": {"id": "", "reason": ""},
        "wave_propagation_steps": [],
        "tensions_detected": [],
        "hypothesis_queue": [],
        "selected_hypothesis": "",
        "query_pipeline_decisions": {"wikipedia_ingest": False, "sufficiency_score": 0.0},
        "synthesis_output": "",
        "graph_state_after": {"new_nodes": 0, "new_edges": 0, "strength_changes": {}},
        "final_obsidian_note_path": "",
        "metadata": {"new_insights": 0, "duration_seconds": 0.0, "consolidation_score": 0.0},
    }


def _fill_missing(trace: dict) -> dict:
    """Merge trace with schema defaults so every field is present."""
    base = _defaults_for_schema()
    for k, v in trace.items():
        if v is not None and k in base:
            base[k] = v
    return base


def save_trace(trace: dict, output_dir: Optional[Path] = None) -> Optional[Path]:
    """Write trace as single-line JSONL. Returns path if saved."""
    output_dir = output_dir or _TRACE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    cycle_id = trace.get("cycle_id", 0)
    # Parallel mode: trace_job{N}_{cycle}.jsonl to prevent collisions across jobs
    if _TRACE_JOB_ID is not None:
        fname = f"trace_job{_TRACE_JOB_ID:02d}_{cycle_id:05d}.jsonl"
        filepath = output_dir / fname
    else:
        fname = f"trace_{cycle_id}.jsonl"
        filepath = output_dir / fname
    filled = _fill_missing(trace)
    filepath.write_text(json.dumps(filled, ensure_ascii=False) + "\n", encoding="utf-8")
    # Tracker: visible in GitHub Actions logs
    if _TRACE_JOB_ID is not None:
        print(f"[Job {_TRACE_JOB_ID}] Saved {fname}", flush=True)
    return filepath


class ThinkingTrace:
    """
    Context manager for capturing one full autonomous cycle.
    Data Factory instrumentation for future model training.
    Auto-captures timestamp, cycle_id; other modules fill via record_* hooks.
    """

    def __init__(self, cycle_id: int, output_dir: Optional[Path] = None):
        self.cycle_id = cycle_id
        self.output_dir = output_dir or _TRACE_OUTPUT_DIR
        self._trace: dict = {}
        self._start_time: Optional[float] = None
        self._token = None

    def __enter__(self) -> "ThinkingTrace":
        if not is_tracing_enabled():
            return self
        import time
        self._start_time = time.perf_counter()
        self._trace = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cycle_id": self.cycle_id,
        }
        self._token = _current_trace.set(self._trace)
        _tracing_enabled.set(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not is_tracing_enabled() or not self._trace:
            if self._token is not None:
                try:
                    _current_trace.reset(self._token)
                except (ValueError, LookupError):
                    pass
            return
        import time
        duration = time.perf_counter() - self._start_time if self._start_time else 0.0
        meta = self._trace.get("metadata") or {}
        if isinstance(meta, dict) and "duration_seconds" not in meta:
            meta = {**meta, "duration_seconds": duration}
            self._trace["metadata"] = meta
        try:
            _current_trace.reset(self._token)
        except (ValueError, LookupError):
            pass
        save_trace(self._trace, self.output_dir)
