# BoggersTheMind

**The final unified Grok-like personal cognitive system.**

One run. One terminal. One mind. No multiple windows, no separate steps — just you and your thoughts, grounded in a living knowledge graph.

---

## What It Is

BoggersTheMind is a **local-first cognitive assistant** that:

- Maintains a **Universal Living Graph** of concepts and relationships
- Runs an autonomous **Strongest Node wave** to surface insights
- Watches your **Obsidian vault** and syncs bidirectionally
- Ingests from **Wikipedia** and reasons with **Ollama** (llama3.2)
- Merges similar insights and writes clean `.md` notes
- Presents **one unified TUI**: status + feed + chat — all in a single terminal

Type naturally. Get immediate, insightful responses grounded in your live graph. The mind runs silently in the background while you chat.

---

## Autonomous Mode (Default)

BoggersTheMind **boots in Autonomous Mode**: it continuously generates hypotheses from current insights and tensions, then explores them using the same query pipeline (topic extraction → indexed graph search → sufficiency check → research if needed → synthesis → consolidation).

**When you type**: The mind finishes its current safe cycle, switches to **User Mode**, answers your query, then returns to **Autonomous Mode**.

Insight, consolidation, and simulation feed new hypotheses back into the explorer. The TUI shows the current mode: *Autonomous Mode • Exploring…* or *User Mode*.

---

## Intelligent Query Handling

Every user message flows through a **single intelligent pipeline**:

1. **Topic breakdown** — Extract 2–5 clean topics from query (fast, no LLM)
2. **Topic-indexed graph search** — Fast lookup via topic filing system (indexed clusters)
3. **Sufficiency check** — Score knowledge (node count + activation + recency)
4. **Conditional research** — If insufficient → ingest Wikipedia for missing topics until contextualized
5. **Synthesis** — Final answer grounded in graph + any new research

Answer appears instantly in chat. Background autonomous loop (ingest/consolidate) continues silently.

---

## One-Command Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running with llama3.2
ollama pull llama3.2

# Launch
./run.bat    # Windows
./run.sh     # Linux / macOS
```

---

## Architecture

| Component | Role |
|-----------|------|
| `core/graph.py` | UniversalLivingGraph — nodes, edges, topic filing system (indexed clusters) |
| `core/mode_manager.py` | AUTO ↔ USER mode with safe handoff (finish cycle before switch) |
| `core/autonomous_explorer.py` | Generates hypotheses, feeds into query pipeline, consolidation |
| `core/query_processor.py` | Intelligent query pipeline (topics → search → research → synthesize) |
| `core/wave.py` | 5-step Strongest Node wave (supports topic clustering) |
| `core/bridge.py` | Vault watcher (Obsidian ↔ graph) |
| `entities/ingestor.py` | Wikipedia ingestion (research fallback) |
| `entities/inference.py` | Ollama reasoning — topic breakdown + synthesis (throttled 1/60s) |
| `entities/simulation.py` | Hypothesis simulation — feeds follow-ups to explorer |
| `entities/consolidation.py` | Merges similar insights — feeds new hypotheses to explorer |
| `entities/insight.py` | Writes .md notes to vault — feeds follow-ups to explorer |
| `interface/tui.py` | Single Rich TUI (status + feed + chat + mode indicator) |

---

## Throttling (Laptop-Friendly)

- **Background cycle**: every 45 seconds
- **Ollama**: max 1 call per 60 seconds
- Cool, always-on, battery-conscious

---

## Obsidian Vault

Notes live at `obsidian/TS-Knowledge-Vault`. The bridge watches for changes and syncs into the graph. Insights written by the mind appear as clean markdown files.

---

## GitHub Actions (Cloud Auto Mode)

Push the repo to GitHub to enable **headless autonomous exploration** every 5 minutes:

- No TUI, no user input
- Runs the same autonomous explorer (hypothesis → query pipeline → consolidation → write insights)
- Commits new insights to `obsidian/TS-Knowledge-Vault` automatically
- Syncs existing vault on cold start

Enable the workflow, grant `contents: write`, and push. Your mind explores in the cloud.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) with `llama3.2` (local runs)
- Internet (Wikipedia API)

---

## Phase 1 — TS Data Factory

Phase 1 generates 80k–150k high-quality structured reasoning traces for future LoRA fine-tuning of BoggersTheMind-1. Each trace captures the *internal cognitive process* of one full autonomous cycle.

### Run commands

```bash
# Continuous mode — runs forever, saves every cycle (Ctrl+C to stop gracefully)
python scripts/generate_ts_synthetic_data.py --mode continuous

# Batch mode — run exactly N cycles (e.g. 1000)
python scripts/generate_ts_synthetic_data.py --mode batch --cycles 1000

# GitHub Actions — optimised for CI (~4 cycles, fits timeout)
python scripts/generate_ts_synthetic_data.py --mode github
```

### TUI / headless with tracing

```bash
# TUI with trace generation enabled
python mind.py --trace

# Headless with trace generation
python mind.py --headless --trace
```

### Output

Traces are saved to `data/training/raw/trace_{cycle_id}.jsonl`. Each JSONL line has the full schema: `timestamp`, `cycle_id`, `graph_state_before`, `strongest_node`, `wave_propagation_steps`, `tensions_detected`, `hypothesis_queue`, `selected_hypothesis`, `query_pipeline_decisions`, `synthesis_output`, `graph_state_after`, `final_obsidian_note_path`, `metadata`.

### Scaling

- **Local**: Run `--mode continuous` overnight; traces accumulate in `data/training/raw/`.
- **GitHub Actions**: Set repo variable `GENERATE_TRACES=true` (Settings → Actions → Variables). The workflow will run the generator and commit traces.
- **Batch**: Use `--mode batch --cycles 1000` for a fixed run; combine multiple runs to reach 80k+ traces.

---

*BoggersTheMind — your mind, online.*
