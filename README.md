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

### Parallel Cloud Generation (Recommended)

Scale trace generation using **GitHub Actions parallel matrix** — free tier, maximum throughput:

1. Go to **Actions** → **Generate Traces — Parallel Cloud Factory** → **Run workflow**
2. Use defaults: **total_cycles: 2000**, **num_jobs: 12**
3. Set **fast: true** for maximum speed (recommended for trace generation; disables 60s throttle on GitHub runners)
4. Each of 12 jobs runs ~167 cycles in parallel; traces use `trace_job{N}_{cycle}.jsonl` (no collisions)
5. Merge job commits all traces in one push

**Estimated time**: ~25–30 min for 2000 traces (12 parallel runners, ~45s/cycle). With fast=true, faster.  
**Trace yield**: ~2000 per run; trigger multiple runs to reach 80k+.  
**Progress trackers** now visible in logs (cycle % per job, "Job X FINISHED chunk").

### Scaling

- **Local**: Run `--mode continuous` overnight; traces accumulate in `data/training/raw/`.
- **GitHub Actions**: Set repo variable `GENERATE_TRACES=true` (Settings → Actions → Variables). The workflow will run the generator and commit traces.
- **Batch**: Use `--mode batch --cycles 1000` for a fixed run; combine multiple runs to reach 80k+ traces.
- **Parallel**: Use `generate-traces-parallel.yml` for 2000 cycles / 12 jobs (recommended for scale).

---

## Phase 2 — Training BoggersTheMind-1

Phase 2 turns raw traces into a clean, ready-to-train dataset for LoRA fine-tuning on Qwen2.5-72B or Llama-3.3-70B.

### 1. Process raw traces

```bash
# Process all traces in data/training/raw/
python scripts/process_training_data.py

# Test with limited traces
python scripts/process_training_data.py --max-traces 100

# Include short cycles (e.g. for testing with synthetic traces)
python scripts/process_training_data.py --min-duration 0
```

Output: `data/training/final/boggersmind-ts-traces-{timestamp}.jsonl` in ShareGPT format.

**Quality filters** (configurable in script): duration ≥ 10s, non-empty synthesis, graph has nodes.

### 2. Upload and fine-tune (Together AI)

```bash
# Set API key
export TOGETHER_API_KEY=your_key

# Dry run: print commands and cost estimate only
python scripts/upload_to_together.py --dry-run

# Upload and create LoRA fine-tuning job
python scripts/upload_to_together.py
```

**LoRA config**: rank=64, alpha=16, dropout=0.05, target_modules=all-linear.

**Base models**:
- `Qwen/Qwen2.5-72B-Instruct` (default)
- `meta-llama/Llama-3.3-70B-Instruct-Reference`

### 3. Manual CLI (if preferred)

```bash
# Upload
together files upload data/training/final/boggersmind-ts-traces-YYYYMMDD-HHMMSS.jsonl

# Create job (replace FILE_ID)
together fine-tuning create \
  --training-file "FILE_ID" \
  --model "Qwen/Qwen2.5-72B-Instruct" \
  --lora \
  --n-epochs 3 \
  --learning-rate 1e-5 \
  --train-on-inputs auto \
  --suffix "boggersmind-ts"
```

### 4. Estimated costs

| Dataset size | Epochs | Est. tokens | Est. cost (LoRA, 70B) |
|--------------|--------|-------------|------------------------|
| 1k traces    | 3      | ~15M        | ~$40                   |
| 10k traces   | 3      | ~150M       | ~$375                  |
| 80k traces   | 3      | ~1.2B       | ~$3,000                |

*Check [together.ai/pricing](https://together.ai/pricing) for current rates. LoRA is cheaper than full fine-tuning.*

### 5. Next steps

- **Validation split**: Use 90/10 train/val; pass `--validation-file` and `--n-evals 10` to the fine-tune job.
- **Fireworks**: Same ShareGPT format; use Fireworks fine-tuning API if preferred.
- **Local (vLLM/Unsloth)**: Export ShareGPT JSONL and use your preferred LoRA trainer.

---

*BoggersTheMind — your mind, online.*
