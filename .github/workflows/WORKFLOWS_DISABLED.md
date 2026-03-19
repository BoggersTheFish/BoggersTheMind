# GitHub Workflows — Disabled

Workflows are currently **disabled** (files renamed to `.disabled`).

**Reason:** At this stage, running autonomous exploration or trace generation on GitHub Actions is inefficient and a waste of time:

- **Autonomous** — Free runners have no GPU; Ollama runs on CPU, cycles are slow. Better to run locally or on a rented pod.
- **Parallel traces** — Same issue: CPU-only, slow. Phase 3 (`run_on_pod.sh` + `full_cloud_train.py`) does everything on a real GPU pod in one shot for ~$15–$45.

**To re-enable:** Rename `*.yml.disabled` back to `*.yml` when/if you want scheduled or manual runs again.
