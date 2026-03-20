#!/usr/bin/env python3
"""
Upload processed TS training data to Together AI and create a LoRA fine-tuning job.
Prints the exact CLI command and estimated cost. Requires TOGETHER_API_KEY.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# LoRA config for BoggersTheMind-1 (user-specified)
LORA_RANK = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"

# Base models (Together AI LoRA fine-tuning supported)
MODELS = {
    "qwen72b": "Qwen/Qwen2.5-72B-Instruct",
    "llama70b": "meta-llama/Llama-3.3-70B-Instruct-Reference",
}

# Rough per-token cost for LoRA SFT (70B class) — check together.ai/pricing for current rates
# Ballpark: ~$2.50 per 1M tokens for LoRA on 70B models
COST_PER_M_TOKENS = 2.50


def _estimate_tokens(jsonl_path: Path) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    total_chars = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                total_chars += len(line)
    if total_chars == 0:
        return 0
    return max(1, total_chars // 4)


def _count_training_examples(jsonl_path: Path) -> Tuple[int, Optional[str]]:
    """
    Count JSONL rows with non-empty ShareGPT conversations or chat messages.
    Returns (count, None) if ok; (0, reason) if file is empty or unusable.
    """
    if not jsonl_path.exists():
        return 0, "file does not exist"
    if jsonl_path.stat().st_size == 0:
        return 0, "file is empty (0 bytes)"
    n = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = obj.get("conversations")
            msgs = obj.get("messages")
            turns = conv if isinstance(conv, list) else msgs if isinstance(msgs, list) else None
            if turns and len(turns) > 0:
                n += 1
    if n == 0:
        return 0, "0 training examples (no valid JSONL rows with conversations/messages)"
    return n, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload TS training data to Together AI and create LoRA fine-tuning job.",
    )
    parser.add_argument(
        "data_file",
        type=Path,
        nargs="?",
        default=None,
        help="Processed JSONL file (default: latest in data/training/final/)",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="qwen72b",
        help="Base model: qwen72b or llama70b",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="boggersmind-ts",
        help="Suffix for fine-tuned model name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and cost estimate only, do not upload",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if the data file is empty or has no training examples (also applies to dry-run)",
    )
    args = parser.parse_args()

    # Resolve data file
    if args.data_file is None:
        final_dir = Path("data/training/final")
        if not final_dir.exists():
            print("Error: data/training/final/ not found. Run process_training_data.py first.", file=sys.stderr)
            sys.exit(1)
        candidates = sorted(final_dir.glob("boggersmind-ts-traces-*.jsonl"), reverse=True)
        if not candidates:
            print("Error: No boggersmind-ts-traces-*.jsonl in data/training/final/. Run process_training_data.py first.", file=sys.stderr)
            sys.exit(1)
        data_file = candidates[0]
    else:
        data_file = Path(args.data_file)
        if not data_file.exists():
            print(f"Error: {data_file} not found.", file=sys.stderr)
            sys.exit(1)

    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    n_examples, bad_reason = _count_training_examples(data_file)
    if n_examples == 0:
        console.print()
        console.print(Panel.fit(
            "[bold yellow]WARNING[/bold yellow]  [yellow]Selected file is empty or has 0 training examples - "
            "training will fail.[/yellow]\n\n"
            f"[dim]{data_file}[/dim]\n"
            f"[dim]{bad_reason}[/dim]\n\n"
            "[dim]Use --strict to exit non-zero in CI; fix the file or run process_training_data.py first.[/dim]",
            title="upload_to_together.py",
            border_style="yellow",
        ))
        if args.strict:
            sys.exit(1)
        if not args.dry_run:
            sys.exit(1)

    model_str = MODELS[args.model]
    est_tokens = _estimate_tokens(data_file)
    n_epochs = args.n_epochs
    total_tokens = est_tokens * n_epochs
    est_cost = (total_tokens / 1_000_000) * COST_PER_M_TOKENS

    from rich.table import Table

    console.print(Panel.fit(
        "[bold]Together AI LoRA Fine-Tuning - BoggersTheMind-1[/bold]",
        border_style="cyan",
    ))
    console.print(f"Data file: [cyan]{data_file}[/cyan]")
    console.print(f"Base model: [cyan]{model_str}[/cyan]")
    console.print(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}, target_modules={LORA_TARGET_MODULES}")

    table = Table(title="Cost Estimate")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Est. tokens (approx)", f"{est_tokens:,}")
    table.add_row("Epochs", str(n_epochs))
    table.add_row("Total tokens", f"{total_tokens:,}")
    table.add_row("Est. cost (LoRA SFT)", f"~${est_cost:.2f} (check together.ai/pricing)")
    console.print(table)

    # ShareGPT uses "conversations" with "from"/"value" — Together expects "messages" with "role"/"content"
    # Convert on-the-fly when uploading, or use a converted file
    console.print("\n[dim]Note: ShareGPT format (conversations/from/value) must become Together JSONL (messages/role/content).[/dim]")
    console.print("[dim]This script converts automatically when you upload (non-dry-run); for manual CLI, convert lines the same way as in main() below.[/dim]")

    # Exact CLI command (user runs after manual upload)
    api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key and not args.dry_run:
        console.print("\n[yellow]TOGETHER_API_KEY not set. Set it to upload and create jobs.[/yellow]")

    console.print("\n[bold]Step 1 — Upload file:[/bold]")
    console.print("  together files upload \\")
    console.print(f'    "{data_file.resolve()}"')
    console.print("\n[bold]Step 2 — Create fine-tuning job (replace FILE_ID with upload response):[/bold]")
    cmd = f"""together fine-tuning create \\
  --training-file "FILE_ID" \\
  --model "{model_str}" \\
  --lora \\
  --lora-r {LORA_RANK} \\
  --lora-alpha {LORA_ALPHA} \\
  --lora-dropout {LORA_DROPOUT} \\
  --lora-trainable-modules {LORA_TARGET_MODULES} \\
  --n-epochs {n_epochs} \\
  --learning-rate 1e-5 \\
  --train-on-inputs auto \\
  --suffix "{args.suffix}" """
    console.print(f"  {cmd}")

    # Python API alternative
    console.print("\n[bold]Python API (after converting ShareGPT to messages):[/bold]")
    console.print("""
  from together import Together
  client = Together(api_key=os.environ["TOGETHER_API_KEY"])
  resp = client.files.upload("your-converted.jsonl", purpose="fine-tune", check=True)
  ft = client.fine_tuning.create(
      training_file=resp.id,
      model="Qwen/Qwen2.5-72B-Instruct",
      training_type={
          "type": "Lora",
          "lora_r": 64,
          "lora_alpha": 16,
          "lora_dropout": 0.05,
          "lora_trainable_modules": "all-linear",
      },
      n_epochs=3,
      learning_rate=1e-5,
      train_on_inputs="auto",
      suffix="boggersmind-ts",
  )
  print(ft.id, ft.output_name)
""")

    if args.dry_run:
        if n_examples == 0:
            console.print("\n[yellow]Dry run finished (see warning above). No upload performed.[/yellow]")
        else:
            console.print("\n[green]Dry run complete. No upload performed.[/green]")
        return

    # Optional: convert and upload
    try:
        from together import Together
    except ImportError:
        console.print("\n[yellow]Install: pip install together[/yellow]")
        return

    if not api_key:
        console.print("\n[yellow]Set TOGETHER_API_KEY to enable upload.[/yellow]")
        return

    # Convert ShareGPT to Together messages format
    converted_path = data_file.with_suffix(".together.jsonl")
    line_count = 0
    with open(data_file, encoding="utf-8") as fin, open(converted_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = obj.get("conversations", [])
            messages = []
            for c in conv:
                role = c.get("from", "").lower()
                if role == "system":
                    messages.append({"role": "system", "content": c.get("value", "")})
                elif role == "user":
                    messages.append({"role": "user", "content": c.get("value", "")})
                elif role == "assistant":
                    messages.append({"role": "assistant", "content": c.get("value", "")})
            if messages:
                fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                line_count += 1

    console.print(f"\n[green]Converted {line_count} examples to Together format: {converted_path}[/green]")

    client = Together(api_key=api_key)
    console.print("Uploading...")
    upload_resp = client.files.upload(str(converted_path), purpose="fine-tune", check=True)
    console.print(f"[green]Uploaded. File ID: {upload_resp.id}[/green]")

    console.print("Creating fine-tuning job...")
    # Together API: lora=True; lora_r, lora_alpha, etc. for LoRA config (rank=64, alpha=16)
    ft_resp = client.fine_tuning.create(
        training_file=upload_resp.id,
        model=model_str,
        lora=True,
        lora_r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_trainable_modules=LORA_TARGET_MODULES,
        n_epochs=n_epochs,
        learning_rate=1e-5,
        train_on_inputs="auto",
        suffix=args.suffix,
    )
    console.print(f"[green]Job created: {ft_resp.id}[/green]")
    console.print(f"Monitor: together fine-tuning retrieve {ft_resp.id}")
    console.print(f"Output model: {getattr(ft_resp, 'output_name', 'N/A')}")


if __name__ == "__main__":
    main()
