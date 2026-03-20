#!/usr/bin/env python3
"""
Full Cloud Training Pipeline — end-to-end BoggersTheMind-1 on a rented GPU pod.
Step 1: Generate traces (fast mode)
Step 2: Process to ShareGPT
Step 3: Unsloth QLoRA fine-tuning
Saves model as boggersmind-1. One command does everything.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Model configs for Unsloth QLoRA (cheap GPU: RTX 4090 / A100 40GB)
MODELS = {
    "qwen14b": "unsloth/Qwen2.5-14B-Instruct",
    "llama8b": "unsloth/Meta-Llama-3.1-8B-Instruct",  # 3.3-8B not on Unsloth; 3.1 cheaper
}

# Cost estimates (Vast.ai / RunPod, ~$0.25–$0.45/hr)
HOURLY_RATE = 0.35
CYCLES_PER_HOUR = 80  # ~45s/cycle, fast mode ~30s


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    """Run command, return exit code."""
    return subprocess.run(cmd, cwd=cwd or Path.cwd()).returncode


def _cost_estimate(trace_hrs: float, train_hrs: float) -> float:
    """Rough total cost in USD."""
    return (trace_hrs + train_hrs) * HOURLY_RATE


def _run_unsloth_qlora_training(
    model_name: str,
    train_path: Path,
    output_dir: Path,
    epochs: int,
) -> None:
    """
    Step 3: Load base model in 4-bit, attach LoRA, run SFTTrainer on messages JSONL.
    Must run on a CUDA machine with unsloth + trl installed.
    """
    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    max_seq_length = 2048
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("json", data_files={"train": str(train_path)}, split="train")
    if len(dataset) == 0:
        raise ValueError(f"No rows in {train_path}; check Step 2 output.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    def format_chat(example: dict) -> dict:
        messages = example.get("messages", [])
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(
        format_chat,
        remove_columns=dataset.column_names,
        num_proc=1,
    )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = not bf16

    training_args = SFTConfig(
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        output_dir=str(output_dir),
        optim="adamw_8bit",
        seed=3407,
        fp16=fp16,
        bf16=bf16,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved LoRA + tokenizer to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full cloud pipeline: generate traces → process → Unsloth QLoRA train",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1000,
        help="Number of trace cycles to generate (default 1000)",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="qwen14b",
        help="Base model: qwen14b or llama8b",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs (default 1)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "data" / "training" / "raw"
    final_dir = root / "data" / "training" / "final"

    console = Console()

    # Cost estimate at start
    trace_hrs = args.cycles / CYCLES_PER_HOUR
    train_hrs = 0.5 + (args.epochs * 0.3)  # rough: 1 epoch ~30 min
    est_cost = _cost_estimate(trace_hrs, train_hrs)

    console.print(Panel.fit(
        "[bold]BoggersTheMind-1 — Full Cloud Pipeline[/bold]\n"
        f"Cycles: {args.cycles} | Model: {args.model} | Epochs: {args.epochs}\n"
        f"Est. cost: [cyan]~${est_cost:.1f}[/cyan] ({trace_hrs + train_hrs:.1f} hrs @ ${HOURLY_RATE}/hr)",
        border_style="cyan",
    ))

    # Step 1: Generate traces (fast mode)
    console.print("\n[bold]Step 1/3[/bold] — Generating traces (fast mode)...")
    os.environ["BOGGERS_FAST_MODE"] = "1"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ret = _run([
        sys.executable,
        str(root / "scripts" / "generate_ts_synthetic_data.py"),
        "--mode", "batch",
        "--cycles", str(args.cycles),
        "--fast",
        "--output-dir", str(raw_dir),
    ], cwd=root)
    if ret != 0:
        console.print("[red]Trace generation failed.[/red]")
        sys.exit(1)
    trace_count = len(list(raw_dir.glob("trace_*.jsonl")))
    console.print(f"[green]Generated {trace_count} traces.[/green]")

    # Step 2: Process to ShareGPT
    console.print("\n[bold]Step 2/3[/bold] — Processing to ShareGPT...")
    ret = _run([
        sys.executable,
        str(root / "scripts" / "process_training_data.py"),
        "--min-duration", "0",
    ], cwd=root)
    if ret != 0:
        console.print("[red]Processing failed.[/red]")
        sys.exit(1)

    # Find latest processed file
    final_files = sorted(final_dir.glob("boggersmind-ts-traces-*.jsonl"), reverse=True)
    if not final_files:
        console.print("[red]No processed data found.[/red]")
        sys.exit(1)
    data_path = final_files[0]
    n_lines = sum(1 for _ in open(data_path, encoding="utf-8") if _.strip())
    console.print(f"[green]Processed {n_lines} examples → {data_path.name}[/green]")

    # Step 3: Unsloth QLoRA training
    console.print("\n[bold]Step 3/3[/bold] — Unsloth QLoRA training...")

    # Convert ShareGPT to messages format for Unsloth
    train_path = final_dir / "train_messages.jsonl"
    with open(data_path, encoding="utf-8") as fin, open(train_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = obj.get("conversations", [])
            messages = []
            for c in conv:
                role = c.get("from", "").lower()
                content = c.get("value", "")
                if role == "system":
                    messages.append({"role": "system", "content": content})
                elif role == "user":
                    messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    messages.append({"role": "assistant", "content": content})
            if messages:
                fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    model_name = MODELS[args.model]
    output_dir = root / "outputs" / "boggersmind-1"

    try:
        _run_unsloth_qlora_training(
            model_name=model_name,
            train_path=train_path,
            output_dir=output_dir,
            epochs=args.epochs,
        )
    except Exception as e:
        console.print(f"[red]Unsloth training failed: {e}[/red]")
        sys.exit(1)

    console.print(f"\n[green]Done.[/green] Model saved to [cyan]{output_dir}[/cyan]")
    console.print(Panel(
        "[bold]Download instructions[/bold]\n\n"
        f"From the pod, copy the model:\n"
        f"  scp -r user@pod-ip:{output_dir} ./boggersmind-1\n\n"
        "Or use rsync:\n"
        f"  rsync -avz user@pod-ip:{output_dir}/ ./boggersmind-1/\n\n"
        "Use in Ollama: merge LoRA into base, then create Modelfile.\n"
        "Use in Python: load with transformers + peft from the saved dir.",
        title="boggersmind-1 ready",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
