"""STAIR milestone 2: LoRA fine-tune on Gemma2 (local, Apple Silicon).

Builds SFT examples from benchmarks/queries.json (query -> ToC leaf path) and
fine-tunes google/gemma-2-9b-it (or 2b) with LoRA. Saves the adapter to
outputs/stair-lora-adapter. No bitsandbytes (MPS-friendly): bf16 + LoRA (not QLoRA).

Usage:
  /Users/haris/.local/share/qlora-venv/bin/python scripts/finetune_stair.py \
      --model google/gemma-2-2b-it --epochs 3 --out outputs/stair-lora-adapter
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def build_dataset(queries_path: Path) -> Dataset:
    rows = json.loads(queries_path.read_text())
    examples = []
    for r in rows:
        prompt = (
            "You are a search assistant. Given a user query, answer with the single "
            "table-of-contents leaf path (section number + title) that best answers it.\n\n"
            f"Query: {r['query']}\n\nLeaf:"
        )
        examples.append({"text": f"{prompt} {r['gold_leaf']}"})
    return Dataset.from_list(examples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--queries", default="benchmarks/queries.json")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="outputs/stair-lora-adapter")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    queries_path = (root / args.queries).resolve()
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(queries_path)
    print(f"loaded {len(ds)} training examples from {queries_path}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="mps")
    model.config.use_cache = False

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        max_length=512,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"adapter saved to {out_dir}")


if __name__ == "__main__":
    main()

