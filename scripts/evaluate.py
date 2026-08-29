#!/usr/bin/env python3
"""Evaluate JSONL text pairs with local observer/performer checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--observer", required=True)
    ap.add_argument("--performer", required=True)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install the models extra: pip install -e .[models]") from exc
    from binoculars_extension import feature_vector

    tokenizer = AutoTokenizer.from_pretrained(args.observer)
    observer = AutoModelForCausalLM.from_pretrained(args.observer).eval()
    performer = AutoModelForCausalLM.from_pretrained(args.performer).eval()
    rows = []
    with torch.no_grad():
        for line_no, line in enumerate(Path(args.input).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row["text"])
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            ids = encoded["input_ids"][0]
            obs = observer(**encoded).logits[0, :len(ids)].cpu().numpy()
            perf = performer(**encoded).logits[0, :len(ids)].cpu().numpy()
            result = feature_vector(obs, perf, ids.cpu().numpy())
            rows.append({"line": line_no, "label": row.get("label"), **result})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()

