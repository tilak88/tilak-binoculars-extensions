# Binoculars Extensions for AI-Text Detection

An independently organized research implementation of zero-shot AI-text detection and token-level feature fusion.

This repository separates the original Binoculars baseline from the extensions developed for the CS590 Deep Learning project. The implementation is written from scratch for this repository; the original method is acknowledged in [`ATTRIBUTION.md`](ATTRIBUTION.md).

## Contributions represented here

- Binoculars observer/performer perplexity and cross-perplexity baseline.
- Single forward-pass token feature extraction: surprisal trajectory curvature (STC), cross-model rank-discordance variance (CMRD-var), rank volatility, roughness, and token agreement rate (TAR).
- Logistic-regression fusion and stratified evaluation utilities.
- Checkpoint-friendly JSONL evaluation design for CNN-Falcon, CNN-Llama, PubMed, and HC3-style paired corpora.
- Explicit negative-result reporting: weak or redundant signals are not promoted as improvements.

The reported project result was 5,422 balanced examples across four datasets. The final presentation reports average AUROC of 0.966 for Binoculars, 0.940 for STC, and 0.709 for CMRD-var; the cumulative HC3 fusion result is 0.9873 to 0.9923. These figures are retained as project evidence, not as universal detector guarantees.

## Project notebook

The original Kaggle run that produced these results is archived in [`notebooks/bino_extension.ipynb`](notebooks/bino_extension.ipynb) ("Orthogonal Binoculars — Optimised Single-Session Run", papermill, python3). It contains the full single-session evaluation across CNN-Falcon, CNN-Llama, PubMed, and HC3 (5,422 samples, 256-token cap), including feature-extraction, metric, and visualisation cells. The packaged `src/` and `scripts/` code in this repository is the cleaned, reusable form of the same implementation. The notebook references Kaggle dataset paths (`/kaggle/input/...`) and is provided for reproducibility evidence; the scripts are the supported entry point.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pytest
```

The included evaluator expects a JSONL file containing `text` and `label` fields and local Hugging Face causal-language-model checkpoints. Large models and datasets are intentionally not committed.

```powershell
python scripts/evaluate.py --input data/sample.jsonl --output results/features.csv `
  --observer tiiuae/falcon-7b --performer tiiuae/falcon-7b-instruct --max-length 256
```

## Repository layout

```text
src/binoculars_extension/  reusable feature and evaluation code
notebooks/                 archived Kaggle project notebook (evidence run)
scripts/                    command-line entry points
tests/                      fast numerical tests
docs/                       method and results notes
```

