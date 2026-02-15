# transformer-structural-compression

Structural compression of Transformer models via low-rank approximations, with an emphasis on **how semantic information degrades across layers** when QKV projections are rank-constrained.

This repository is designed to be:
- **reproducible** (configs + outputs)
- **TFG-friendly** (clear pipeline, clean metrics, strong analysis)
- **extendable** (later: Tucker / pruning / quantization if needed)

## Project goals
1. Train a strong **baseline** on a semantic NLP task (GoEmotions).
2. Apply **rank constraints** to attention projections (Q/K/V) and measure:
   - performance (macro/micro F1, per-class F1)
   - compression (parameter reduction)
   - representation drift (CLS embeddings, cosine similarity, geometry/UMAP)
3. Analyze **layer sensitivity** and identify redundancy patterns.

## Repository structure
- `src/tsc/` : reusable library code
- `scripts/` : runnable scripts (train/eval/sweep)
- `configs/` : experiment configs (YAML)
- `outputs/` : saved models, metrics, embeddings (ignored by git)
- `docs/` : thesis notes + figures

## Setup

### Option A — venv (simple)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements-dev.txt
```