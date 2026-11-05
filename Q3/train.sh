#!/bin/bash
set -euo pipefail

# Usage: ./train.sh [output_dir]
# Writes feature files and exp2_model.joblib into predictions/Q3 by default.

out="${1:-../predictions/Q3}"
mkdir -p "$out"

python3 train.py features ../data/public/q3_qa_train.jsonl "$out/train.jsonl"
python3 train.py features ../data/public/q3_qa_dev.jsonl "$out/dev.jsonl"
python3 train.py features ../data/public/q3_qa_dev_ood.jsonl "$out/ood.jsonl"
python3 train.py fit \
  --train "$out/train.jsonl" \
  --dev "$out/dev.jsonl" \
  --ood "$out/ood.jsonl" \
  --out "$out/exp2_model.joblib"
