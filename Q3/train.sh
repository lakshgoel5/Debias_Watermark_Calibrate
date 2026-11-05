#!/bin/bash
set -euo pipefail

# Usage: ./train.sh [output_dir]
# Writes feature files, hidden states, and exp3_model.joblib. infer.py loads that file.

out="${1:-../predictions/Q3}"
mkdir -p "$out"

python3 train.py features ../data/public/q3_qa_train.jsonl "$out/train.jsonl"
python3 train.py features ../data/public/q3_qa_dev.jsonl "$out/dev.jsonl"
python3 train.py features ../data/public/q3_qa_dev_ood.jsonl "$out/ood.jsonl"
python3 train.py hidden ../data/public/q3_qa_train.jsonl "$out/train.jsonl" "$out/hid_train.npz"
python3 train.py hidden ../data/public/q3_qa_dev.jsonl "$out/dev.jsonl" "$out/hid_dev.npz"
python3 train.py hidden ../data/public/q3_qa_dev_ood.jsonl "$out/ood.jsonl" "$out/hid_ood.npz"
python3 train.py fit \
  --train "$out/train.jsonl" \
  --dev "$out/dev.jsonl" \
  --ood "$out/ood.jsonl" \
  --train-hid "$out/hid_train.npz" \
  --dev-hid "$out/hid_dev.npz" \
  --ood-hid "$out/hid_ood.npz" \
  --out "$out/exp3_model.joblib"
