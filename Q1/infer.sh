#!/bin/bash
set -euo pipefail

# Usage: ./infer.sh bias <absolute_path_to_output_dir> <input_jsonl>

# Action: writes output_bias.jsonl to the provided output directory
# Format: {"id": "q1_bbq_test_00000", "pred_idx": 2}
#
# Usage: ./infer.sh qa <absolute_path_to_output_dir> <input_jsonl>
# Action: writes output_qa.jsonl to the provided output directory
# Format: {"id": "q1_utility_test_00000", "pred_idx": 6}
#
# Defaults:
#   mode: bias
#   output_dir: <repo>/predictions/Q1
#   input_jsonl:
#     - bias -> <repo>/data/public/q1_bbq_dev.jsonl
#     - qa   -> <repo>/data/public/q1_qa_dev.jsonl

mode="${1:-bias}"
out_dir="${2:-../predictions/Q1}"

if [ "$mode" = "bias" ]; then
  default_input="../data/public/q1_bbq_dev.jsonl"
elif [ "$mode" = "qa" ]; then
  default_input="../data/public/q1_qa_dev.jsonl"
else
  echo "Invalid mode: $mode (expected: bias or qa)"
  exit 1
fi

input_jsonl="${3:-$default_input}"

# bash download.sh

python3 infer.py "$mode" "$input_jsonl" "$out_dir" --model-dir "$out_dir"
