#!/bin/bash
set -euo pipefail

# Usage: ./infer.sh <input_jsonl> <output_dir>
# Action: writes <output_dir>/pred_<input file name>

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_jsonl> <output_dir>"
  exit 1
fi

input_jsonl="$1"
output_dir="$2"
input_name="$(basename "$input_jsonl")"

mkdir -p "$output_dir"
python3 infer.py "$input_jsonl" "$output_dir"
cp "$output_dir/output.jsonl" "$output_dir/pred_$input_name"
