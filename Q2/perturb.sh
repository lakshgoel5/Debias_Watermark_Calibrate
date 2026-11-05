#!/bin/bash
# Usage: ./perturb.sh <generated_<seed>.jsonl> <output_dir> <seed>
python perturb.py "$1" "$2" "$3"
