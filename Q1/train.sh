#!/bin/bash

## Usage: ./train.sh <abs path to output dir>
### auto input from ../data/public/

python train.py --output-dir "$1" --train-jsonl ../data/public/q1_bbq_train.jsonl --dev-jsonl ../data/public/q1_bbq_dev.jsonl