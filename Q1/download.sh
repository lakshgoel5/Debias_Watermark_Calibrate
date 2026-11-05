#!/bin/bash
set -euo pipefail

# Usage: ./download.sh [out_dir]
# Fetches lakshgoel/COL873_Q1_checkpoint into checkpoints/ by default, or <out_dir> if passed.

OUT="${1:-checkpoints}"
HF_REPO="lakshgoel/COL873_Q1_checkpoint"
mkdir -p "$OUT"

python3 - "$OUT" "$HF_REPO" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

out, repo = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=out)
print("download done")
PYEOF
