#!/bin/bash
set -euo pipefail

# Usage: ./download.sh <out_dir>
# Fetches exp3_model.joblib

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <out_dir>"
  exit 1
fi

OUT="$1"
HF_REPO="lakshgoel/col873-Q2-exp2"
mkdir -p "$OUT"

python3 - "$OUT" "$HF_REPO" <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download, snapshot_download

out, repo = sys.argv[1], sys.argv[2]
hf_hub_download(repo_id=repo, filename="exp3_model.joblib", local_dir=out)
print("download done")
PYEOF
