"""Download or verify M-base (Qwen/Qwen2.5-1.5B). Used in Task 2 (generator and detector).

Usage:
    python get_m_base.py                  # uses the default HF cache
    python get_m_base.py --local_dir ./models/m_base
"""
import argparse
from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen2.5-1.5B"


def ensure_model(local_dir=None, cache_dir=None):
    kwargs = dict(repo_id=REPO_ID, cache_dir=cache_dir, local_dir=local_dir)
    try:
        path = snapshot_download(local_files_only=True, **kwargs)
        print(f"[M-base] already present: {path}")
    except Exception:
        print(f"[M-base] not found locally, downloading {REPO_ID} ...")
        path = snapshot_download(**kwargs)
        print(f"[M-base] downloaded to: {path}")
    return path


def verify(path):
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(path)
    print("[M-base] tokenizer loads OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default=None)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()
    verify(ensure_model(args.local_dir, args.cache_dir))