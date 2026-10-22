"""Download or verify M-inst (Qwen/Qwen2.5-1.5B-Instruct). Used in Tasks 1 and 3.

Usage:
    python get_m_inst.py                  # uses the default HF cache
    python get_m_inst.py --local_dir ./models/m_inst
"""
import argparse
from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def ensure_model(local_dir=None, cache_dir=None):
    kwargs = dict(repo_id=REPO_ID, cache_dir=cache_dir, local_dir=local_dir)
    try:
        path = snapshot_download(local_files_only=True, **kwargs)
        print(f"[M-inst] already present: {path}")
    except Exception:
        print(f"[M-inst] not found locally, downloading {REPO_ID} ...")
        path = snapshot_download(**kwargs)
        print(f"[M-inst] downloaded to: {path}")
    return path


def verify(path):
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(path)
    print("[M-inst] tokenizer loads OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default=None)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()
    verify(ensure_model(args.local_dir, args.cache_dir))