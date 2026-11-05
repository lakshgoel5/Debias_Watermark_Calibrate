"""Uses only the TEXT of each line: re-tokenize, then score with M-base white-box LLR + log-multiplicity z-score."""
import json, os, re, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import MODEL, WHITEBOX_DETECT, load_key, z_score, whitebox_score

# Init
input_jsonl, out_dir = sys.argv[1], sys.argv[2]
key = load_key(out_dir)
t_load = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if WHITEBOX_DETECT:
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()
else:
    model, device = None, "cpu"
if device == "cuda":
    torch.cuda.synchronize()
elif device == "mps":
    torch.mps.synchronize()
load_s = time.perf_counter() - t_load

def _sync():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

# Read input and compute scores
rows = [json.loads(l) for l in open(input_jsonl) if l.strip()]
scores = []
t_score = time.perf_counter()
for i, r in enumerate(rows):
    # Re-tokenize from text (not stored token_ids): perturbed/paraphrased text has no valid ids.
    text = r["text"] if "text" in r else r["human_text"]   # KeyError if neither exists
    ids = tokenizer(text, add_special_tokens=False).input_ids
    scores.append(whitebox_score(ids, key, model, device) if WHITEBOX_DETECT else z_score(ids, key))
    if (i + 1) % 20 == 0 or i + 1 == len(rows):
        print(f"{i+1}/{len(rows)}  {time.perf_counter() - t_score:.0f}s", flush=True)

_sync()
score_s = time.perf_counter() - t_score
per = score_s / len(rows) if rows else 0.0
# Grader limit: one file of 500 texts in 15 minutes. Model load happens once.
projected = load_s + per * 500
print(f"detect: load {load_s:.1f}s, {len(rows)} texts {score_s:.1f}s, {per:.2f}s/text")
print(f"detect: projected 500 texts {projected / 60:.1f} min (limit 15 min)")

# File naming
stem = os.path.basename(input_jsonl)[:-len(".jsonl")].removeprefix("generated_")
m = re.fullmatch(r"(.*?)_?(\d+)", stem)
cond = (m.group(1) or "clean") if m else stem
with open(os.path.join(out_dir, f"scores_{cond}.jsonl"), "w") as f:
    for r, s in zip(rows, scores):
        f.write(json.dumps({"id": r["id"], "score": s}) + "\n")