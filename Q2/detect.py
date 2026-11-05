"""Uses only the TEXT of each line: re-tokenize, then score with M-base white-box LLR + log-multiplicity z-score."""
import json, os, re, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import MODEL, WHITEBOX_DETECT, load_key, z_score, whitebox_score

# Init
input_jsonl, out_dir = sys.argv[1], sys.argv[2]
key = load_key(out_dir)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if WHITEBOX_DETECT:
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device).eval()
else:
    model, device = None, "cpu"

# Read input and compute scores
rows = [json.loads(l) for l in open(input_jsonl) if l.strip()]
scores = []
for r in rows:
    # Re-tokenize from text (not stored token_ids): perturbed/paraphrased text has no valid ids.
    text = r["text"] if "text" in r else r["human_text"]   # KeyError if neither exists
    ids = tokenizer(text, add_special_tokens=False).input_ids
    scores.append(whitebox_score(ids, key, model, device) if WHITEBOX_DETECT else z_score(ids, key))

# File naming
stem = os.path.basename(input_jsonl)[:-len(".jsonl")].removeprefix("generated_")
m = re.fullmatch(r"(.*?)_?(\d+)", stem)
cond = (m.group(1) or "clean") if m else stem
with open(os.path.join(out_dir, f"scores_{cond}.jsonl"), "w") as f:
    for r, s in zip(rows, scores):
        f.write(json.dumps({"id": r["id"], "score": s}) + "\n")