"""Re-tokenize each text and write one z-score per line. No model."""
import json, os, re, sys
from transformers import AutoTokenizer
from common import MODEL, load_key, z_score

input_jsonl, out_dir = sys.argv[1], sys.argv[2]
key = load_key(out_dir)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

rows = [json.loads(l) for l in open(input_jsonl) if l.strip()]
scores = []
for r in rows:
    text = r["text"] if "text" in r else r["human_text"]
    ids = tokenizer(text, add_special_tokens=False).input_ids
    scores.append(z_score(ids, key))

stem = os.path.basename(input_jsonl)[:-len(".jsonl")].removeprefix("generated_")
m = re.fullmatch(r"(.*?)_?(\d+)", stem)
cond = (m.group(1) or "clean") if m else stem
with open(os.path.join(out_dir, f"scores_{cond}.jsonl"), "w") as f:
    for r, s in zip(rows, scores):
        f.write(json.dumps({"id": r["id"], "score": s}) + "\n")
