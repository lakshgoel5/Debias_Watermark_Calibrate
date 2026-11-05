"""Builds the 4 test conditions from generated_<seed>.jsonl.

Usage: python perturb.py <input_jsonl> <output_dir> <seed>
Writes <output_dir>/generated_{clean,short,substitution,paraphrase}_<seed>.jsonl
Each line: {"id", "text", "token_ids"}  (detect.py only reads "id" and "text").
"""
import json, os, random, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import MODEL

SUB_FRAC = 0.2          # 40 of 200 positions
SHORT_LEN = 50
PARA_BATCH = 16
PARA_MAX_NEW = 320

input_jsonl, out_dir, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out_dir, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
rows = [json.loads(l) for l in open(input_jsonl) if l.strip()]


def write(cond, items):
    with open(os.path.join(out_dir, f"generated_{cond}_{seed}.jsonl"), "w") as f:
        for x in items:
            f.write(json.dumps(x) + "\n")


def dec(ids):
    return tok.decode(ids, skip_special_tokens=True)


# clean
write("clean", [{"id": r["id"], "text": r["text"], "token_ids": r["token_ids"]} for r in rows])

# short: first 50 tokens
short = []
for r in rows:
    ids = r["token_ids"][:SHORT_LEN]
    short.append({"id": r["id"], "text": dec(ids), "token_ids": ids})
write("short", short)

# substitution: 20% of positions -> uniform non-special token id
special = set(tok.all_special_ids)
special |= {i for i, t in tok.added_tokens_decoder.items() if t.special}
vocab = [i for i in range(len(tok)) if i not in special]
sub = []
for idx, r in enumerate(rows):
    rng = random.Random(seed * 1_000_003 + idx)
    ids = list(r["token_ids"])
    k = round(SUB_FRAC * len(ids))
    for p in rng.sample(range(len(ids)), k):
        ids[p] = rng.choice(vocab)
    sub.append({"id": r["id"], "text": dec(ids), "token_ids": ids})
write("substitution", sub)

# paraphrase: M-base only. M-inst is for Tasks 1 and 3, so it is not loaded here.
device = "cuda" if torch.cuda.is_available() else "cpu"
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
torch.manual_seed(seed)


def prompt_for(text):
    return ("Rewrite the passage in different words. Keep the meaning and roughly the same length. "
            "The passage may end mid-sentence; do not finish it or add a title.\n\nPassage:\n"
            + text.strip() + "\n\nRewrite:\n")


para = []
for b in range(0, len(rows), PARA_BATCH):
    batch = rows[b:b + PARA_BATCH]
    enc = tok([prompt_for(r["text"]) for r in batch], return_tensors="pt", padding=True,
              add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=PARA_MAX_NEW, do_sample=True, temperature=0.7,
                             top_p=0.9, pad_token_id=tok.pad_token_id)
    for r, o in zip(batch, out):
        text = tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        para.append({"id": r["id"], "text": text, "token_ids": tok(text, add_special_tokens=False).input_ids})
    print(f"paraphrase {len(para)}/{len(rows)}", end="\r")
write("paraphrase", para)
