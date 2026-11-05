import inspect, json, os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import DELTA, MODEL, NUM_TOKENS, load_key, green_mask

input_jsonl, out_dir, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
batch_size = int(os.environ.get("WM_BATCH", "8"))
key = load_key(out_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"

t_load = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
fwd_extra = {}
params = inspect.signature(model.forward).parameters
if "logits_to_keep" in params:
    fwd_extra["logits_to_keep"] = 1
elif "num_logits_to_keep" in params:
    fwd_extra["num_logits_to_keep"] = 1
if device == "cuda":
    torch.cuda.synchronize()
load_s = time.perf_counter() - t_load

rows = [json.loads(l) for l in open(input_jsonl) if l.strip()]
vocab = key["vocab_size"]
green = green_mask(key, vocab).to(device)
gen = torch.Generator(device=device)
gen.manual_seed(seed)

results = [None] * len(rows)
t_gen = time.perf_counter()
for start in range(0, len(rows), batch_size):
    chunk = rows[start:start + batch_size]
    enc = tokenizer([r["prompt"] for r in chunk], return_tensors="pt", padding=True,
                    add_special_tokens=False).to(device)
    attn = enc["attention_mask"]
    # Left padding must not shift RoPE. Real prompt tokens keep positions 0..L-1.
    prompt_len = attn.long().sum(dim=1)
    position_ids = attn.long().cumsum(dim=1) - 1
    position_ids = position_ids.masked_fill(attn == 0, 0)
    new = []
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=attn, position_ids=position_ids,
                    use_cache=True, **fwd_extra)
        for step in range(NUM_TOKENS):
            logits = out.logits[:, -1, :vocab].float()
            logits = logits + DELTA * green.float()
            nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen)
            new.append(nxt)
            attn = torch.cat([attn, torch.ones(len(chunk), 1, dtype=attn.dtype, device=device)], 1)
            step_pos = (prompt_len + step).unsqueeze(1)
            out = model(input_ids=nxt, attention_mask=attn, position_ids=step_pos,
                        past_key_values=out.past_key_values, use_cache=True, **fwd_extra)
    ids = torch.cat(new, 1).tolist()
    for j, r in enumerate(chunk):
        row = {
            "id": r["id"],
            "text": tokenizer.decode(ids[j], skip_special_tokens=True),
            "token_ids": ids[j],
        }
        if r.get("human_text"):
            row["human_text"] = r["human_text"]
        results[start + j] = row
    print(f"{min(start + batch_size, len(rows))}/{len(rows)}  {time.perf_counter() - t_gen:.0f}s", flush=True)

if device == "cuda":
    torch.cuda.synchronize()
gen_s = time.perf_counter() - t_gen
per = gen_s / len(rows) if rows else 0.0
projected = load_s + per * 300
print(f"generate: load {load_s:.1f}s, {len(rows)} prompts {gen_s:.1f}s, {per:.2f}s/prompt, batch {batch_size}")
print(f"generate: projected 300 prompts {projected / 60:.1f} min (limit 30 min)")

with open(os.path.join(out_dir, f"generated_{seed}.jsonl"), "w") as f:
    for x in results:
        f.write(json.dumps(x) + "\n")
