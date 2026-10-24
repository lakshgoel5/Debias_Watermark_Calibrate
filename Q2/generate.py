import json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import MODEL, DELTA, NUM_TOKENS, load_key, green_mask

input_jsonl, out_dir, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
key = load_key(out_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()

#input rows
rows = [json.loads(l) for l in open(input_jsonl) if l.strip()]

results = []
for i, r in enumerate(rows):
    # Per-prompt RNG derived from (seed, index): same seed -> identical output,
    # different seed -> different output, regardless of run order.
    gen = torch.Generator(device=device)
    gen.manual_seed(seed * 1_000_003 + i) # 

    ids = tokenizer(r["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prev = ids[0, -1].item()
    new_ids = []
    with torch.no_grad():
        out = model(ids, use_cache=True)  # KV cache: only feed 1 token per step
        for _ in range(NUM_TOKENS):
            logits = out.logits[0, -1].float()
            assert logits.shape[0] == key["vocab_size"]
            # The ONLY change to the distribution: +DELTA on green tokens. Temperature stays 1.0,
            # no top-k/top-p. 
            # EOS is not blocked: if sampled we just keep going, so we never stop early.
            logits = logits + DELTA * green_mask(prev, key, key["vocab_size"]).to(device)
            # logits modified before softmax, so green tokens are more likely to be sampled
            nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen)  # samples an index according to the supplied weights or probabilities
            # multinomial sampling allows lower-probability tokens to be selected too
            prev = nxt.item()  # converts the one-element tensor into a regular Python integer
            new_ids.append(prev)
            out = model(nxt.view(1, 1), past_key_values=out.past_key_values, use_cache=True)

    results.append({"id": r["id"], "text": tokenizer.decode(new_ids, skip_special_tokens=True), "token_ids": new_ids})
    print(f"{i+1}/{len(rows)}", end="\r")

with open(os.path.join(out_dir, f"generated_{seed}.jsonl"), "w") as f:
    for x in results:
        f.write(json.dumps(x) + "\n")