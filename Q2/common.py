"""Shared watermark logic (used by generator, detector AND eval so they can never disagree).

Scheme: green-list watermark (Kirchenbauer-style), written from scratch.
  - Green list = pseudo-random subset (fraction GAMMA) of the vocab, seeded by the secret key
    and (optionally) the previous token.
  - Generator adds DELTA to green logits, then samples normally.
  - Detector counts green tokens -> z-score.

Knobs:
  DELTA logit bonus. Bigger = easier detection, but worse perplexity (hurts Q).
  CONTEXT green list depends on previous token (h=1); 0 = one fixed list for the whole text, h=0 survives substitution/paraphrase much better
"""
import functools, hashlib, json, math, os, torch

MODEL = "Qwen/Qwen2.5-1.5B"
GAMMA = 0.25
DELTA = 2.0
CONTEXT = 1
NUM_TOKENS = 200


def load_key(output_dir):
    with open(os.path.join(output_dir, "key.json")) as f:
        return json.load(f)

# This is a Python decorator that caches the function's return values, so repeated calls with the same arguments can reuse a previously computed result instead of executing the function again.
@functools.lru_cache(maxsize=4)   # with CONTEXT=0 the same mask is reused for every token
def _mask(secret, prev, vocab_size):
    # sha256 -> seed of a CPU generator: identical on every machine, for generator and detector.
    h = hashlib.sha256(f"{secret}-{prev}".encode()).digest()
    g = torch.Generator()
    g.manual_seed(int.from_bytes(h[:8], "big") & (2**63 - 1))
    return torch.rand(vocab_size, generator=g) < GAMMA


def green_mask(prev_token, key, vocab_size):
    """Bool mask over the logit dimension: True = green. (Do not mutate the result; it is cached.)"""
    return _mask(key["secret"], prev_token if CONTEXT else 0, vocab_size)

# z = G - n*GAMMA / 
#                 sqrt(n*GAMMA*(1-GAMMA))
def z_score(ids, key):
    """Detection statistic for a list of token ids.

    Null hypothesis (human text): each scored item is green with prob GAMMA, independently.
    Items are UNIQUE (context, token) pairs: repeated phrases would otherwise be counted many
    times, breaking independence and inflating human scores.
    With h=1 the first token is skipped (its context was in the prompt, which we don't have).
    """
    if CONTEXT: #Context=1
        items = set(zip(ids[:-1], ids[1:]))
    else: #Context=0
        items = {(0, t) for t in ids}
    n = len(items)
    if n == 0:
        return 0.0
    green = sum(bool(green_mask(p, key, key["vocab_size"])[t]) for p, t in items)
    return (green - GAMMA * n) / math.sqrt(n * GAMMA * (1 - GAMMA))