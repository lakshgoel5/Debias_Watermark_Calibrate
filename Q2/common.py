"""Shared watermark logic (used by generator, detector AND eval so they can never disagree).

Green-list watermark. One fixed green list (fraction GAMMA of the vocab), seeded by the secret.
Generator adds DELTA to those logits, then samples at temperature 1.
Detector is a z-score over unique tokens, weighted by 1 + ln(count).
"""
import collections, functools, hashlib, json, math, os, torch

MODEL = "Qwen/Qwen2.5-1.5B"
GAMMA = 0.25
DELTA = 2.0
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


def green_mask(key, vocab_size):
    """Bool mask over the vocab: True = green. Do not mutate the result; it is cached."""
    return _mask(key["secret"], 0, vocab_size)


def z_score(ids, key):
    """z = sum_i w_i (g_i - GAMMA) / sqrt(GAMMA (1 - GAMMA) sum_i w_i^2), w_i = 1 + ln(count_i)."""
    counts = collections.Counter(ids)
    if not counts:
        return 0.0
    mask = green_mask(key, key["vocab_size"])
    num = 0.0
    sum_w2 = 0.0
    for t, k in counts.items():
        w = 1.0 + math.log(k)
        num += w * ((1.0 if mask[t] else 0.0) - GAMMA)
        sum_w2 += w * w
    return num / math.sqrt(sum_w2 * GAMMA * (1.0 - GAMMA))