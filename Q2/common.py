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
import collections, functools, hashlib, json, math, os, torch

MODEL = "Qwen/Qwen2.5-1.5B"
GAMMA = 0.25
DELTA = float(os.environ.get("WM_DELTA", 2.0))
CONTEXT = int(os.environ.get("WM_CONTEXT", 0))
ENTROPY_GATE = int(os.environ.get("WM_GATE", 0))  # 0: flat DELTA. 1 was slower and weaker.
WHITEBOX_DETECT = int(os.environ.get("WM_WHITEBOX", 1))  # M-base Neyman-Pearson LLR in detect.py
LLR_DELTA = float(os.environ.get("WM_LLR_DELTA", 1.0))
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

# Weighted z-score over unique (context, token) items with log-multiplicity weights w_i = 1 + ln(k_i):
#   z = sum_i w_i * (1[item_i in Green] - GAMMA) / sqrt(GAMMA * (1 - GAMMA) * sum_i w_i^2)
def z_score(ids, key):
    """Detection statistic for a list of token ids.

    Null hypothesis (human text): each distinct (context, token) item is green with prob GAMMA
    independently. Rather than discarding repeat occurrences (w_i = 1) or counting raw repeats
    linearly (w_i = k_i, which lets a single repeated word blow up human variance), each unique
    item with count k_i gets sublinear weight w_i = 1 + ln(k_i), normalized by the exact null
    standard deviation sqrt(GAMMA * (1 - GAMMA) * sum_i w_i^2).
    With h=1 the first token is skipped (its context was in the prompt, which we don't have).
    """
    if CONTEXT:  # Context=1
        counts = collections.Counter(zip(ids[:-1], ids[1:]))
    else:        # Context=0
        counts = collections.Counter((0, t) for t in ids)
    if not counts:
        return 0.0
    vocab_size = key["vocab_size"]
    num = 0.0
    sum_w2 = 0.0
    for (p, t), k in counts.items():
        w = 1.0 + math.log(k)
        is_green = 1.0 if green_mask(p, key, vocab_size)[t] else 0.0
        num += w * (is_green - GAMMA)
        sum_w2 += w * w
    return num / math.sqrt(sum_w2 * GAMMA * (1.0 - GAMMA))


@torch.no_grad()
def whitebox_score(ids, key, model, device):
    """Neyman-Pearson log-likelihood ratio of the watermark against M-base.

    ell_t = d * 1[token_t in Green] - log(1 + P_G(t) * (exp(d) - 1))
    score = sum_t ell_t / sqrt(T)

    P_G(t) is the probability mass M-base puts on the green list at that step.
    A forced green token in human text has P_G near 1, so ell_t is about 0.
    This is the score that reached S_det 0.95 on the Run 2 files. It is not mixed
    with the z-score: that mix lowered paraphrase detection on the later run.
    """
    if len(ids) < 2:
        return z_score(ids, key)
    vocab_size = key["vocab_size"]
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(inp).logits[0, :-1, :vocab_size].float()
    probs = torch.softmax(logits, dim=-1)
    d = LLR_DELTA
    ex = math.exp(d) - 1.0
    chosen = torch.tensor(ids[1:], dtype=torch.long, device=device)
    if not CONTEXT:
        mask = green_mask(0, key, vocab_size).to(device)
        p_green = probs[:, mask].sum(dim=-1)
        green = mask[chosen].float()
    else:
        p_green, green = [], []
        for step, (prev_t, cur_t) in enumerate(zip(ids[:-1], ids[1:])):
            m = green_mask(prev_t, key, vocab_size).to(device)
            p_green.append(probs[step, m].sum())
            green.append(m[cur_t].float())
        p_green = torch.stack(p_green)
        green = torch.stack(green)
    llr = (d * green - torch.log1p(p_green * ex)).sum()
    return float(llr / math.sqrt(len(ids) - 1))