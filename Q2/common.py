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
GAMMA = 0.5
# WM_DELTA / WM_CONTEXT / WM_GATE / WM_WHITEBOX only exist so dev sweeps need no file edits.
# Hardcode the winners here before submitting: the grader sets no environment variables.
DELTA = float(os.environ.get("WM_DELTA", 2.2))
CONTEXT = int(os.environ.get("WM_CONTEXT", 0))
ENTROPY_GATE = int(os.environ.get("WM_GATE", 1))  # Idea 3: scale delta by (1 - p_max)
WHITEBOX_DETECT = int(os.environ.get("WM_WHITEBOX", 1)) # Idea 4: M-base Neyman-Pearson LLR in detect.py
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
    """White-box Neyman-Pearson log-likelihood ratio under M-base blended with log-multiplicity z_score.

    At step t >= 1, M-base predicts next-token distribution p_base(. | ids[:t]).
    Let P_G(t) = sum_{v in Green} p_base(v | ids[:t]) be the probability mass M-base puts on the
    green list. Under a watermark with logit shift d on green tokens, the step log-likelihood
    ratio log(p_wm(ids[t] | ids[:t]) / p_base(ids[t] | ids[:t])) is:
        ell_t = d * 1[ids[t] in Green] - log(1 + P_G(t) * (exp(d) - 1))
    Forced green tokens in human text have P_G(t) -> 1, so ell_t -> 0 (suppressing false positives).
    Forced red tokens in watermarked text have P_G(t) -> 0, so ell_t -> 0 (avoiding false penalties).
    Blending with the context-free log-multiplicity z_score keeps robustness when random token
    substitutions corrupt local prefixes.
    """
    z_lex = z_score(ids, key)
    if len(ids) < 2:
        return z_lex
    vocab_size = key["vocab_size"]
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(inp).logits[0, :-1, :vocab_size].float()
    probs = torch.softmax(logits, dim=-1)

    d = LLR_DELTA
    ex = math.exp(d) - 1.0
    if not CONTEXT:
        mask = green_mask(0, key, vocab_size).to(device)
        p_green = probs[:, mask].sum(dim=-1).cpu().tolist()
        g_hits = [1.0 if mask[t].item() else 0.0 for t in ids[1:]]
    else:
        p_green, g_hits = [], []
        for step, (prev_t, cur_t) in enumerate(zip(ids[:-1], ids[1:])):
            m = green_mask(prev_t, key, vocab_size).to(device)
            p_green.append(float(probs[step, m].sum().item()))
            g_hits.append(1.0 if m[cur_t].item() else 0.0)

    # Include first token with unconditional prior P_G = GAMMA
    g0 = 1.0 if green_mask(0, key, vocab_size)[ids[0]] else 0.0
    llr = d * g0 - math.log(1.0 + GAMMA * ex)
    for g, pg in zip(g_hits, p_green):
        llr += d * g - math.log(1.0 + pg * ex)
    llr_norm = llr / math.sqrt(len(ids) * GAMMA * (1.0 - GAMMA))
    return 0.75 * llr_norm + 0.25 * z_lex