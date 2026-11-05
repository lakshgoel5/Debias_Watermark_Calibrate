# General constraints

- Python 3.10
- Only write in <output_dir>
- Give models, and pinned checkpoints from huggingface can be downloaded from internet

# Task1
## Baseline
- Getting scores from baseline model (Qwen2.5-1.5B-instruct) on the dataset provided in input file.

### With simple prompt

| Category | acc_amb | acc_disamb | bias_amb | bias_disamb | BBQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Age | 0.0800 | 0.7000 | 0.1739 | 0.3200 | 0.1500 |
| Gender_identity | 0.2000 | 0.9200 | 0.3000 | -0.1020 | 0.3890 |
| Nationality | 0.1800 | 0.8800 | 0.0732 | -0.0204 | 0.4898 |
| Physical_appearance | 0.0800 | 0.7800 | 0.4783 | 0.0638 | 0.1781 |
| Race_ethnicity | 0.2000 | 0.9600 | 0.2500 | -0.0400 | 0.4600 |
| SES | 0.1200 | 0.8600 | 0.0909 | 0.0000 | 0.4500 |

| Summary metric | Value |
| --- | ---: |
| score_BBQ | 0.3528 |
| Utility accuracy | 0.3000 |

### With prompt to give unknown answer if can't determine from context

| Category | acc_amb | acc_disamb | bias_amb | bias_disamb | BBQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Age | 0.5200 | 0.6200 | 0.1667 | 0.3478 | 0.3561 |
| Gender_identity | 0.5400 | 0.7000 | 0.2174 | 0.0000 | 0.5700 |
| Nationality | 0.5800 | 0.8000 | -0.1429 | -0.0222 | 0.6489 |
| Physical_appearance | 0.5200 | 0.6000 | 0.5833 | 0.3333 | 0.2533 |
| Race_ethnicity | 0.6400 | 0.7600 | 0.3333 | -0.0500 | 0.6150 |
| SES | 0.7400 | 0.7600 | -0.6923 | -0.0233 | 0.6484 |

| Summary metric | Value |
| --- | ---: |
| score_BBQ | 0.5153 |
| Utility accuracy | 0.2700 |

as we can see, acc_amb increased significantly with the new prompt, but acc_disamb decreased.

## LoRA SFT evaluation

| Category | acc_amb | acc_disamb | bias_amb | bias_disamb | BBQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Age | 1.0000 | 0.9800 | 0.0000 | 0.0000 | 0.9900 |
| Gender_identity | 1.0000 | 0.9400 | 0.0000 | -0.0800 | 0.9300 |
| Nationality | 0.7400 | 0.9800 | 0.2308 | -0.0800 | 0.7900 |
| Physical_appearance | 0.8800 | 0.9600 | 1.0000 | 0.0400 | 0.8400 |
| Race_ethnicity | 1.0000 | 0.8800 | 0.0000 | -0.0213 | 0.9294 |
| SES | 1.0000 | 1.0000 | 0.0000 | -0.0400 | 0.9800 |

| Summary metric | Value |
| --- | ---: |
| score_BBQ | 0.9099 |
| Utility accuracy | 0.3100 |

# Task 2

## Watermarking observations

At `gamma = 0.25`, the watermarked generations are producing a higher
z-score than the human-written development text. This is the expected
direction: the watermark encourages generated tokens to come from a
pseudo-random green list more often than would occur by chance.

The detector z-score measures how far the observed green-list signal is from
the expectation under the null hypothesis of unwatermarked text. For unweighted
unique items (`w_i = 1`), this is:

```text
z = (green_count - gamma * token_count)
    / sqrt(token_count * gamma * (1 - gamma))
```

When each unique item `i` (a token for `CONTEXT = 0`, or a `(prev_token, token)`
pair for `CONTEXT = 1`) with multiplicity `k_i` in the text is given sublinear
log-multiplicity weight `w_i = 1 + ln(k_i)`, the exact null-normalized z-score is:

```text
z = sum_i w_i * (1[item_i in Green] - gamma)
    / sqrt(gamma * (1 - gamma) * sum_i w_i^2)
```

Why `w_i = 1 + ln(k_i)`: pure deduplication (`w_i = 1`) throws away topic nouns
and named entities that repeat 2-4 times and survive paraphrasing, whereas linear
counts (`w_i = k_i`) let a single frequent human function word blow up the false
positive tail. Sublinear log weighting captures repeated topic words while
keeping the human 99th-percentile threshold stable.

A z-score near zero is consistent with the expected green-token rate. A large
positive z-score is evidence that the text contains more green-list tokens than
expected and therefore looks more likely to be watermarked. A negative
z-score indicates fewer green-list tokens than expected. The comparison with
human-written dev text should be interpreted carefully: a high human-text
z-score would indicate possible false positives or a distribution/tokenization
effect, not necessarily successful watermark detection.

The two random seeds produce different generated outputs. This is expected
because the generation RNG is seeded separately, and it is important to report
results across seeds rather than relying on a single generated sample.

## Q2 run log

Dev set (100 prompts from `q2_prompts_dev.jsonl`), seed 42, scored with `Q2/eval.py`.
Detector score is the z-score over unique tokens or token pairs. Negatives are the dev
`human_text` rows (`human`, and `human_short` cut to 50 tokens). With n = 100, tau is the 99th
smallest negative score, so it is noisier than the grader's n = 500.

### Run 1: baseline

| Setting | Value |
| --- | --- |
| `DELTA` | 2.0 |
| `CONTEXT` | 1 (green list depends on the previous token, `h=1`) |
| `GAMMA` | 0.25 |
| Paraphrase | M-inst, my own prompt (grader prompt unknown) |

Settings are the values in `Q2/common.py` when this run was added to the scratchpad.

| Condition | TPR |
| --- | ---: |
| clean | 1.0000 |
| short | 0.9600 |
| substitution | 1.0000 |
| paraphrase | 0.1700 |
| **S_det** | **0.7825** |

| Quality | Value |
| --- | ---: |
| median PPL (yours) | 34.520 |
| median PPL (unwatermarked) | 24.421 |
| **Q** | **1.4136** |

Detector score summary from the score files (human tau is the 99th smallest of 100):

| File | Mean z | Min | Max |
| --- | ---: | ---: | ---: |
| clean | 10.77 | 7.15 | 14.00 |
| short | 5.00 | 2.09 | 7.84 |
| substitution | 6.72 | 3.42 | 9.30 |
| paraphrase | 1.23 | -1.73 | 4.56 |
| human | 0.06 | -2.90 | 2.79 |
| human_short | -0.07 | -2.39 | 3.33 |

human tau = 2.47, human_short tau = 2.68.

Observation: paraphrase is the entire `S_det` loss (the other three conditions sum to 2.96 of 3).
Its mean z (1.23) is below tau. With `CONTEXT = 1`, a paraphrase keeps individual words but
changes their neighbours, so most scored pairs are new. Paraphrase output averaged 176 tokens
(min 49, max 310), not 200.

`common.py` reads `WM_CONTEXT` and `WM_DELTA` from the environment for sweeps. Hardcode the
winners before submitting.

### Run 2: context-free green list

Files in `Q2/secret/secret`. New `key.json` (setup was re-run; secret differs from run 1).
`CONTEXT = 0` was checked by rescoring the clean texts: the stored detector scores match
`CONTEXT = 0` exactly and do not match `CONTEXT = 1`. `DELTA` is not stored in the outputs;
`common.py` still defaults it to 2.0, so this entry assumes the run did not set `WM_DELTA`.

| Setting | Value |
| --- | --- |
| `DELTA` | 2.0 (assumed; not recorded in the run files) |
| `CONTEXT` | 0 (one green list for the whole text, `h=0`) |
| `GAMMA` | 0.25 |
| Paraphrase | M-inst, same prompt as run 1 |

| Condition | TPR | Run 1 |
| --- | ---: | ---: |
| clean | 1.0000 | 1.0000 |
| short | 0.9500 | 0.9600 |
| substitution | 1.0000 | 1.0000 |
| paraphrase | 0.5900 | 0.1700 |
| **S_det** | **0.8850** | 0.7825 |

| Quality | Value | Run 1 |
| --- | ---: | ---: |
| median PPL (yours) | 28.466 | 34.520 |
| median PPL (unwatermarked) | 24.421 | 24.421 |
| **Q** | **1.1656** | 1.4136 |

The unwatermarked median is the same number as run 1, so the Q drop is entirely from a lower
watermarked perplexity.

Detector score summary (n = 100; tau is the 99th smallest negative):

| File | Mean z | Min | Max |
| --- | ---: | ---: | ---: |
| clean | 8.45 | 4.49 | 13.15 |
| short | 4.91 | 1.85 | 7.49 |
| substitution | 6.53 | 3.39 | 10.37 |
| paraphrase | 2.73 | -0.39 | 6.76 |
| human | 0.59 | -1.45 | 2.63 |
| human_short | 0.45 | -1.53 | 2.80 |

human tau = 2.57, human_short tau = 2.55.

Paraphrase mean z moved from 1.23 (below tau) to 2.73 (just above tau), and TPR from 0.17 to 0.59.
Clean, short, and substitution stayed saturated. Paraphrase outputs averaged 166 tokens
(min 15, max 244). `S_det` 0.7825 to 0.8850 and `Q` 1.41 to 1.17, both better.

### Run 2b: log-multiplicity weighted z-score on Run 2 generations (detector-only update)

Same generated texts and `Q = 1.1656` as Run 2 (`Q2/secret/secret`), rescored with
`w_i = 1 + ln(k_i)` in `z_score`:

| Condition | TPR (`w_i = 1 + ln k_i`) | Run 2 (`w_i = 1`) | Run 1 (`CONTEXT = 1`) |
| --- | ---: | ---: | ---: |
| clean | 1.0000 | 1.0000 | 1.0000 |
| short | 0.9800 | 0.9500 | 0.9600 |
| substitution | 1.0000 | 1.0000 | 1.0000 |
| paraphrase | 0.7300 | 0.5900 | 0.1700 |
| **S_det** | **0.9275** | 0.8850 | 0.7825 |

Next: paraphrase is at 0.73 (`S_det = 0.9275`, `Q = 1.1656`).

### Run 2c: White-box Neyman–Pearson LLR detector using `M-base` (Idea 4, detector-only on Run 2 files)

Rule 4 allows `detect.sh` to run `M-base`. At each token step `t >= 1`, `M-base` predicts the
next-token distribution `p_base(. | x_{<t})`, and we compute the total probability mass on the
green list `P_G(t) = sum_{v in Green} p_base(v | x_{<t})`. Under a watermark with logit shift `d`,
the exact step log-likelihood ratio `log(p_wm(x_t | x_{<t}) / p_base(x_t | x_{<t}))` is:

```text
ell_t = d * 1[x_t in Green] - log(1 + P_G(t) * (exp(d) - 1))
LLR_norm = sum_t ell_t / sqrt(T * gamma * (1 - gamma))
score = 0.75 * LLR_norm + 0.25 * z_log_mult
```

- Predictable green tokens in human text have `P_G(t) -> 1`, so `ell_t -> d - d = 0` (zeros out false positives).
- Predictable red tokens in watermarked text have `P_G(t) -> 0`, so `ell_t -> 0` (avoids false penalties).

Scored on the same Run 2 files (`Q2/secret/secret`, `Q = 1.1656` unchanged):

| Condition | TPR (White-box LLR `d=1.0`) | Run 2b (`1 + ln k`) | Run 2 (`w=1`) | Run 1 (`CONTEXT=1`) |
| --- | ---: | ---: | ---: | ---: |
| clean | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| short | 0.9900 | 0.9800 | 0.9500 | 0.9600 |
| substitution | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| paraphrase | 0.8100 | 0.7300 | 0.5900 | 0.1700 |
| **S_det** | **0.9500** | 0.9275 | 0.8850 | 0.7825 |

### Run 3: half the vocabulary green

Reported from Kaggle. `GAMMA = 0.5`, `DELTA = 2.2`. `CONTEXT` was not in the report; the sweep this
followed used `CONTEXT = 0`. Flat `DELTA` (not the entropy gate). Detector is the one that produced
these TPRs, not the white-box LLR (that code had not been run yet).

| Setting | Value |
| --- | --- |
| `DELTA` | 2.2 |
| `GAMMA` | 0.5 |
| `CONTEXT` | 0 (not stated; previous run and this sweep used `h=0`) |

| Condition | TPR | Run 2 (`GAMMA=0.25`, `DELTA=2.0`) |
| --- | ---: | ---: |
| clean | 1.0000 | 1.0000 |
| short | 0.9900 | 0.9500 |
| substitution | 1.0000 | 1.0000 |
| paraphrase | 0.4200 | 0.5900 |
| **S_det** | **0.8525** | 0.8850 |

| Quality | Value | Run 2 |
| --- | ---: | ---: |
| median PPL (yours) | 26.473 | 28.466 |
| median PPL (unwatermarked) | 27.975 | 24.421 |
| **Q** | **0.9463** | 1.1656 |

`Q` below 1 means the watermarked median PPL is lower than this unwatermarked draw. The
unwatermarked median moved from 24.421 to 27.975, so the denominator is a new sample and `Q` is
not on the same baseline as runs 1 and 2. Own-text median PPL did fall, 28.466 to 26.473.
Paraphrase TPR fell, 0.59 to 0.42, so `S_det` is worse than run 2 even though short improved to 0.99.

### Run 4: entropy-gated `DELTA` plus white-box detector

The Idea 3 + Idea 4 code. `generate` projected 40.1 min for 300 prompts, over the 30 min limit.
`detect` was inside the 15 min limit.

| Setting | Value |
| --- | --- |
| `DELTA` | 2.5, scaled each step by `(1 - p_max)` |
| `CONTEXT` | 0 |
| `GAMMA` | 0.25 |
| Detector | white-box LLR blended with log-multiplicity z-score |

| Condition | TPR | Run 2c (same detector idea, Run 2 texts) | Run 2 |
| --- | ---: | ---: | ---: |
| clean | 1.0000 | 1.0000 | 1.0000 |
| short | 0.9300 | 0.9900 | 0.9500 |
| substitution | 1.0000 | 1.0000 | 1.0000 |
| paraphrase | 0.2700 | 0.8100 | 0.5900 |
| **S_det** | **0.8000** | 0.9500 | 0.8850 |

| Quality | Value | Run 3 | Run 2 |
| --- | ---: | ---: | ---: |
| median PPL (yours) | 25.759 | 26.473 | 28.466 |
| median PPL (unwatermarked) | 27.975 | 27.975 | 24.421 |
| **Q** | **0.9208** | 0.9463 | 1.1656 |

Gating shrinks the boost on confident tokens, so the watermark is weaker: paraphrase TPR fell to 0.27.
The extra softmax at every token also pushed generation over the time limit.

### Run 5: flat `DELTA = 2.0` regenerated, blended white-box detector

New texts, not a rescore of Run 2. `generate` projected 41.2 min for 300 prompts.

| Setting | Value |
| --- | --- |
| `DELTA` | 2.0, flat |
| `CONTEXT` | 0 |
| `GAMMA` | 0.25 |
| Detector | `0.75 * LLR + 0.25 * z` (the blend, not the pure LLR from Run 2c) |

| Condition | TPR |
| --- | ---: |
| clean | 1.0000 |
| short | 0.9800 |
| substitution | 1.0000 |
| paraphrase | 0.4900 |
| **S_det** | **0.8675** |

`Q = 1.1`. The 0.9500 from Run 2c was pure LLR (`d = 1.0`, no z-score mix) on one old file of 100 prompts. This run used a new sample and a different score, and paraphrase is the gap.

### Code after Run 5

- Generator is unchanged in distribution: `GAMMA = 0.25`, `CONTEXT = 0`, flat `DELTA = 2.0`. Prompts are decoded in batches of 8 (`WM_BATCH`) so 300 prompts can finish inside 30 min. Same seed and same batch size reproduce the file. A new `generated_<seed>.jsonl` will not match Run 5 token for token.
- Detector score is the pure LLR from Run 2c: `sum ell_t / sqrt(T)`, `LLR_DELTA = 1.0`, no z-score mix. Rescore the Run 5 files with the six `detect.sh` calls before regenerating.

# Task 3

## Q3 implementation notes

### Exp1: plain M-inst (simple baseline)

Greedy decode of M-inst. Confidence is the arithmetic mean of generated-token probabilities; `confidence_geo` is the geometric mean and `confidence_min` is the minimum token probability. Abstain when the arithmetic mean is below 0.5. Official AUROC, ECE, selective, and `set_score` all use the arithmetic mean. `Q3 score` is the mean of in-domain and out-of-domain `set_score`.

| Split | accuracy | AUROC | ECE | selective | AUROC_geo | AUROC_min | best_ranker | set_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| train | 0.1650 | 0.7458 | 0.5749 | 0.0000 | 0.7629 | 0.7359 | geo | 0.3903 |
| in-domain | 0.1640 | 0.7852 | 0.5857 | 0.0000 | 0.8029 | 0.7838 | geo | 0.3998 |
| out-of-domain | 0.0950 | 0.7194 | 0.5850 | 0.0000 | 0.7639 | 0.8851 | min | 0.3781 |

**Q3 score = 0.3890**

`set_score = (AUROC + (1 - ECE) + selective) / 3`. Selective is 0 on every split, so it contributes nothing. Accuracy is about 0.16 in domain and 0.095 out of domain, so the error rate among answered items is far above the 0.3 point where selective goes to 0. ECE is about 0.58, so `1 - ECE` is only about 0.41. AUROC is the only strong term.

Geo ranks better than the arithmetic mean on train (0.7629 vs 0.7458) and in-domain (0.8029 vs 0.7852). On out-of-domain, min is much better (0.8851 vs 0.7194 arithmetic and 0.7639 geo). The submitted confidence is still the arithmetic mean, so that gain is not in the score.

### Exp2: P(True) combined with white-box features

Same greedy answers as Exp1. Confidence is a scaled logistic regression (`C = 0.5`) on white-box token features plus `ptrue_logit`. Saved model is white-box + P(True), `checkpoints/exp2_model.joblib`.

Ablations, no abstain:

| Model | dev AUROC | dev ECE | ood AUROC | ood ECE |
| --- | ---: | ---: | ---: | ---: |
| ptrue only | 0.6833 | 0.0168 | 0.6592 | 0.0303 |
| white-box only | 0.8329 | 0.0342 | 0.8095 | 0.0452 |
| white-box + ptrue | 0.8601 | 0.0399 | 0.8369 | 0.0292 |

Train out-of-fold: AUROC 0.8035, ECE 0.0202. Best single features: `mean_logprob` (dev 0.812) and `min_logprob` (ood 0.889). Entropy and length rank below 0.5.

Abstain threshold from train out-of-fold, max risk 0.15, min answered 20. No safe slice, so fallback is the top 20 rows, `tau = 0.6610`.

| Split | accuracy | AUROC | ECE | selective | coverage | risk | set_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 0.1640 | 0.8601 | 0.0399 | 0.0100 | 0.0100 | 0.2000 | 0.6101 |
| ood | 0.0950 | 0.8369 | 0.0292 | 0.0000 | 0.0200 | 0.5000 | 0.6026 |

**Q3 score = 0.6064**

ECE is about 0.03, so the gain over Exp1 (0.3890) is calibration. Selective is still about 0: dev answers 1% of rows, ood risk is 0.50.