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

### LoRA SFT evaluation

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

# Task 3

## Q3 implementation notes

The Q3 inference and evaluation scripts are intentionally straightforward, but
the following points should be considered before treating the results as final:

- The current confidence is the mean probability of the generated tokens. This
  is a useful baseline, but it is only a proxy for answer correctness, is
  sensitive to answer length, and is not necessarily calibrated.
- The abstention threshold should not be chosen arbitrarily. It should be tuned
  on the in-domain dev set and then kept fixed when evaluating both the
  in-domain and out-of-domain sets.
- Greedy decoding is deterministic and makes experiments reproducible. If
  sampling is used later, the seed and sampling parameters should be recorded.
- The evaluator follows the task specification: lowercase normalization,
  punctuation/article removal, collapsed spaces, and contiguous gold-answer
  matching. It reports AUROC, 10-bin ECE, selective score, per-set score, and
  the average of the in-domain and out-of-domain scores.
- A possible improvement is to replace the arithmetic mean token probability
  with a length-normalized log-probability (geometric mean), then compare both
  confidence estimates on the dev set.
- Prediction files should be validated strictly, especially the types and
  ranges of `confidence` and `abstain`, and should include the exact inference
  configuration used to produce them.

# Task 2

## Watermarking observations

At `gamma = 0.25`, the watermarked generations are producing a higher
z-score than the human-written development text. This is the expected
direction: the watermark encourages generated tokens to come from a
pseudo-random green list more often than would occur by chance.

The detector z-score measures how far the observed number of green-list tokens
is from the number expected under the null hypothesis of unwatermarked text:

```text
z = (green_count - gamma * token_count)
    / sqrt(token_count * gamma * (1 - gamma))
```

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

## Q2 next steps

- Compute perplexity (PPL) for the generations from both seeds using the same
  base model, tokenizer, and scoring convention.
- Compare the two seeds on z-score, detection rate, text length, PPL, and
  generation quality.
- Apply meaning-preserving perturbations such as word substitutions,
  sentence-level paraphrasing, punctuation changes, whitespace changes, and
  sentence reordering, then measure how detection changes.
- Test whether the detector remains effective after truncation, prefix/suffix
  edits, and changes in text length.
- Sweep `gamma` and the detector threshold, and report detection rate,
  false-positive rate on human text, and the resulting ROC/precision-recall
  trade-offs.
- Generate multiple samples per prompt and report averages and variation across
  prompts and seeds, rather than only aggregate means.
- Check whether z-score differences remain after controlling for token count,
  prompt, and topic, since longer texts naturally give more stable detector
  statistics.
- Inspect examples with unusually high human z-scores and unusually low
  watermarked z-scores to identify tokenizer, prompt, or implementation
  artifacts.