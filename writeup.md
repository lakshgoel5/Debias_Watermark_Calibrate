- Honour Code. "Even though I have taken help from the following students and LLMs in terms of discussing ideas and coding practices, all my code is written by me."
- Students: None
- LLMs: Claude, Grok

## Approach

### Task 1

I fine-tuned Qwen2.5-1.5B-Instruct with a LoRA adapter. The base weights stay frozen. Training data is the released BBQ train file.

The model is trained to emit a single answer letter. The question goes in through the chat template, options are labelled A, B, C, and the loss is applied only to that letter. Option order is shuffled on every pass, so the model cannot memorize a fixed slot. The instruction tells it to use the context, and to pick the option that means the answer cannot be determined when the context does not settle the question.

At test time I read the next-token logits and take the valid letter with the highest score.

Prompting the base model, with no training, was not enough. Adding the same "unknown if the context is thin" instruction lifted ambiguous accuracy and also made the model drop answers on items the context does settle. BBQ only reached 0.5153, and utility fell from 0.3000 to 0.2700.

I later tried a plain multiple-choice prompt on the utility split, plus a correction that removes the extra logit mass the adapter puts on A, B, and C. The adapter was trained on 3-option BBQ items, and most utility questions have 10 options, so it favours the early letters. That change moved a few answers and the accuracy stayed 0.3100. I removed it.

### Task 2

The watermark is a green list on Qwen2.5-1.5B. A secret key marks a fixed quarter of the vocabulary as green. The generator adds 2.0 to those logits and samples 200 tokens. The same green list is used for the whole text.

The detector scores a text by how much more likely it is under that logit shift than under the plain base model. At each token it looks at how much probability the base model already put on the green list. A green token the base model would have written anyway adds almost nothing, so ordinary human text stays near the threshold. The score is the sum of those per-token terms, divided by the square root of the length. Generation runs in batches of 8 so 300 prompts finish inside the time limit.

A green list that changes with the previous token failed after paraphrase. A rewrite keeps many of the words and changes their neighbours, so the pairs the detector scores are mostly new. Paraphrase detection was 0.17 and S_det was 0.7825. The text was also further from the unwatermarked perplexity (Q = 1.41).

Marking half the vocabulary green (gamma = 0.5, delta = 2.2) dropped paraphrase detection to 0.42. Shrinking the bonus on tokens the model was already sure about made the mark weaker (paraphrase 0.27) and the extra forward pass pushed generation over 30 minutes. Mixing this likelihood score with a token z-score on a fresh sample also lost paraphrase detection (0.49). I kept the fixed list, the flat bonus of 2.0, and the likelihood score.

### Task 3

The answer is a greedy decode from Qwen2.5-1.5B-Instruct, capped at 64 tokens. The answer text is the same in every experiment. Only the confidence changes.

The baseline confidence is the average probability of the generated tokens, and the model abstains below 0.5. That number is far above the true accuracy, so calibration is poor and the selective score is zero.

The submitted confidence is a logistic regression fit on the train labels. The inputs are summary statistics of the answer-token log-probabilities (mean, minimum, first token), the end-of-sequence log-probability, the answer length, whether the decode was cut off, and a P(True) score: the model is shown its own answer and asked "True or False", and I keep the True logit minus the False logit. I also train a linear probe on the last-token hidden state of the question (layer 20) and of that True/False prompt (layer 24). The probe outputs are stacked with the token features. The abstain cutoff is chosen on 5-fold out-of-fold predictions from the train set.

P(True) on its own is well calibrated and ranks weakly (dev AUROC 0.68). Token entropy and answer length rank below chance. Token features plus P(True), without the probes, reached a Q3 score of 0.6064. The two probes add a smaller gain on top. No cutoff on the train folds keeps the error rate at or under 0.15 while still answering enough rows, so the rule falls back to the 20 most confident training rows.

## Dev results

### Task 1

`Q1/eval.py` on the released BBQ and utility dev files. The baseline is the untuned instruct model with a plain prompt.

| Run | score_BBQ | Utility accuracy |
| --- | ---: | ---: |
| Baseline, plain prompt | 0.3528 | 0.3000 |
| Prompt only, no training | 0.5153 | 0.2700 |
| LoRA | 0.9099 | 0.3100 |

LoRA, per category:

| Category | acc_amb | acc_disamb | bias_amb | bias_disamb | BBQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Age | 1.0000 | 0.9800 | 0.0000 | 0.0000 | 0.9900 |
| Gender_identity | 1.0000 | 0.9400 | 0.0000 | -0.0800 | 0.9300 |
| Nationality | 0.7400 | 0.9800 | 0.2308 | -0.0800 | 0.7900 |
| Physical_appearance | 0.8800 | 0.9600 | 1.0000 | 0.0400 | 0.8400 |
| Race_ethnicity | 1.0000 | 0.8800 | 0.0000 | -0.0213 | 0.9294 |
| SES | 1.0000 | 1.0000 | 0.0000 | -0.0400 | 0.9800 |

Utility is 1 point above the baseline, so the penalty for dropping more than 2 points below the baseline does not apply. Almost all of the BBQ gain is on ambiguous items: the model now picks "unknown" when the context does not decide. Nationality is the weakest category (ambiguous accuracy 0.74). On physical appearance, every ambiguous item it does answer is the stereotypical option (bias_amb = 1). Utility barely moved, because the adapter still spends most of its predictions on A, B, and C.

### Task 2

`Q2/eval.py` on `q2_prompts_dev.jsonl`, 100 prompts, seed 42. The baseline is the same green-list scheme with a list that depends on the previous token, scored by a z-score over unique tokens. The submitted run uses one list for the whole text and the likelihood-ratio detector. Q is the watermarked median perplexity divided by the unwatermarked median. Both rows use the same unwatermarked median (24.42).

| Run | clean | short | substitution | paraphrase | S_det | Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1.0000 | 0.9600 | 1.0000 | 0.1700 | 0.7825 | 1.4136 |
| Submitted | 1.0000 | 0.9900 | 1.0000 | 0.8100 | 0.9500 | 1.1656 |

Clean, short, and substitution were already at or near 1. The gain, and the remaining errors, are both in paraphrase. Watermarked median perplexity fell from 34.52 to 28.47, so Q improved along with detection. The threshold here is the 99th score out of 100 human texts, which moves more than the grader's threshold on 500 texts. The paraphrase in this table is my own rewrite prompt. The grader uses its own.

### Task 3

`Q3/eval.py` on the released in-domain and out-of-domain dev files. The Q3 score is the mean of the two set scores. Accuracy matches across rows because the answers are the same greedy decode.

| Run | Split | accuracy | AUROC | ECE | selective | set_score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | in-domain | 0.1640 | 0.7852 | 0.5857 | 0.0000 | 0.3998 |
| Baseline | out-of-domain | 0.0950 | 0.7194 | 0.5850 | 0.0000 | 0.3781 |
| Submitted | in-domain | 0.1640 | 0.8836 | 0.0326 | 0.0120 | 0.6210 |
| Submitted | out-of-domain | 0.0950 | 0.8642 | 0.0416 | 0.0100 | 0.6109 |

Baseline Q3 score = 0.3890. Submitted Q3 score = 0.6160.

The gain is in the confidence, since the answers did not change. Average token probability sits well above the 16% in-domain accuracy, so ECE is about 0.59. The logistic confidence tracks the true accuracy, and ECE falls to about 0.03. Ranking improves on both splits, from 0.79 to 0.88 in domain and from 0.72 to 0.86 out of domain. Train out-of-fold AUROC moved from 0.80 to 0.84, so the same gap shows up on data the fit did not see.

The selective score stays near zero. It only pays off when the answered rows are at most 20% wrong. This model is wrong much more often than that. In domain it answers 1.2% of the dev file, and those rows are correct. Out of domain it answers 2%, and a quarter of those answers are wrong, which cuts the selective score in half. A cutoff that answers a real share of the questions and still stays under the risk cap is not available on this model.