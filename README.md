# Debias, Watermark, Abstain

- Honour Code. "Even though I have taken help from the following students and LLMs in terms of discussing ideas and coding practices, all my code is written by me."
- Students: None
- LLMs: Claude, Grok

## Running the code

```bash
pip install -r requirements.txt
```

Run each task from inside its own folder, since the scripts use relative paths.

### Task 1

```bash
cd Q1
bash download.sh ../predictions/Q1
bash infer.sh bias ../predictions/Q1 ../data/public/q1_bbq_dev.jsonl
bash infer.sh qa   ../predictions/Q1 ../data/public/q1_qa_dev.jsonl
python eval.py --bbq-data ../data/public/q1_bbq_dev.jsonl --bbq-preds ../predictions/Q1/output_bias.jsonl --qa-data  ../data/public/q1_qa_dev.jsonl  --qa-preds  ../predictions/Q1/output_qa.jsonl
```

### Task 2

```bash
cd Q2
bash setup.sh secret
bash generate.sh ../data/public/q2_prompts_dev.jsonl secret 42
bash perturb.sh secret/generated_42.jsonl secret 42

  bash detect.sh "secret/generated_clean_42.jsonl" secret
  bash detect.sh "secret/generated_substitution_42.jsonl" secret
  bash detect.sh "secret/generated_paraphrase_42.jsonl" secret
  bash detect.sh "secret/human.jsonl" secret
  bash detect.sh "secret/human_short.jsonl" secret

python eval.py ../data/public/q2_prompts_dev.jsonl secret 42
```

### Task 3

```bash
cd Q3
bash download.sh  ../predictions/Q3
bash infer.sh     ../data/public/q3_qa_dev.jsonl              ../predictions/Q3
bash infer.sh     ../data/public/q3_qa_dev_ood.jsonl          ../predictions/Q3

python eval.py q3 ../predictions/Q3/pred_q3_qa_dev.jsonl      ../data/public/q3_qa_dev.jsonl
python eval.py q3 ../predictions/Q3/pred_q3_qa_dev_ood.jsonl  ../data/public/q3_qa_dev_ood.jsonl
```

### Notes

- `download.sh` fetches the trained model. To train instead: `bash train.sh ../predictions/Q3` for
  Task 3, and for Task 1 `python train_sft_lora.py --mode bias --train-jsonl
  ../data/public/q1_bbq_train.jsonl --dev-jsonl ../data/public/q1_bbq_dev.jsonl --output-dir
  ../predictions/Q1` (`train.sh` calls `train.py`, which is not the filename here).
- Q1 and Q3 use one directory for both the model and the predictions, so download into the same
  directory you pass to `infer.sh`.
- `Q2/eval.py` is dev-only and downloads Qwen2.5-3B (about 6 GB) to compute Q.

## Approach

### Task 1

I fine-tuned Qwen2.5-1.5B-Instruct with a LoRA adapter. The base weights stay frozen. Training data is the released BBQ train file.

The model is trained to emit a single answer letter. The question goes in through the chat template, options are labelled A, B, C, and the loss is applied only to that letter. Option order is shuffled on every pass, so the model cannot memorize a fixed slot. The instruction tells it to use the context, and to pick the option that means the answer cannot be determined when the context does not settle the question.

At test time I read the next-token logits and take the valid letter with the highest score.

Prompting the base model, with no training, was not enough. Adding the same "unknown if the context is thin" instruction lifted ambiguous accuracy and also made the model drop answers on items the context does settle. BBQ only reached 0.5153, and utility fell from 0.3000 to 0.2700. Category scores are in the Archive.

I later tried a plain multiple-choice prompt on the utility split, plus a correction that removes the extra logit mass the adapter puts on A, B, and C. The adapter was trained on 3-option BBQ items, and most utility questions have 10 options, so it favours the early letters. That change moved a few answers and the accuracy stayed 0.3100. I removed it.

### Task 2

The watermark is a green list on Qwen2.5-1.5B. A secret key marks a fixed quarter of the vocabulary as green. The generator adds 2.0 to those logits and samples 200 tokens at temperature 1. The same green list is used for the whole text. Prompts are decoded in batches of 8.

The detector retokenizes the text and does not load the language model. Each distinct token is scored once, with weight 1 + ln(count). The score is the weighted excess of green tokens over the 0.25 rate, divided by the null standard deviation. A repeated word adds evidence, and one word repeated many times cannot dominate the score.

A green list tied to the previous token does not survive paraphrase. The other settings I ran are in the Archive.

### Task 3

The answer is a greedy decode from Qwen2.5-1.5B-Instruct, capped at 64 tokens. The answer text is the same in every experiment. Only the confidence changes.

The baseline confidence is the average probability of the generated tokens, and the model abstains below 0.5. That number is far above the true accuracy, so calibration is poor and the selective score is zero.

The submitted confidence is a logistic regression fit on the train labels. The inputs are summary statistics of the answer-token log-probabilities (mean, minimum, first token), the end-of-sequence log-probability, the answer length, whether the decode was cut off, and a P(True) score: the model is shown its own answer and asked "True or False", and I keep the True logit minus the False logit. I also train a linear probe on the last-token hidden state of the question (layer 20) and of that True/False prompt (layer 24). The probe outputs are stacked with the token features. The abstain cutoff is chosen on 5-fold out-of-fold predictions from the train set.

P(True) on its own is well calibrated and ranks weakly (dev AUROC 0.68). Token entropy and answer length rank below chance. Token features plus P(True), without the probes, reached a Q3 score of 0.6064. The two probes add a smaller gain on top. The comparisons are in the Archive. I look for a cutoff on the train folds that keeps the error rate at or under 0.15, a margin stricter than the eval script's own 0.2 risk cap, while still answering enough rows. No such cutoff exists, so the rule falls back to the 20 most confident training rows: the threshold is set to the confidence score of the 20th most confident training row.

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

`Q2/eval.py` on `q2_prompts_dev.jsonl`, 100 prompts, seed 42. The baseline uses a green list that depends on the previous token, so each scored unit is a (previous token, token) pair rather than a token alone, and the z-score counts each unique pair once. The submitted run uses one list for the whole text and the log-count z-score above. Q is the watermarked median perplexity divided by the unwatermarked median. Both rows use the same unwatermarked median (24.421). The dev run, from generation through scoring, finished in 500 seconds.

| Run | clean | short | substitution | paraphrase | S_det | Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1.0000 | 0.9600 | 1.0000 | 0.1700 | 0.7825 | 1.4136 |
| Submitted | 1.0000 | 1.0000 | 1.0000 | 0.5100 | 0.8775 | 1.2721 |

Clean, short, and substitution are saturated. Paraphrase is the remaining gap, at 0.51. Watermarked median perplexity is 31.067, against 34.520 for the baseline, so Q falls from 1.4136 to 1.2721. The threshold here is the 99th score out of 100 human texts, which moves more than the grader's threshold on 500 texts. The paraphrase in this table is my own rewrite prompt. The grader uses its own.

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

The selective score stays near zero. The eval script's selective-score formula only pays off in full when the answered rows are at most 20% wrong (a looser bar than the 0.15 cap used to pick the cutoff above), and it decays to zero past 30% wrong. This model is wrong much more often than that. In domain it answers 1.2% of the dev file, and those rows are correct. Out of domain it answers 2%, and a quarter of those answers are wrong, which cuts the selective score in half. A cutoff that answers a real share of the questions and still stays under the risk cap is not available on this model.

## Archive

### Task 1

Prompt only, no training, on the BBQ dev file. The summary is in Dev results (BBQ 0.5153, utility 0.2700).

| Category | acc_amb | acc_disamb | bias_amb | bias_disamb | BBQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Age | 0.5200 | 0.6200 | 0.1667 | 0.3478 | 0.3561 |
| Gender_identity | 0.5400 | 0.7000 | 0.2174 | 0.0000 | 0.5700 |
| Nationality | 0.5800 | 0.8000 | -0.1429 | -0.0222 | 0.6489 |
| Physical_appearance | 0.5200 | 0.6000 | 0.5833 | 0.3333 | 0.2533 |
| Race_ethnicity | 0.6400 | 0.7600 | 0.3333 | -0.0500 | 0.6150 |
| SES | 0.7400 | 0.7600 | -0.6923 | -0.0233 | 0.6484 |

Against the plain prompt, ambiguous accuracy rose in every category (from 0.08–0.20 to 0.52–0.74) and disambiguated accuracy fell in every category (from 0.70–0.96 to 0.60–0.80). Physical appearance stayed the worst category. A later QA prompt and an A/B/C logit correction left both scores where the LoRA run already was, 0.9099 and 0.3100.

### Task 2

Ablations on the same dev file (100 prompts, seed 42). Each row is a Kaggle generation. The half-vocabulary and entropy-gate runs use a different unwatermarked median (27.975), so their Q is not on the 24.421 baseline in the table above.

| Run | clean | short | substitution | paraphrase | S_det | Q | Generation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Previous-token list | 1.0000 | 0.9600 | 1.0000 | 0.1700 | 0.7825 | 1.4136 | not logged |
| Fixed list, unweighted z | 1.0000 | 0.9500 | 1.0000 | 0.5900 | 0.8850 | 1.1656 | not logged |
| Half the vocabulary | 1.0000 | 0.9900 | 1.0000 | 0.4200 | 0.8525 | 0.9463 | not logged |
| Entropy gate | 1.0000 | 0.9300 | 1.0000 | 0.2700 | 0.8000 | 0.9208 | 40.1 min projected |
| White-box, flat bonus | 1.0000 | 0.9800 | 1.0000 | 0.4900 | 0.8675 | 1.1 | 41.2 min projected |

The previous-token list is the baseline (gamma = 0.25, delta = 2.0). A rewrite keeps words and changes their neighbours, so the pairs the detector scores are mostly new, and paraphrase detection is 0.17. Watermarked median perplexity was 34.520.

One fixed list is what raises paraphrase detection. On that sample each token had weight 1, gamma = 0.25, and delta = 2.0. Paraphrase detection was 0.59, S_det was 0.8850, and watermarked median perplexity was 28.466, so Q was 1.1656 on the same unwatermarked median of 24.421.

Marking half the vocabulary green (gamma = 0.5, delta = 2.2, one list, flat bonus, z-score) dropped paraphrase detection to 0.42. Watermarked median perplexity was 26.473 against an unwatermarked median of 27.975.

The entropy-gate run scaled the bonus by how unsure the model was, delta = 2.5 times (1 − p_max), and scored with a base-model likelihood ratio mixed with the z-score. Paraphrase detection fell to 0.27, and generation projected to 40.1 minutes for 300 prompts. Watermarked median perplexity was 25.759 against the same 27.975 unwatermarked draw.

The same mixed detector with a flat bonus of 2.0, gamma = 0.25, and one list, on a new sample, scored paraphrase detection 0.49 and S_det 0.8675, with Q = 1.1. Generation projected to 41.2 minutes.

I kept the fixed list, gamma = 0.25, and the flat bonus of 2.0. The submitted detector is the log-count z-score, and generation is batched.

### Task 3

Same greedy answers. These rows are the confidence models that were not submitted. Q3 score is filled in only where selective was also scored.

| Model | dev AUROC | dev ECE | ood AUROC | ood ECE | Q3 score |
| --- | ---: | ---: | ---: | ---: | ---: |
| P(True) only | 0.6833 | 0.0168 | 0.6592 | 0.0303 | — |
| White-box tokens | 0.8329 | 0.0342 | 0.8095 | 0.0452 | — |
| Tokens + P(True) | 0.8601 | 0.0399 | 0.8369 | 0.0292 | 0.6064 |

P(True) alone is the best calibrated and the worst ranker. The token features carry the ranking. Adding P(True) to them is a small AUROC gain and a small ECE cost. The best single token feature is the mean log-probability in domain (dev AUROC 0.812) and the minimum log-probability out of domain (0.889).

Probe scores stacked on tokens + P(True). The layer was chosen by train out-of-fold AUROC: question layer 20 (0.7944 alone) and P(True) layer 24 (0.8219 alone).

| Added probe | train-OOF AUROC | dev AUROC | ood AUROC | dev ECE | ood ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | 0.8035 | 0.8601 | 0.8369 | 0.0399 | 0.0292 |
| Question | 0.8351 | 0.8817 | 0.8689 | 0.0397 | 0.0238 |
| P(True) | 0.8383 | 0.8832 | 0.8473 | 0.0369 | 0.0270 |
| Both | 0.8423 | 0.8836 | 0.8642 | 0.0326 | 0.0416 |

Both probes have the highest train out-of-fold AUROC, which is why that model was kept. On the out-of-domain file the question probe alone is a bit better (AUROC 0.8689, ECE 0.0238) than both together. The Q3 score only moves from 0.6064 to 0.6160, and the selective score stays near zero in every version.

## Note

The sampling rule fixes the distribution in `generate.sh`. It does not freeze one random draw, so a second run can print a different `S_det` from the same code.

Each of the 200 steps adds 2.0 to the green-list logits and draws with softmax then multinomial, which is temperature 1.0. There is no top-k, top-p, or repetition penalty, and the loop does not stop if an end-of-sequence token is drawn. `human_text` is copied onto the output line after those 200 draws, so it does not change the distribution.

Multinomial sampling is still random. The same seed does not repeat the same 200 tokens on a GPU, because the attention math is not bitwise identical from run to run. A new `generated_<seed>.jsonl` is a new sample from the same watermark, and both `S_det` and `Q` can move.

Temperature 0.7 and top-p 0.9 are only in the paraphrase rewrite inside `perturb.sh`. That step is not the watermark. Re-running it draws a new paraphrase, and paraphrase is the condition that moves `S_det` when the watermarked tokens themselves are unchanged.
