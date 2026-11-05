import argparse
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

START = time.time()
REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ARTIFACT = "exp2_model.joblib"
MAX_NEW = 64


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(adapter_dir=None, model_dir=None):
    path = model_dir or REPO_ID
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model


def format_prompt(row):
    return (
        "Answer the question with a short answer. Do not explain your reasoning. "
        "If you are not confident, still provide your best answer.\n\n"
        f"Question: {row['question']}\n\n"
        "Answer:"
    )


def ptrue_prompt(question, answer):
    return (
        "You are grading an answer to a trivia question.\n\n"
        f"Question: {question}\n"
        f"Proposed answer: {answer}\n\n"
        "Is the proposed answer correct? Reply with exactly one word: True or False."
    )


def chat(tok, content):
    return tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def extract(row, tok, model, true_id, false_id, eos_ids):
    inputs = tok(chat(tok, format_prompt(row)), return_tensors="pt", add_special_tokens=False).to(model.device)
    gen = model.generate(
        **inputs, max_new_tokens=MAX_NEW, do_sample=False,
        repetition_penalty=1.0, return_dict_in_generate=True,
        output_scores=True, pad_token_id=tok.eos_token_id,
    )
    new = gen.sequences[0, inputs["input_ids"].shape[1]:]
    answer = tok.decode(new, skip_special_tokens=True).strip()

    lps, ents = [], []
    for sc, t in zip(gen.scores, new):
        lp = torch.log_softmax(sc[0].float(), dim=-1)
        lps.append(lp[int(t)].item())
        ents.append(-(lp.exp() * lp).sum().item())

    ended = len(new) > 0 and int(new[-1]) in eos_ids
    body_lp = lps[:-1] if ended else lps
    body_ent = ents[:-1] if ended else ents
    if not body_lp:
        body_lp, body_ent = lps or [-20.0], ents or [0.0]
    eos_lp = lps[-1] if ended else -14.0

    p = tok(chat(tok, ptrue_prompt(row["question"], answer)), return_tensors="pt", add_special_tokens=False).to(model.device)
    logits = model(**p).logits[0, -1].float()
    ptrue_logit = (logits[true_id] - logits[false_id]).item()

    feats = {
        "ptrue_logit": ptrue_logit,
        "mean_logprob": sum(body_lp) / len(body_lp),
        "mean_prob": sum(math.exp(x) for x in body_lp) / len(body_lp),
        "min_logprob": min(body_lp),
        "first_logprob": body_lp[0],
        "first_entropy": body_ent[0],
        "mean_entropy": sum(body_ent) / len(body_ent),
        "log_len": math.log1p(len(body_lp)),
        "eos_logprob": eos_lp,
        "truncated": 0.0 if ended else 1.0,
    }
    return answer, feats


def score(feats, art):
    row = np.array([[feats[name] for name in art["features"]]], dtype=np.float64)
    return float(art["model"].predict_proba(row)[0, 1])


def find_artifact(output_dir):
    candidates = [
        Path(output_dir) / ARTIFACT,
        Path(__file__).resolve().parent / "checkpoints" / ARTIFACT,
        Path(__file__).resolve().parent / ARTIFACT,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"{ARTIFACT} not found in {output_dir} or beside infer.py")


def find_base_model(model_dir, output_dir):
    if model_dir:
        return model_dir
    for path in (Path(output_dir), Path(__file__).resolve().parent / "checkpoints"):
        if (path / "config.json").is_file():
            return str(path)
    return None


def main():
    ap = argparse.ArgumentParser(description="Run Q3 calibrated QA inference (Exp 2).")
    ap.add_argument("input_jsonl")
    ap.add_argument("output_dir")
    ap.add_argument("--model-dir", default=None, help="Local M-inst directory")
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--time-budget", type=float, default=2400.0, help="seconds; after this, remaining rows get a cheap fallback so every id has a row")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art = joblib.load(find_artifact(args.output_dir))
    rows = read_jsonl(args.input_jsonl)
    tok, model = load_model(args.adapter_dir, find_base_model(args.model_dir, args.output_dir))

    true_ids = tok.encode("True", add_special_tokens=False)
    false_ids = tok.encode("False", add_special_tokens=False)
    assert len(true_ids) == 1 and len(false_ids) == 1
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])

    n_fallback = 0
    output_path = out_dir / "output.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            pred = None
            if time.time() - START < args.time_budget:
                try:
                    answer, feats = extract(row, tok, model, true_ids[0], false_ids[0], eos_ids)
                    conf = max(0.0, min(1.0, score(feats, art)))
                    pred = {"id": row["id"], "answer": answer, "confidence": conf, "abstain": conf < art["threshold"]}
                except Exception as e:  # e.g. OOM on one row: never lose the whole file
                    print(f"row {i}: {type(e).__name__}: {e}", flush=True)
            if pred is None:
                n_fallback += 1
                pred = {"id": row["id"], "answer": "unknown", "confidence": float(art.get("prior", 0.16)), "abstain": True}
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(rows)}  {time.time() - START:.0f}s", flush=True)
    print(f"Wrote {len(rows)} predictions to {output_path} ({n_fallback} fallback rows)")


if __name__ == "__main__":
    main()