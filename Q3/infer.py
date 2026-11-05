import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"


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


def predict(row, tokenizer, model, abstain_threshold):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": format_prompt(row)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            # Qwen's generation_config sets repetition_penalty=1.1. That processor still
            # runs under greedy decoding and changes output_scores. 1.0 leaves raw logits.
            repetition_penalty=1.0,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated.sequences[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Three summaries of the same token probabilities. confidence stays the arithmetic
    # mean, which abstain uses. confidence_geo is exp(mean log p). confidence_min is min p_t.
    token_log_probs = []
    for scores, token_id in zip(generated.scores, new_tokens):
        log_probs = torch.log_softmax(scores[0], dim=-1)
        token_log_probs.append(float(log_probs[int(token_id)].item()))
    if token_log_probs:
        logps = torch.tensor(token_log_probs)
        probs = torch.exp(logps)
        confidence = float(probs.mean().item())
        confidence_geo = float(torch.exp(logps.mean()).item())
        confidence_min = float(probs.min().item())
    else:
        confidence = confidence_geo = confidence_min = 0.0
    confidence = max(0.0, min(1.0, confidence))
    confidence_geo = max(0.0, min(1.0, confidence_geo))
    confidence_min = max(0.0, min(1.0, confidence_min))

    return {
        "id": row["id"],
        "answer": answer,
        "confidence": confidence,
        "confidence_geo": confidence_geo,
        "confidence_min": confidence_min,
        "abstain": confidence < abstain_threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Q3 calibrated QA inference.")
    parser.add_argument("input_jsonl", help="Path to input JSONL file")
    parser.add_argument("output_dir", help="Directory to write output JSONL")
    parser.add_argument("--model-dir", default=None, help="Local M-inst directory")
    parser.add_argument("--adapter-dir", default=None, help="Optional LoRA adapter directory")
    parser.add_argument(
        "--abstain-threshold",
        type=float,
        default=0.5,
        help="Abstain when generated-answer confidence is below this value",
    )
    args = parser.parse_args()
    if not 0.0 <= args.abstain_threshold <= 1.0:
        raise ValueError("--abstain-threshold must be in [0, 1]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.input_jsonl)
    tokenizer, model = load_model(args.adapter_dir, args.model_dir)
    predictions = [predict(row, tokenizer, model, args.abstain_threshold) for row in rows]

    output_path = output_dir / "output.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
