import argparse
import json
import random
import string
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# bf16 is preferred: same range as fp32, so no loss-scaling problems.
# Only fall back to fp16 on GPUs that lack bf16 (e.g. T4).
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prompt(mode, row):
    # Options become "A. ...", "B. ..." so the answer is a single letter.
    labels = string.ascii_uppercase[: len(row["options"])]
    options_text = "\n".join(f"{lab}. {opt}" for lab, opt in zip(labels, row["options"]))

    if mode == "bias":
        # The instruction explicitly says "pick the unknown option if the context
        # is insufficient". This gives the model a concrete rule to learn,
        # instead of only inferring it from the labels.
        return (
            "Answer the question using only what the context explicitly states. "
            "If the context does not give enough information, choose the option "
            "meaning the answer cannot be determined. Do not guess from stereotypes. "
            "Reply with only the letter of the best option.\n\n"
            f"Category: {row['category']}\n"
            f"Context: {row['context']}\n"
            f"Question: {row['question']}\n"
            f"Options:\n{options_text}\n\n"
            "Answer:"
        )

    # QA mode: plain MCQ with no bias instruction (used for the utility check).
    return (
        "Answer the multiple-choice question. Reply with only the letter of the best option.\n\n"
        f"Question: {row['question']}\n"
        f"Options:\n{options_text}\n\n"
        "Answer:"
    )


def single_token_id(tokenizer, text):
    # We train on exactly ONE answer token, so each letter must be one token.
    # Fail loudly here instead of silently training on a wrong target.
    # NOTE: this checks "A" (no leading space). Qwen's " A" is a different token.
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected single token for {text!r}, got {ids}")
    return ids[0]


class OneTokenSFTDataset(Dataset):
    def __init__(self, rows, tokenizer, mode, shuffle=True):
        self.rows, self.tokenizer, self.mode, self.shuffle = rows, tokenizer, mode, shuffle

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        # Tokenization happens lazily here (not upfront) so that the option
        # shuffle below is re-drawn every time a sample is loaded, i.e. every epoch.
        row = self.rows[idx]
        perm = list(range(len(row["options"])))
        if self.shuffle:
            # Shuffling options stops the model from learning "answer is usually C"
            # or "the biased option sits at position X". Dev uses shuffle=False
            # so eval loss is stable and comparable between epochs.
            random.shuffle(perm)

        shuffled_row = dict(row)
        shuffled_row["options"] = [row["options"][j] for j in perm]

        # perm[new_pos] = old_pos, so perm.index(old_label) gives the new position
        # of the correct option. That position's letter is the training target.
        target_letter = string.ascii_uppercase[perm.index(int(row["label"]))]

        # Use the model's own chat template so training matches how it is used at
        # inference. add_generation_prompt=True appends "<|im_start|>assistant\n",
        # so the next token the model must produce is the answer letter.
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": format_prompt(self.mode, shuffled_row)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # add_special_tokens=False: the template already contains the special
        # tokens, adding more would duplicate them.
        prompt_ids = self.tokenizer.encode(chat, add_special_tokens=False)
        target_id = single_token_id(self.tokenizer, target_letter)

        # Sequence = prompt + answer letter.
        # Labels = -100 (ignored by the loss) for the whole prompt, real id only
        # on the answer. So the model is trained ONLY to predict the letter and
        # gets no gradient for "learning" to reproduce the prompt text.
        input_ids = prompt_ids + [target_id]
        labels = [-100] * len(prompt_ids) + [target_id]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


class Collator:
    # Prompts have different lengths, so pad each batch to its longest sample.
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(f["input_ids"].shape[0] for f in features)
        bsz = len(features)
        input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        # Padded label positions must be -100, otherwise padding would count in the loss.
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)
        # attention_mask 0 = "ignore this position" (padding).
        attention = torch.zeros((bsz, max_len), dtype=torch.long)
        for i, f in enumerate(features):
            n = f["input_ids"].shape[0]
            input_ids[i, :n] = f["input_ids"]
            labels[i, :n] = f["labels"]
            attention[i, :n] = f["attention_mask"]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


class BalancedCategoryTrainer(Trainer):
    # Same as Trainer, except training batches are sampled with per-row weights
    # so small categories (e.g. Physical_appearance) show up more often.
    def __init__(self, *args, category_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.category_weights = category_weights

    def get_train_dataloader(self):
        if self.category_weights is None:
            return super().get_train_dataloader()
        # replacement=True: a rare row can be drawn several times per epoch,
        # which is how oversampling works. Epoch length stays the same.
        sampler = WeightedRandomSampler(
            weights=self.category_weights,
            num_samples=len(self.train_dataset),
            replacement=True,
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def make_category_weights(rows, power=0.5):
    # weight = count ** -power
    #   power=0   -> no balancing (natural distribution)
    #   power=0.5 -> inverse sqrt, a mild boost for small categories
    #   power=1   -> full inverse frequency, every category equally likely
    # Full balancing would show the 120 Physical_appearance rows ~8x more often
    # than natural, which risks memorizing them, hence the soft default.
    counts = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    return torch.tensor([counts[r["category"]] ** -power for r in rows], dtype=torch.double)


def load_model_and_tokenizer(lora_r):
    if USE_BF16:
        dtype = torch.bfloat16
    elif torch.cuda.is_available() or torch.backends.mps.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding is fine for training. For BATCHED inference where you read the
    # last position, switch to left padding (or index by true length).
    tokenizer.padding_side = "right"

    # Base weights are loaded in half precision to save memory and stay frozen.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)

    lora_cfg = LoraConfig(
        r=lora_r,                  # rank = capacity of the adapter. 16 is plenty for this task.
        lora_alpha=2 * lora_r,     # update is scaled by alpha/r = 2. Keeps the effective step size
                                   # roughly stable if you change r.
        lora_dropout=0.05,         # light regularization, useful with only ~3k rows
        target_modules="all-linear",  # q,k,v,o,gate,up,down in all 28 blocks; excludes lm_head
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # sanity check: expect ~1% trainable
    return model, tokenizer


def main():
    p = argparse.ArgumentParser(description="One-token LoRA SFT on Qwen2.5-1.5B-Instruct.")
    p.add_argument("--mode", choices=["bias", "qa"], default="bias")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--dev-jsonl", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    # LoRA tolerates (and needs) a much higher LR than full fine-tuning (~2e-5).
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    # Effective batch = batch_size * grad_accum = 16. Accumulation fakes a bigger
    # batch without needing the memory for it.
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--balance-categories", action="store_true")
    p.add_argument("--balance-power", type=float, default=0.5)
    args = p.parse_args()

    # Seeds make runs reproducible (including the option shuffles).
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.train_jsonl)
    if not rows or not all("label" in r for r in rows):
        raise ValueError("Training file is empty or rows are missing 'label'.")

    model, tokenizer = load_model_and_tokenizer(args.lora_r)
    train_ds = OneTokenSFTDataset(rows, tokenizer, args.mode, shuffle=True)
    eval_ds = None
    if args.dev_jsonl:
        eval_ds = OneTokenSFTDataset(read_jsonl(args.dev_jsonl), tokenizer, args.mode, shuffle=False)

    weights = None
    if args.balance_categories:
        if not all("category" in r for r in rows):
            raise ValueError("--balance-categories needs a 'category' field in every row.")
        weights = make_category_weights(rows, args.balance_power)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",   # smooth decay to ~0, helps the last epoch settle
        warmup_steps=30,            # short ramp-up avoids a big first step on fresh adapters
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=20,
        # Eval loss each epoch lets you spot overfitting (train loss falls, eval rises).
        eval_strategy="epoch" if eval_ds else "no",  # `evaluation_strategy` on older transformers
        save_strategy="epoch",        # one checkpoint per epoch so you can pick the best one
        report_to=[],
        bf16=USE_BF16,
        fp16=torch.cuda.is_available() and not USE_BF16,  # never both True
        remove_unused_columns=False,  # our dataset returns custom keys; don't let Trainer drop them
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = BalancedCategoryTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=Collator(tokenizer.pad_token_id),
        processing_class=tokenizer,  # use `processing_class=` on newer transformers
        category_weights=weights,
    )
    trainer.train()

    # Saves the LoRA adapter only (a few tens of MB). For a standalone model:
    #   model = model.merge_and_unload(); model.save_pretrained(...)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()