"""Train the Q3 confidence model that infer.py loads (Exp 3).

features: greedy answers plus P(True) and white-box token features.
hidden: last-token states for the question prompt and the P(True) prompt.
fit: hidden-state probes stacked on those features. Writes exp3_model.joblib.

Usage (or just ./train.sh):
  python3 train.py features ../data/public/q3_qa_train.jsonl ../predictions/Q3/train.jsonl
  python3 train.py hidden ../data/public/q3_qa_train.jsonl ../predictions/Q3/train.jsonl ../predictions/Q3/hid_train.npz
  python3 train.py fit --train ../predictions/Q3/train.jsonl --dev ../predictions/Q3/dev.jsonl --ood ../predictions/Q3/ood.jsonl \
      --train-hid ../predictions/Q3/hid_train.npz --dev-hid ../predictions/Q3/hid_dev.npz --ood-hid ../predictions/Q3/hid_ood.npz
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval import auroc, ece, is_correct, selective_score
from infer import chat, extract, format_prompt, load_model, ptrue_prompt, read_jsonl

ALL = ["ptrue_logit", "mean_logprob", "min_logprob", "first_logprob", "first_entropy",
       "mean_entropy", "log_len", "eos_logprob", "truncated"]
NO_PTRUE = [f for f in ALL if f != "ptrue_logit"]
PROBE_C = [0.0003, 0.001, 0.003, 0.01]


def extract_features(args):
    rows = read_jsonl(args.input_jsonl)
    tok, model = load_model(args.adapter_dir, args.model_dir)

    true_ids = tok.encode("True", add_special_tokens=False)
    false_ids = tok.encode("False", add_special_tokens=False)
    assert len(true_ids) == 1 and len(false_ids) == 1, (true_ids, false_ids)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            answer, feats, _ = extract(row, tok, model, true_ids[0], false_ids[0], eos_ids)
            rec = {"id": row["id"], "answer": answer, "features": feats}
            if "answers" in row:
                rec["correct"] = int(is_correct(answer, row["answers"]))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(rows)}", flush=True)
    print(f"Wrote {len(rows)} rows to {args.output_jsonl}")


@torch.no_grad()
def last_token_states(texts, tok, model, layers, bs):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        hs = model(**enc, position_ids=pos, output_hidden_states=True).hidden_states
        out.append(torch.stack([hs[l][:, -1, :].float().cpu() for l in layers], dim=1).numpy())
        if (i // bs) % 20 == 0:
            print(f"{i}/{len(texts)}", flush=True)
    return np.concatenate(out).astype(np.float16)


def extract_hidden(args):
    rows = read_jsonl(args.input_jsonl)
    answers = {str(r["id"]): r["answer"] for r in read_jsonl(args.feats_jsonl)}
    tok, model = load_model(args.adapter_dir, args.model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    n_layers = model.config.num_hidden_layers
    layers = args.layers or sorted({round(n_layers * f) for f in (0.4, 0.55, 0.7, 0.85, 1.0)})
    print("layers:", layers, "of", n_layers)

    q_texts = [chat(tok, format_prompt(r)) for r in rows]
    p_texts = [chat(tok, ptrue_prompt(r["question"], answers[str(r["id"])])) for r in rows]
    hq = last_token_states(q_texts, tok, model, layers, args.bs)
    hp = last_token_states(p_texts, tok, model, layers, args.bs)

    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz, q=hq, pt=hp, layers=np.array(layers),
        ids=np.array([str(r["id"]) for r in rows]),
    )
    print("saved", args.output_npz, hq.shape)


def align_hidden(npz_path, feats_path):
    z = np.load(npz_path)
    ids_f = [str(json.loads(line)["id"]) for line in open(feats_path, encoding="utf-8") if line.strip()]
    pos = {str(i): k for k, i in enumerate(z["ids"])}
    order = [pos[i] for i in ids_f]
    return z["q"][order], z["pt"][order], [int(layer) for layer in z["layers"]]


def load_features(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    X = {k: np.array([r["features"][k] for r in rows]) for k in rows[0]["features"]}
    y = np.array([r["correct"] for r in rows])
    return X, y


def mat(X, cols):
    return np.stack([X[c] for c in cols], axis=1)


def metrics(conf, y, abstain=None):
    c, yy = list(map(float, conf)), list(map(int, y))
    out = {"acc": float(np.mean(y)), "AUROC": auroc(yy, c), "ECE": ece(yy, c),
           "Brier": float(np.mean((np.array(c) - y) ** 2))}
    if abstain is not None:
        out["selective"] = selective_score(yy, c, list(map(bool, abstain)))
        out["coverage"] = float(1 - np.mean(abstain))
        ans = ~np.array(abstain)
        out["risk"] = float(1 - y[ans].mean()) if ans.any() else 0.0
    return out


def choose_threshold(conf, y, max_risk, min_n):
    """Lowest threshold (max coverage) whose answered-risk <= max_risk.
    Needs at least min_n answered rows so the estimate is not a fluke.
    Fallback (no slice is safe enough): answer only the top-min_n most confident rows.
    """
    best = None
    for t in np.unique(conf):
        ans = conf >= t
        if ans.sum() < min_n:
            continue
        risk = 1 - y[ans].mean()
        if risk <= max_risk and (best is None or ans.mean() > best[1]):
            best = (float(t), float(ans.mean()))
    if best:
        return best[0], True
    return float(np.sort(conf)[::-1][min_n - 1]), False


def show(name, m):
    print(f"  {name:22s} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))


def make_probe(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def fit_model(args):
    Xtr, ytr = load_features(args.train)
    Xdv, ydv = load_features(args.dev)
    Xod, yod = load_features(args.ood)

    print("Accuracy  train/dev/ood: %.3f / %.3f / %.3f" % (ytr.mean(), ydv.mean(), yod.mean()))

    print("\n[1] Single-feature AUROC (higher = better ranking)")
    for f in ALL:
        print(f"  {f:16s} train={auroc(list(ytr), list(Xtr[f])):.3f} "
              f"dev={auroc(list(ydv), list(Xdv[f])):.3f} ood={auroc(list(yod), list(Xod[f])):.3f}")

    def fit(cols):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=1000))
        m.fit(mat(Xtr, cols), ytr)
        return m

    print("\n[2] Logistic regression ablations (raw LR probabilities, no abstain)")
    models = {}
    for name, cols in [("ptrue only", ["ptrue_logit"]),
                       ("white-box only", NO_PTRUE),
                       ("white-box + ptrue", ALL)]:
        models[name] = (fit(cols), cols)
        print(f" {name}")
        for split, X, y in [("dev", Xdv, ydv), ("ood", Xod, yod)]:
            show(split, metrics(models[name][0].predict_proba(mat(X, cols))[:, 1], y))

    splits = ("train", "dev", "ood")
    feats_path = {"train": args.train, "dev": args.dev, "ood": args.ood}
    hid_path = {"train": args.train_hid, "dev": args.dev_hid, "ood": args.ood_hid}
    H, layer_lists = {}, []
    for split in splits:
        q, pt, layers = align_hidden(hid_path[split], feats_path[split])
        H[split] = {"q": q, "pt": pt}
        layer_lists.append(layers)
    if any(layers != layer_lists[0] for layers in layer_lists):
        raise ValueError(f"hidden-state layer lists differ: {layer_lists}")
    layers = layer_lists[0]
    y = {"train": ytr, "dev": ydv, "ood": yod}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    print("\n[3] Probe search: train 5-fold OOF AUROC (top 3 per source)")
    probes, P = {}, {split: {} for split in splits}
    for src in ("q", "pt"):
        results = []
        for li, layer in enumerate(layers):
            Z = H["train"][src][:, li, :].astype(np.float32)
            for C in PROBE_C:
                oof = cross_val_predict(make_probe(C), Z, y["train"], cv=cv, method="decision_function")
                results.append((roc_auc_score(y["train"], oof), li, layer, C, oof))
        results.sort(key=lambda row: -row[0])
        for auc, li, layer, C, _ in results[:3]:
            print(f"  {src:2s} layer={layer:2d} C={C:<7} oofAUROC={auc:.4f}")
        auc, li, layer, C, oof = results[0]
        final = make_probe(C).fit(H["train"][src][:, li, :].astype(np.float32), y["train"])
        probes[src] = {"layer": int(layer), "layer_index": int(li), "C": C, "pipe": final}
        P["train"][src] = oof
        for split in ("dev", "ood"):
            P[split][src] = final.decision_function(H[split][src][:, li, :].astype(np.float32))
        print(f"  -> {src}: layer {layer}, C={C}; alone dev AUROC="
              f"{roc_auc_score(y['dev'], P['dev'][src]):.3f} "
              f"ood AUROC={roc_auc_score(y['ood'], P['ood'][src]):.3f}")

    base = {split: mat({"train": Xtr, "dev": Xdv, "ood": Xod}[split], ALL) for split in splits}
    variants = {"exp2 features": [], "+ probe q": ["q"], "+ probe pt": ["pt"], "+ both probes": ["q", "pt"]}

    def design(split, srcs):
        return np.column_stack([base[split]] + [P[split][src] for src in srcs])

    print("\n[4] Stacked logistic regression (train-OOF is the most reliable column)")
    fitted, oofs = {}, {}
    for name, srcs in variants.items():
        stacked = make_probe(args.C).fit(design("train", srcs), y["train"])
        oofs[name] = cross_val_predict(
            make_probe(args.C), design("train", srcs), y["train"], cv=cv, method="predict_proba"
        )[:, 1]
        fitted[name] = (stacked, srcs)
        print(f" {name}")
        show("train-oof", metrics(oofs[name], y["train"]))
        for split in ("dev", "ood"):
            show(split, metrics(stacked.predict_proba(design(split, srcs))[:, 1], y[split]))

    best = "exp2 features"
    best_auc = roc_auc_score(y["train"], oofs[best])
    for name in variants:
        auc = roc_auc_score(y["train"], oofs[name])
        if auc > best_auc + 0.003:
            best, best_auc = name, auc
    stacker, srcs = fitted[best]
    print(f"\n[5] Chosen variant (train-OOF AUROC {best_auc:.4f}): {best}")

    tau, feasible = choose_threshold(oofs[best], y["train"], args.max_risk, args.min_n)
    note = "" if feasible else "  (fallback: top-%d rows)" % args.min_n
    print(f"    threshold from train-OOF: tau={tau:.4f}{note}")
    for split in ("dev", "ood"):
        conf = stacker.predict_proba(design(split, srcs))[:, 1]
        scored = metrics(conf, y[split], abstain=conf < tau)
        scored["set_score"] = (scored["AUROC"] + (1 - scored["ECE"]) + scored["selective"]) / 3
        show(split, scored)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"variant": best, "srcs": srcs, "probes": probes, "stacker": stacker,
                 "features": ALL, "threshold": tau}, out)
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser(description="Train the Q3 confidence model.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    fe = sub.add_parser("features", help="Extract answer, features, and correctness labels")
    fe.add_argument("input_jsonl")
    fe.add_argument("output_jsonl")
    fe.add_argument("--model-dir", default=None)
    fe.add_argument("--adapter-dir", default=None)

    hid = sub.add_parser("hidden", help="Save question and P(True) last-token hidden states")
    hid.add_argument("input_jsonl")
    hid.add_argument("feats_jsonl")
    hid.add_argument("output_npz")
    hid.add_argument("--model-dir", default=None)
    hid.add_argument("--adapter-dir", default=None)
    hid.add_argument("--bs", type=int, default=16)
    hid.add_argument("--layers", type=int, nargs="+", default=None)

    fit = sub.add_parser("fit", help="Fit the Exp3 stacker and write exp3_model.joblib")
    fit.add_argument("--train", required=True)
    fit.add_argument("--dev", required=True)
    fit.add_argument("--ood", required=True)
    fit.add_argument("--train-hid", required=True)
    fit.add_argument("--dev-hid", required=True)
    fit.add_argument("--ood-hid", required=True)
    fit.add_argument("--C", type=float, default=0.5)
    fit.add_argument("--max-risk", type=float, default=0.15, help="safety margin below 0.2")
    fit.add_argument("--min-n", type=int, default=20, help="min answered rows in train-oof")
    fit.add_argument("--out", default="../predictions/Q3/exp3_model.joblib")

    args = ap.parse_args()
    if args.cmd == "features":
        extract_features(args)
    elif args.cmd == "hidden":
        extract_hidden(args)
    else:
        fit_model(args)


if __name__ == "__main__":
    main()
