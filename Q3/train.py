"""Train the Q3 confidence model.

features: greedy M-inst answers plus P(True) and white-box token features.
fit: scaled logistic regression on the train labels, abstain threshold from
out-of-fold train predictions. Writes the joblib artifact infer.py loads.

Usage:
  python3 train.py features ../data/public/q3_qa_train.jsonl ../predictions/Q3/train.jsonl
  python3 train.py features ../data/public/q3_qa_dev.jsonl ../predictions/Q3/dev.jsonl
  python3 train.py features ../data/public/q3_qa_dev_ood.jsonl ../predictions/Q3/ood.jsonl
  python3 train.py fit --train ../predictions/Q3/train.jsonl --dev ../predictions/Q3/dev.jsonl --ood ../predictions/Q3/ood.jsonl --out ../predictions/Q3/exp2_model.joblib
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval import auroc, ece, is_correct, selective_score
from infer import extract, load_model, read_jsonl

ALL = ["ptrue_logit", "mean_logprob", "min_logprob", "first_logprob", "first_entropy",
       "mean_entropy", "log_len", "eos_logprob", "truncated"]
NO_PTRUE = [f for f in ALL if f != "ptrue_logit"]


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

    model, cols = models["white-box + ptrue"]
    cdv = model.predict_proba(mat(Xdv, cols))[:, 1]
    cod = model.predict_proba(mat(Xod, cols))[:, 1]

    # Threshold is chosen on 5-fold out-of-fold predictions over TRAIN, so dev and ood stay clean.
    oof = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=1000)),
        mat(Xtr, cols), ytr,
        cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba",
    )[:, 1]
    print("\n[3a] Out-of-fold train metrics (honest estimate on train)")
    show("train-oof", metrics(oof, ytr))

    tau, feasible = choose_threshold(oof, ytr, args.max_risk, args.min_n)
    note = "" if feasible else "  (NO safe slice found; fallback = top-%d rows)" % args.min_n
    print(f"\n[3b] Abstain threshold from train-oof (max risk {args.max_risk}, "
          f"min answered {args.min_n}): tau={tau:.4f}{note}")
    print("  mean confidence vs accuracy -> dev: %.3f vs %.3f | ood: %.3f vs %.3f"
          % (cdv.mean(), ydv.mean(), cod.mean(), yod.mean()))
    for split, c, y in [("dev", cdv, ydv), ("ood", cod, yod)]:
        m = metrics(c, y, abstain=c < tau)
        m["set_score"] = (m["AUROC"] + (1 - m["ECE"]) + m["selective"]) / 3
        show(split, m)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": cols, "threshold": tau}, out)
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser(description="Train the Q3 confidence model.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    fe = sub.add_parser("features", help="Extract answer, features, and correctness labels")
    fe.add_argument("input_jsonl")
    fe.add_argument("output_jsonl")
    fe.add_argument("--model-dir", default=None)
    fe.add_argument("--adapter-dir", default=None)

    fit = sub.add_parser("fit", help="Fit the confidence model and write exp2_model.joblib")
    fit.add_argument("--train", required=True)
    fit.add_argument("--dev", required=True)
    fit.add_argument("--ood", required=True)
    fit.add_argument("--C", type=float, default=0.5)
    fit.add_argument("--max-risk", type=float, default=0.15, help="safety margin below 0.2")
    fit.add_argument("--min-n", type=int, default=20, help="min answered rows in train-oof")
    fit.add_argument("--out", default="../predictions/Q3/exp2_model.joblib")

    args = ap.parse_args()
    if args.cmd == "features":
        extract_features(args)
    else:
        fit_model(args)


if __name__ == "__main__":
    main()
