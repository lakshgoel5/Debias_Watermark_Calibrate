import argparse
import json
import re
import string


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_correct(answer, gold_answers):
    # Space padding makes the gold answer a contiguous word sequence, not a character substring.
    normalized_answer = f" {normalize(answer)} "
    return any(gold and f" {gold} " in normalized_answer for gold in map(normalize, gold_answers))


def load_predictions(path):
    predictions = {}
    for row in read_jsonl(path):
        for field in ("id", "answer", "confidence", "abstain"):
            if field not in row:
                raise ValueError(f"Prediction row missing {field!r}: {row}")
        item_id = str(row["id"])
        if item_id in predictions:
            raise ValueError(f"Duplicate prediction for id {item_id!r}")
        confidence = float(row["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1] for id {item_id!r}")
        prediction = {
            "answer": str(row["answer"]),
            "confidence": confidence,
            "abstain": bool(row["abstain"]),
        }
        for key in ("confidence_geo", "confidence_min"):
            if key in row:
                prediction[key] = float(row[key])
        predictions[item_id] = prediction
    return predictions


def auroc(correct, confidences):
    positives = [(c, y) for c, y in zip(confidences, correct) if y]
    negatives = [(c, y) for c, y in zip(confidences, correct) if not y]
    if not positives or not negatives:
        return 0.5
    wins = sum(
        1.0 if positive[0] > negative[0] else 0.5 if positive[0] == negative[0] else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def ece(correct, confidences):
    total = len(correct)
    if not total:
        return 0.0
    result = 0.0
    for bin_index in range(10):
        lower, upper = bin_index / 10.0, (bin_index + 1) / 10.0
        indices = [
            i for i, confidence in enumerate(confidences)
            if (confidence >= lower if bin_index == 0 else confidence > lower)
            and confidence <= upper
        ]
        if indices:
            mean_correct = sum(correct[i] for i in indices) / len(indices)
            mean_confidence = sum(confidences[i] for i in indices) / len(indices)
            result += len(indices) / total * abs(mean_correct - mean_confidence)
    return result


def selective_score(correct, confidences, abstains):
    answered = [i for i, abstain in enumerate(abstains) if not abstain]
    coverage = len(answered) / len(correct) if correct else 0.0
    if not answered:
        return 0.0
    risk = sum(1 - correct[i] for i in answered) / len(answered)
    if risk <= 0.2:
        return coverage
    return coverage * max(0.0, 1.0 - (risk - 0.2) / 0.1)


def evaluate(data_path, predictions_path):
    rows = read_jsonl(data_path)
    predictions = load_predictions(predictions_path)
    correct, confidences, abstains = [], [], []
    geo, minimum = [], []
    for row in rows:
        item_id = str(row["id"])
        if item_id not in predictions:
            raise KeyError(f"Missing prediction for id {item_id!r}")
        prediction = predictions[item_id]
        correct.append(int(is_correct(prediction["answer"], row["answers"])))
        confidences.append(prediction["confidence"])
        abstains.append(prediction["abstain"])
        if "confidence_geo" in prediction:
            geo.append(prediction["confidence_geo"])
        if "confidence_min" in prediction:
            minimum.append(prediction["confidence_min"])

    metrics = {
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "AUROC": auroc(correct, confidences),
        "ECE": ece(correct, confidences),
        "selective": selective_score(correct, confidences, abstains),
    }
    ranking = {"arith": metrics["AUROC"]}
    if len(geo) == len(correct):
        ranking["geo"] = auroc(correct, geo)
        metrics["AUROC_geo"] = ranking["geo"]
    if len(minimum) == len(correct):
        ranking["min"] = auroc(correct, minimum)
        metrics["AUROC_min"] = ranking["min"]
    if len(ranking) > 1:
        metrics["best_ranker"] = max(ranking, key=ranking.get)
    metrics["set_score"] = (
        metrics["AUROC"] + (1.0 - metrics["ECE"]) + metrics["selective"]
    ) / 3.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Q3 calibrated QA predictions.")
    parser.add_argument("--dev-data", default="../data/public/q3_qa_dev.jsonl")
    parser.add_argument("--dev-preds", default="../predictions/Q3/dev/output.jsonl")
    parser.add_argument("--ood-data", default="../data/public/q3_qa_dev_ood.jsonl")
    parser.add_argument("--ood-preds", default="../predictions/Q3/ood/output.jsonl")
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--train-preds", default=None)
    args = parser.parse_args()

    def show(name, metrics):
        print(f"{name}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric} = {value:.4f}")
            else:
                print(f"  {metric} = {value}")

    if args.train_data and args.train_preds:
        show("train", evaluate(args.train_data, args.train_preds))
        return

    in_domain = evaluate(args.dev_data, args.dev_preds)
    out_of_domain = evaluate(args.ood_data, args.ood_preds)
    for name, metrics in (("in-domain", in_domain), ("out-of-domain", out_of_domain)):
        show(name, metrics)
    print(f"Q3 score = {(in_domain['set_score'] + out_of_domain['set_score']) / 2:.4f}")


if __name__ == "__main__":
    main()
