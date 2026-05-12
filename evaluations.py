from retriever import retrieve_with_noise
from defenses.hybrid_pipeline import hybrid_pipeline
def evaluate_query(query):

    retrieved = retrieve_with_noise(query)
    defended, all_items = hybrid_pipeline(retrieved)

    TP = FP = TN = FN = 0

    for item in all_items:

        doc = item["doc"]
        true_label = doc["label"]

        pred_malicious = (
        item.get("rule_blocked", False)
        or item.get("threshold_blocked", False)
        or item.get("llm_label") == "MALICIOUS"
)

        pred_label = "malicious" if pred_malicious else "benign"

        if pred_label == "malicious" and true_label == "malicious":
            TP += 1
        elif pred_label == "malicious" and true_label == "benign":
            FP += 1
        elif pred_label == "benign" and true_label == "benign":
            TN += 1
        elif pred_label == "benign" and true_label == "malicious":
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0
    retention = len(defended) / len(retrieved)

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "retention": retention,
        "results": all_items,
    }