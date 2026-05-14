from retriever import retrieve_with_noise
from defenses.hybrid_pipeline import hybrid_pipeline


def safe_divide(a, b):
    return a / b if b else 0


def evaluate_query(query):
    retrieved = retrieve_with_noise(query)
    defended, all_items = hybrid_pipeline(retrieved)

    TP = FP = TN = FN = 0

    malicious_before = 0
    malicious_after = 0
    benign_before = 0
    benign_after = 0

    for item in all_items:
        doc = item["doc"]
        true_label = doc["label"]

        if true_label == "malicious":
            malicious_before += 1
        else:
            benign_before += 1

        removed = item.get("stage_removed") is not None

        if not removed:
            if true_label == "malicious":
                malicious_after += 1
            else:
                benign_after += 1

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

    precision = safe_divide(TP, TP + FP)
    recall = safe_divide(TP, TP + FN)
    accuracy = safe_divide(TP + TN, TP + TN + FP + FN)

    retention = safe_divide(len(defended), len(retrieved))
    malicious_removal_rate = safe_divide(
        malicious_before - malicious_after,
        malicious_before
    )
    benign_preservation_rate = safe_divide(
        benign_after,
        benign_before
    )

    contamination_before = malicious_before > 0
    contamination_after = malicious_after > 0
    full_elimination = malicious_after == 0

    return {
        "query": query,

        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,

        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "retention": retention,

        "malicious_before": malicious_before,
        "malicious_after": malicious_after,
        "benign_before": benign_before,
        "benign_after": benign_after,

        "malicious_removal_rate": malicious_removal_rate,
        "benign_preservation_rate": benign_preservation_rate,
        "contamination_before": contamination_before,
        "contamination_after": contamination_after,
        "full_elimination": full_elimination,

        "results": all_items,
    }
