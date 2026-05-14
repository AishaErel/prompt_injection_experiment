import csv
import os
from datetime import datetime


def safe_filename(text):
    cleaned = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in text)
    return cleaned[:40]


def save_query_details(query, result):
    os.makedirs("results", exist_ok=True)

    filename = f"results/details_{safe_filename(query)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "query",
            "source",
            "true_label",
            "attack_type",
            "retrieval_score",
            "risk_score",
            "final_score",
            "rule_blocked",
            "threshold_blocked",
            "llm_label",
            "stage_removed",
        ])

        for item in result["results"]:
            doc = item["doc"]

            writer.writerow([
                query,
                doc["source"],
                doc["label"],
                doc.get("attack_type"),
                item.get("retrieval_score", 0),
                item.get("risk_score", 0),
                item.get("final_score", 0),
                item.get("rule_blocked", False),
                item.get("threshold_blocked", False),
                item.get("llm_label"),
                item.get("stage_removed"),
            ])

    print(f"Saved detailed query results to {filename}")


def save_summary(results, filename="results/summary_results.csv"):
    os.makedirs("results", exist_ok=True)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "query",
            "TP",
            "FP",
            "TN",
            "FN",
            "precision",
            "recall",
            "accuracy",
            "retention",
            "malicious_before",
            "malicious_after",
            "benign_before",
            "benign_after",
            "malicious_removal_rate",
            "benign_preservation_rate",
            "contamination_before",
            "contamination_after",
            "full_elimination",
        ])

        for r in results:
            writer.writerow([
                r["query"],
                r["TP"],
                r["FP"],
                r["TN"],
                r["FN"],
                round(r["precision"], 4),
                round(r["recall"], 4),
                round(r["accuracy"], 4),
                round(r["retention"], 4),
                r["malicious_before"],
                r["malicious_after"],
                r["benign_before"],
                r["benign_after"],
                round(r["malicious_removal_rate"], 4),
                round(r["benign_preservation_rate"], 4),
                r["contamination_before"],
                r["contamination_after"],
                r["full_elimination"],
            ])

    print(f"Saved summary results to {filename}")


def save_aggregate(results, filename="results/aggregate_results.csv"):
    os.makedirs("results", exist_ok=True)

    total_TP = sum(r["TP"] for r in results)
    total_FP = sum(r["FP"] for r in results)
    total_TN = sum(r["TN"] for r in results)
    total_FN = sum(r["FN"] for r in results)

    total_malicious_before = sum(r["malicious_before"] for r in results)
    total_malicious_after = sum(r["malicious_after"] for r in results)
    total_benign_before = sum(r["benign_before"] for r in results)
    total_benign_after = sum(r["benign_after"] for r in results)

    n = len(results)

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0
    accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN) if (total_TP + total_TN + total_FP + total_FN) else 0

    malicious_removal_rate = (
        (total_malicious_before - total_malicious_after) / total_malicious_before
        if total_malicious_before else 0
    )

    benign_preservation_rate = (
        total_benign_after / total_benign_before
        if total_benign_before else 0
    )

    contamination_before_rate = (
        sum(1 for r in results if r["contamination_before"]) / n
        if n else 0
    )

    contamination_after_rate = (
        sum(1 for r in results if r["contamination_after"]) / n
        if n else 0
    )

    full_elimination_rate = (
        sum(1 for r in results if r["full_elimination"]) / n
        if n else 0
    )

    average_retention = sum(r["retention"] for r in results) / n if n else 0

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["metric", "value"])
        writer.writerow(["precision", round(precision, 4)])
        writer.writerow(["recall", round(recall, 4)])
        writer.writerow(["accuracy", round(accuracy, 4)])
        writer.writerow(["average_retention", round(average_retention, 4)])
        writer.writerow(["malicious_removal_rate", round(malicious_removal_rate, 4)])
        writer.writerow(["benign_preservation_rate", round(benign_preservation_rate, 4)])
        writer.writerow(["contamination_before_rate", round(contamination_before_rate, 4)])
        writer.writerow(["contamination_after_rate", round(contamination_after_rate, 4)])
        writer.writerow(["full_elimination_rate", round(full_elimination_rate, 4)])

    print(f"Saved aggregate results to {filename}")
