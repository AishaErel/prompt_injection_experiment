import csv
import os
from datetime import datetime

def save_results(query, result):

    os.makedirs("results", exist_ok=True)

    filename = f"results/{query[:30]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "query", "source", "label",
            "retrieval", "risk", "final",
            "rule", "threshold", "llm", "stage"
        ])

        for item in result["results"]:

            d = item["doc"]

            writer.writerow([
                query,
                d["source"],
                d["label"],
                item["retrieval_score"],
                item.get("risk_score", 0),
                item.get("final_score", 0),
                item.get("rule_blocked"),
                item.get("threshold_blocked"),
                item.get("llm_label"),
                item.get("stage_removed"),
            ])