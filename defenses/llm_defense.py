from defenses.ollama import ollama_classify

def apply_llm_classification(items):

    for item in items:

        if item.get("rule_blocked") or item.get("threshold_blocked"):
            continue

        label = ollama_classify(item["doc"]["text"])

        item["llm_label"] = label

        if label == "MALICIOUS":
            item["final_score"] *= 0.2

            if item.get("stage_removed") is None:
                item["stage_removed"] = "llm"

    return sorted(items, key=lambda x: x["final_score"], reverse=True)