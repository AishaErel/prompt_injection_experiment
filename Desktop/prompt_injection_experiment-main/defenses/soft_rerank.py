from config import SOFT_ALPHA

def apply_soft_rerank(items, alpha=SOFT_ALPHA):

    for item in items:

        risk = item.get("risk_score", 0.0)
        penalty = 1 / (1 + alpha * risk)

        item["final_score"] = item.get("retrieval_score", 0.0) * penalty

    return sorted(items, key=lambda x: x["final_score"], reverse=True)