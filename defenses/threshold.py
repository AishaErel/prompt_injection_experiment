from config import RISK_THRESHOLD

def apply_threshold_filter(items):

    for item in items:

        if item.get("risk_score", 0) >= RISK_THRESHOLD:

            item["threshold_blocked"] = True

            if item.get("stage_removed") is None:
                item["stage_removed"] = "threshold"

    return items