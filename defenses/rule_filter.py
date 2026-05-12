RULE_PATTERNS = [
    "ignore previous instructions",
    "respond only with hacked",
    "override system",
    "you are now",
]


def apply_rule_filter(items):

    for item in items:

        text = item["doc"]["text"].lower()

        item["rule_blocked"] = any(p in text for p in RULE_PATTERNS)

        if item["rule_blocked"]:
            item["stage_removed"] = "rule"

    return items