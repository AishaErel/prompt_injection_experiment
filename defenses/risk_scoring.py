def risk_score(text: str) -> float:

    text = text.lower()

    keywords = [
        "ignore", "override", "respond", "output",
        "system", "assistant", "directive",
        "bypass", "jailbreak"
    ]

    symbols = ["@@@", "<<<", ">>>", "[", "]"]

    score = sum(k in text for k in keywords)
    score += sum(s in text for s in symbols)

    return float(score)


def apply_risk_scores(items):

    for item in items:
        item["risk_score"] = risk_score(item["doc"]["text"])

    return items