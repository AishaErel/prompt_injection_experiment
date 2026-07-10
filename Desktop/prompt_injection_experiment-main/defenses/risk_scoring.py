# The original 9-word list ("ignore, override, respond, output, system,
# assistant, directive, bypass, jailbreak") missed the single most
# attack-indicative word in this whole task — "instructions" — along with
# "previous", "disregard", "forget", "reveal", "prompt", "configuration",
# and others that show up constantly in real injection attempts. Live
# testing showed explicit attacks scoring 0 on this list purely because
# they were reworded around the original 9 words.
#
# This list is deliberately larger and more specific to injection language,
# to reduce false negatives on reworded attacks. It will also raise some
# benign risk scores (e.g. legitimate text mentioning "system" or "rules")
# — that trade-off is intentional and is exactly what RISK_THRESHOLD and
# the soft-rerank penalty exist to manage. Tune RISK_THRESHOLD using the
# app's batch-test threshold sweep rather than by eye.
KEYWORDS = [
    # Original
    "ignore", "override", "respond", "output", "system",
    "assistant", "directive", "bypass", "jailbreak",

    # Instruction-override language
    "instructions", "instruction", "previous", "prior",
    "disregard", "forget", "cancel",

    # Prompt-leak language
    "reveal", "prompt", "configuration", "developer",

    # Unrestricted / jailbreak language
    "unrestricted", "restrictions", "unfiltered", "restriction",

    # Compliance-manipulation language
    "comply", "command", "hidden", "pretend", "roleplay",
]

SYMBOLS = ["@@@", "<<<", ">>>", "[", "]", "###", "|||"]


def risk_score(text: str) -> float:

    text = text.lower()

    score = sum(k in text for k in KEYWORDS)
    score += sum(s in text for s in SYMBOLS)

    return float(score)


def apply_risk_scores(items):

    for item in items:
        item["risk_score"] = risk_score(item["doc"]["text"])

    return items