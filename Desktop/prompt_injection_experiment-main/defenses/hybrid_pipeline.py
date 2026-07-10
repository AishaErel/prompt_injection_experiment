from defenses.rule_filter import apply_rule_filter
from defenses.risk_scoring import apply_risk_scores
from defenses.soft_rerank import apply_soft_rerank
from defenses.threshold import apply_threshold_filter
from defenses.llm_defense import apply_llm_classification


def hybrid_pipeline(items):

    items = apply_rule_filter(items)
    items = apply_risk_scores(items)
    items = apply_soft_rerank(items)
    items = apply_threshold_filter(items)
    items = apply_llm_classification(items)

    final = [
        i for i in items
        if not (
            i.get("rule_blocked")
            or i.get("threshold_blocked")
            or i.get("llm_label") == "MALICIOUS"
        )
    ]

    return final, items