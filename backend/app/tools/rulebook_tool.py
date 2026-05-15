from __future__ import annotations

import csv
from functools import lru_cache

from app.schemas.objective import ParsedObjective
from app.schemas.segment import RulebookMatch
from app.tools.data_paths import data_dir


LARGE_RULEBOOK_FILE = "rulebook_msisdn_count.csv"

INTENT_OPPORTUNITY_TERMS: dict[str, tuple[str, ...]] = {
    "increase_arpu": ("upsell", "growth", "nbo", "cross-sell", "user maximization", "premium"),
    "upsell": ("upsell", "growth", "premium", "aggressive upsell", "maximum growth"),
    "cross_sell": ("cross-sell", "combo", "attach", "nbo"),
    "reduce_churn": ("retention", "churn", "rescue", "win-back", "save"),
    "increase_data_usage": ("data", "growth", "upsell", "activation", "nurture", "booster"),
    "increase_activity": ("nurture", "activation", "recovery", "engagement", "growth support"),
    "reactivate_inactive": ("win-back", "reactivation", "activation", "dormant", "nbo"),
    "recommend_best_campaign": (),
}

INTENT_ACTION_FAMILIES: dict[str, list[str]] = {
    "increase_arpu": ["upsell", "cross_sell", "bundle"],
    "upsell": ["upsell", "premium_bundle"],
    "cross_sell": ["cross_sell", "bundle"],
    "reduce_churn": ["retention", "save", "winback"],
    "increase_data_usage": ["upsell", "cross_sell", "activity_booster"],
    "increase_activity": ["nurture", "reactivation", "activity_booster"],
    "reactivate_inactive": ["reactivation", "winback", "light_nbo"],
    "recommend_best_campaign": ["best_next_action"],
}


def load_rulebook() -> list[RulebookMatch]:
    path = data_dir() / "rulebook.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return [
            RulebookMatch(
                trend=row["trend"],
                trend_meaning=row["trend_meaning"],
                typical_action=row["typical_action"],
                allowed_action_families=row["allowed_action_families"].split("|"),
                eligible_intents=row["eligible_intents"].split("|"),
                rulebook_fit_score=float(row["rulebook_fit_score"]),
            )
            for row in rows
        ]


@lru_cache(maxsize=1)
def load_large_rulebook_rows() -> list[dict[str, str]]:
    path = data_dir() / LARGE_RULEBOOK_FILE
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = []
        for row in csv.DictReader(handle):
            cleaned = {(key or "").strip(): (value or "").strip() for key, value in row.items()}
            if cleaned.get("RFM_Segment") and cleaned.get("data_usage_trend"):
                rows.append(cleaned)
        return rows


def large_rulebook_available() -> bool:
    return bool(load_large_rulebook_rows())


def score_large_rulebook_row(row: dict[str, str], parsed_objective: ParsedObjective) -> float:
    intent = parsed_objective.campaign_intent
    terms = INTENT_OPPORTUNITY_TERMS.get(intent, ())
    searchable = " ".join(
        [
            row.get("Opportunity", ""),
            row.get("NBO_Action", ""),
            row.get("segment", ""),
            row.get("Customer Signal", ""),
            row.get("Customer Meaning", ""),
            row.get("data_usage_trend", ""),
            row.get("voice_usage_trend", ""),
            row.get("data_usage_segment", ""),
            row.get("voice_usage_segment", ""),
            row.get("RFM_Segment", ""),
        ]
    ).lower()
    score = 0.0
    if intent == "recommend_best_campaign":
        score += 0.4
    else:
        score += sum(0.35 for term in terms if term in searchable)

    growth_trends = {"gradual growth", "strong growth", "rapid expansion", "early recovery"}
    risk_trends = {"declining", "dormant", "no_trend"}
    row_trends = {row.get("data_usage_trend", "").lower(), row.get("voice_usage_trend", "").lower()}
    if intent in {"increase_arpu", "increase_data_usage", "upsell", "cross_sell"}:
        score += 0.25 * len(row_trends & growth_trends)
        if "gradual growth" in row_trends:
            score += 0.2
    if intent in {"reduce_churn", "reactivate_inactive", "increase_activity"}:
        score += 0.25 * len(row_trends & risk_trends)

    hint = (parsed_objective.target_segment_hint or "").replace("_", " ").lower()
    if hint and hint in searchable:
        score += 0.3
    if hint in {"mid arpu", "mid"} and (
        row.get("data_usage_segment", "").lower() == "medium"
        or row.get("voice_usage_segment", "").lower() == "medium"
    ):
        score += 0.2

    score += min(_row_customer_count(row) / 100_000, 0.2)
    return round(score, 4)


def get_large_rulebook_candidates(parsed_objective: ParsedObjective, limit: int = 5) -> list[dict[str, str]]:
    rows = load_large_rulebook_rows()
    if not rows:
        return []
    scored = [(score_large_rulebook_row(row, parsed_objective), row) for row in rows]
    scored = [item for item in scored if item[0] > 0 or parsed_objective.campaign_intent == "recommend_best_campaign"]
    if not scored:
        scored = [(0.01, row) for row in rows]
    scored.sort(key=lambda item: (item[0], _row_customer_count(item[1])), reverse=True)
    return [row for _, row in scored[:limit]]


def rulebook_match_from_large_row(row: dict[str, str], parsed_objective: ParsedObjective) -> RulebookMatch:
    trend = row.get("data_usage_trend") or row.get("voice_usage_trend") or "no_trend"
    meaning = row.get("Customer Meaning") or row.get("Customer Signal") or "Matched from population rulebook."
    action = row.get("Opportunity") or row.get("NBO_Action") or "Best next action"
    return RulebookMatch(
        trend=trend,
        trend_meaning=meaning,
        typical_action=action,
        allowed_action_families=INTENT_ACTION_FAMILIES.get(parsed_objective.campaign_intent, ["best_next_action"]),
        eligible_intents=[parsed_objective.campaign_intent],
        rulebook_fit_score=max(0.65, min(0.96, score_large_rulebook_row(row, parsed_objective))),
    )


def get_rulebook_matches(parsed_objective: ParsedObjective) -> list[RulebookMatch]:
    large_matches = [
        rulebook_match_from_large_row(row, parsed_objective)
        for row in get_large_rulebook_candidates(parsed_objective, limit=5)
    ]
    if large_matches:
        return large_matches

    matches = [
        rule
        for rule in load_rulebook()
        if parsed_objective.campaign_intent in rule.eligible_intents
        or parsed_objective.campaign_intent == "recommend_best_campaign"
    ]
    if matches:
        return matches
    return [rule for rule in load_rulebook() if "gentle_nbo" in rule.allowed_action_families]


def rulebook_summary() -> dict:
    rules = load_rulebook()
    large_rows = load_large_rulebook_rows()
    return {
        "rfm_categories_supported_in_seed_data": [
            "Champions",
            "Loyal Customers",
            "Potential Loyalists",
            "At Risk",
            "About to Sleep",
            "Hibernating",
            "Lost",
        ],
        "trend_dimensions": [rule.trend for rule in rules],
        "large_rulebook_available": bool(large_rows),
        "large_rulebook_rows_loaded": len(large_rows),
        "total_combinations_reference": 14700,
        "large_rulebook_dimensions": {
            "rfm_segments": sorted({row.get("RFM_Segment", "") for row in large_rows if row.get("RFM_Segment")}),
            "data_usage_segments": sorted({row.get("data_usage_segment", "") for row in large_rows if row.get("data_usage_segment")}),
            "voice_usage_segments": sorted({row.get("voice_usage_segment", "") for row in large_rows if row.get("voice_usage_segment")}),
            "data_usage_trends": sorted({row.get("data_usage_trend", "") for row in large_rows if row.get("data_usage_trend")}),
            "voice_usage_trends": sorted({row.get("voice_usage_trend", "") for row in large_rows if row.get("voice_usage_trend")}),
        },
        "actions_by_trend": {
            rule.trend: {
                "meaning": rule.trend_meaning,
                "typical_action": rule.typical_action,
                "allowed_action_families": rule.allowed_action_families,
            }
            for rule in rules
        },
    }


def _row_customer_count(row: dict[str, str]) -> int:
    for key in ("msisdn", "MSISDN", "Subs", "subs", "customer_count", "Customer Count"):
        value = (row.get(key) or "").replace(",", "").strip()
        if value.isdigit():
            return int(value)
    return 0
