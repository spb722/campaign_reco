from __future__ import annotations

import csv

from app.schemas.objective import ParsedObjective
from app.schemas.segment import RulebookMatch, Segment
from app.tools.data_paths import data_dir
from app.tools.rulebook_tool import get_large_rulebook_candidates


def _to_segment(row: dict[str, str]) -> Segment:
    return Segment(
        segment_id=row["segment_id"],
        segment_name=row["segment_name"],
        rfm_segment=row["rfm_segment"],
        data_usage_segment=row["data_usage_segment"],
        voice_usage_segment=row["voice_usage_segment"],
        data_usage_trend=row["data_usage_trend"],
        voice_usage_trend=row["voice_usage_trend"],
        customer_count=int(row["customer_count"]),
        avg_arpu=float(row["avg_arpu"]),
        avg_data_gb=float(row["avg_data_gb"]),
        avg_voice_min=float(row["avg_voice_min"]),
        recharge_frequency_days=int(row["recharge_frequency_days"]),
        churn_risk_score=float(row["churn_risk_score"]),
        activity_score=float(row["activity_score"]),
        inactive_days=int(row["inactive_days"]),
        current_pack_type=row["current_pack_type"],
        offer_affinity=row["offer_affinity"],
        business_context=row["business_context"],
        campaign_family=row.get("campaign_family") or None,
        campaign_intent=row.get("campaign_intent") or None,
        customer_signal=row.get("customer_signal") or None,
        customer_meaning=row.get("customer_meaning") or None,
        opportunity=row.get("opportunity") or None,
        nbo_action=row.get("nbo_action") or None,
    )


def load_mock_segments() -> list[Segment]:
    path = data_dir() / "mock_segments.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return [_to_segment(row) for row in csv.DictReader(handle)]


def _large_rulebook_segment(row: dict[str, str], index: int, parsed_objective: ParsedObjective) -> Segment:
    data_usage = (row.get("data_usage_segment") or "medium").lower()
    voice_usage = (row.get("voice_usage_segment") or "medium").lower()
    data_gb = _usage_gb(data_usage)
    voice_min = _voice_minutes(voice_usage)
    avg_arpu = _estimated_arpu(row.get("RFM_Segment", ""), data_gb, voice_min)
    count = _row_customer_count(row) or _fallback_customer_count(row, index)
    opportunity = row.get("Opportunity", "")
    nbo_action = row.get("NBO_Action", "")
    return Segment(
        segment_id=f"RB{index + 1:03d}",
        segment_name=_segment_name(row, index),
        rfm_segment=row.get("RFM_Segment") or "no_rfm",
        data_usage_segment=data_usage,
        voice_usage_segment=voice_usage,
        data_usage_trend=row.get("data_usage_trend") or "no_trend",
        voice_usage_trend=row.get("voice_usage_trend") or "no_trend",
        customer_count=count,
        avg_arpu=avg_arpu,
        avg_data_gb=data_gb,
        avg_voice_min=voice_min,
        recharge_frequency_days=_recharge_frequency(row),
        churn_risk_score=_churn_risk(row),
        activity_score=_activity_score(row),
        inactive_days=_inactive_days(row),
        current_pack_type=_pack_type(avg_arpu),
        offer_affinity=_offer_affinity(opportunity, nbo_action, parsed_objective.campaign_intent),
        business_context=parsed_objective.business_context or "prepaid",
        customer_signal=row.get("Customer Signal") or None,
        customer_meaning=row.get("Customer Meaning") or None,
        opportunity=opportunity or None,
        nbo_action=nbo_action or None,
    )


def get_segment_candidates(
    rulebook_matches: list[RulebookMatch], parsed_objective: ParsedObjective
) -> list[Segment]:
    large_rows = get_large_rulebook_candidates(parsed_objective, limit=5)
    allowed_trends = {match.trend for match in rulebook_matches}
    allowed_trends.update(row.get("data_usage_trend", "") for row in large_rows)
    allowed_trends.update(row.get("voice_usage_trend", "") for row in large_rows)
    allowed_actions = _allowed_actions(rulebook_matches, large_rows)
    prompt_hint = (parsed_objective.target_segment_hint or "").replace("_", " ").lower()
    candidates = _filter_segments(parsed_objective, allowed_trends, allowed_actions, prompt_hint)
    if not candidates and prompt_hint:
        candidates = _filter_segments(parsed_objective, allowed_trends, allowed_actions, "")

    ranked = sorted(candidates, key=lambda segment: _segment_rank(segment, allowed_actions), reverse=True)
    return ranked[:5]


def _filter_segments(
    parsed_objective: ParsedObjective,
    allowed_trends: set[str],
    allowed_actions: set[str],
    prompt_hint: str,
) -> list[Segment]:
    candidates = []
    for segment in load_mock_segments():
        if segment.business_context != parsed_objective.business_context:
            continue
        if parsed_objective.campaign_intent != "recommend_best_campaign":
            segment_intent = (segment.campaign_intent or "").strip()
            if segment_intent and segment_intent != parsed_objective.campaign_intent:
                continue
        trend_match = (
            segment.data_usage_trend in allowed_trends or segment.voice_usage_trend in allowed_trends
        )
        action_match = _matches_action(segment, allowed_actions)
        if not trend_match and not action_match:
            continue
        if prompt_hint and prompt_hint not in segment.segment_name.lower() and prompt_hint not in segment.rfm_segment.lower():
            if prompt_hint in {"mid arpu", "mid"} and 250 <= segment.avg_arpu <= 500:
                pass
            else:
                continue
        candidates.append(segment)
    return candidates


def _allowed_actions(rulebook_matches: list[RulebookMatch], large_rows: list[dict[str, str]]) -> set[str]:
    actions: set[str] = set()
    for match in rulebook_matches:
        actions.add(_norm(match.typical_action))
        actions.update(_norm(action) for action in match.allowed_action_families)
    for row in large_rows:
        actions.add(_norm(row.get("Opportunity", "")))
        actions.add(_norm(row.get("NBO_Action", "")))
    return {action for action in actions if action}


def _matches_action(segment: Segment, allowed_actions: set[str]) -> bool:
    if not allowed_actions:
        return False
    values = [
        segment.opportunity or "",
        segment.nbo_action or "",
        segment.campaign_family or "",
        segment.offer_affinity or "",
    ]
    searchable = " ".join(_norm(value) for value in values)
    return any(action and (action in searchable or searchable in action) for action in allowed_actions)


def _segment_rank(segment: Segment, allowed_actions: set[str]) -> tuple[int, float, int]:
    action_score = 1 if _matches_action(segment, allowed_actions) else 0
    confidence_proxy = (1 - segment.churn_risk_score) + segment.activity_score
    return (action_score, confidence_proxy, segment.customer_count)


def _norm(value: str) -> str:
    return value.lower().replace("_", " ").replace("-", " ").strip()


def _segment_name(row: dict[str, str], index: int) -> str:
    base = row.get("segment") or row.get("Customer Signal") or f"Rulebook Segment {index + 1}"
    usage = row.get("data_usage_segment") or row.get("voice_usage_segment")
    if usage and usage.lower() not in base.lower():
        return f"{base} - {usage.title()} Usage"
    return base


def _row_customer_count(row: dict[str, str]) -> int:
    for key in ("msisdn", "MSISDN", "Subs", "subs", "customer_count", "Customer Count"):
        value = (row.get(key) or "").replace(",", "").strip()
        if value.isdigit():
            return int(value)
    return 0


def _fallback_customer_count(row: dict[str, str], index: int) -> int:
    # The first uploaded full rulebook has blank count fields. Keep demo counts stable
    # until the population count extract is dropped into the same file.
    base_by_usage = {"very_high": 58_000, "high": 46_000, "medium": 33_000, "low": 22_000, "zero": 12_000}
    data_count = base_by_usage.get((row.get("data_usage_segment") or "").lower(), 24_000)
    voice_count = base_by_usage.get((row.get("voice_usage_segment") or "").lower(), 24_000)
    return max(8_000, int((data_count + voice_count) / 2) - index * 1_500)


def _usage_gb(value: str) -> float:
    return {"zero": 0.1, "low": 3.0, "medium": 9.0, "high": 22.0, "very_high": 38.0}.get(value, 8.0)


def _voice_minutes(value: str) -> float:
    return {"zero": 0.0, "low": 45.0, "medium": 140.0, "high": 320.0, "very_high": 520.0}.get(value, 120.0)


def _estimated_arpu(rfm_segment: str, data_gb: float, voice_min: float) -> float:
    rfm_adjustment = {
        "Champions": 160,
        "Loyal Customers": 120,
        "Potential Loyalist": 80,
        "Promising Customers": 50,
        "Recent Customers": 30,
        "About to Sleep": -20,
        "At Risk": -35,
        "Hibernating": -70,
        "Lost": -90,
    }.get(rfm_segment, 0)
    return max(60.0, round(130 + data_gb * 8 + voice_min * 0.35 + rfm_adjustment, 2))


def _recharge_frequency(row: dict[str, str]) -> int:
    trend_text = f"{row.get('data_usage_trend', '')} {row.get('voice_usage_trend', '')}".lower()
    if "dormant" in trend_text:
        return 55
    if "declining" in trend_text:
        return 38
    if "rapid expansion" in trend_text or "strong growth" in trend_text:
        return 24
    return 30


def _churn_risk(row: dict[str, str]) -> float:
    text = " ".join(row.values()).lower()
    score = 0.28
    if "declining" in text or "retention" in text or "churn" in text:
        score += 0.34
    if "dormant" in text or "hibernating" in text or "lost" in text:
        score += 0.2
    if "growth" in text or "champion" in text:
        score -= 0.12
    return round(max(0.05, min(0.9, score)), 2)


def _activity_score(row: dict[str, str]) -> float:
    text = " ".join(row.values()).lower()
    score = 0.55
    if "rapid expansion" in text or "strong growth" in text:
        score += 0.22
    if "gradual growth" in text:
        score += 0.12
    if "declining" in text:
        score -= 0.22
    if "dormant" in text or "zero" in text:
        score -= 0.28
    return round(max(0.05, min(0.95, score)), 2)


def _inactive_days(row: dict[str, str]) -> int:
    text = " ".join(row.values()).lower()
    if "dormant" in text or "lost" in text or "hibernating" in text:
        return 45
    if "declining" in text or "about to sleep" in text:
        return 18
    return 0


def _pack_type(avg_arpu: float) -> str:
    if avg_arpu >= 450:
        return "premium_monthly"
    if avg_arpu >= 250:
        return "monthly"
    return "low_value"


def _offer_affinity(opportunity: str, nbo_action: str, intent: str) -> str:
    text = f"{opportunity} {nbo_action}".lower()
    if "retention" in text or "rescue" in text or intent == "reduce_churn":
        return "retention_discount"
    if "reactivation" in text or "win-back" in text or intent == "reactivate_inactive":
        return "reactivation_bonus"
    if "cross" in text or intent == "cross_sell":
        return "combo_pack"
    if "premium" in text or intent == "upsell":
        return "ott_bundle"
    if "activity" in text or "nurture" in text or intent == "increase_activity":
        return "weekday_booster"
    if "data" in text or intent in {"increase_arpu", "increase_data_usage"}:
        return "data_addon"
    return "data_addon"
