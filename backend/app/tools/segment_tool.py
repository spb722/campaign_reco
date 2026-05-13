from __future__ import annotations

import csv

from app.schemas.objective import ParsedObjective
from app.schemas.segment import RulebookMatch, Segment
from app.tools.data_paths import data_dir


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
    )


def load_mock_segments() -> list[Segment]:
    path = data_dir() / "mock_segments.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return [_to_segment(row) for row in csv.DictReader(handle)]


def get_segment_candidates(
    rulebook_matches: list[RulebookMatch], parsed_objective: ParsedObjective
) -> list[Segment]:
    allowed_trends = {match.trend for match in rulebook_matches}
    prompt_hint = (parsed_objective.target_segment_hint or "").replace("_", " ").lower()
    candidates = []
    for segment in load_mock_segments():
        if segment.business_context != parsed_objective.business_context:
            continue
        trend_match = (
            segment.data_usage_trend in allowed_trends or segment.voice_usage_trend in allowed_trends
        )
        if not trend_match:
            continue
        if prompt_hint and prompt_hint not in segment.segment_name.lower() and prompt_hint not in segment.rfm_segment.lower():
            if prompt_hint in {"mid arpu", "mid"} and 250 <= segment.avg_arpu <= 500:
                pass
            else:
                continue
        candidates.append(segment)

    if parsed_objective.campaign_intent == "recommend_best_campaign":
        return candidates[:5]
    return candidates[:5]
