from __future__ import annotations

import csv

from app.schemas.objective import ParsedObjective
from app.schemas.segment import Offer, Segment
from app.tools.data_paths import data_dir


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def _norm(value: str) -> str:
    return value.lower().replace("'", "").replace("_", " ").strip()


def load_offer_catalog() -> list[Offer]:
    path = data_dir() / "offer_catalog.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        offers = []
        for row in csv.DictReader(handle):
            offers.append(_offer_from_row(row))
        return offers


def get_offer_candidates(segments: list[Segment], parsed_objective: ParsedObjective) -> dict[str, list[Offer]]:
    offers = load_offer_catalog()
    by_segment: dict[str, list[Offer]] = {}
    for segment in segments:
        matching = []
        for offer in offers:
            intent_match = offer.campaign_intent == parsed_objective.campaign_intent
            if parsed_objective.campaign_intent == "recommend_best_campaign":
                intent_match = True
            if not intent_match:
                continue
            if _norm(segment.data_usage_segment) not in {_norm(value) for value in offer.eligible_usage_segment}:
                continue
            eligible_rfm = {_norm(value) for value in offer.eligible_rfm_segment}
            segment_rfm = _norm(segment.rfm_segment)
            if segment_rfm not in eligible_rfm and f"{segment_rfm}s" not in eligible_rfm:
                continue
            matching.append(offer)
        if not matching:
            matching = [offer for offer in offers if offer.campaign_intent == parsed_objective.campaign_intent][:1]
        if not matching:
            matching = [offers[0]]
        by_segment[segment.segment_id] = matching
    return by_segment


def get_next_best_offer_candidates(segment: Segment, parsed_objective: ParsedObjective, current_offer_id: str) -> list[Offer]:
    """Return alternate eligible offers for a segment without changing planner selection rules."""
    path = data_dir() / "offer_catalog.csv"
    strict_matches: list[Offer] = []
    eligible_fallbacks: list[Offer] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["offer_id"] == current_offer_id:
                continue
            if not _segment_is_eligible(segment, row):
                continue
            offer = _offer_from_row(row)
            if _row_matches_intent(row, parsed_objective.campaign_intent):
                strict_matches.append(offer)
            else:
                eligible_fallbacks.append(offer)

    candidates = strict_matches or eligible_fallbacks
    return sorted(candidates, key=lambda offer: _offer_rank_score(offer, parsed_objective.campaign_intent), reverse=True)


def _offer_from_row(row: dict[str, str]) -> Offer:
    return Offer(
        offer_id=row["offer_id"],
        offer_name=row["offer_name"],
        offer_type=row["offer_type"],
        campaign_intent=row["campaign_intent"],
        price=float(row["price"]),
        benefit=row["benefit"],
        validity_days=int(row["validity_days"]),
        eligible_usage_segment=_split(row["eligible_usage_segment"]),
        eligible_rfm_segment=_split(row["eligible_rfm_segment"]),
        estimated_arpu_lift=float(row["estimated_arpu_lift"]),
        estimated_data_lift_gb=float(row["estimated_data_lift_gb"]),
        estimated_save_rate=float(row["estimated_save_rate"]),
        cost_per_user=float(row["cost_per_user"]),
        margin_impact=row["margin_impact"],
        description=row["description"],
    )


def _segment_is_eligible(segment: Segment, row: dict[str, str]) -> bool:
    if _norm(segment.data_usage_segment) not in {_norm(value) for value in _split(row["eligible_usage_segment"])}:
        return False
    eligible_rfm = {_norm(value) for value in _split(row["eligible_rfm_segment"])}
    segment_rfm = _norm(segment.rfm_segment)
    return segment_rfm in eligible_rfm or f"{segment_rfm}s" in eligible_rfm


def _row_matches_intent(row: dict[str, str], campaign_intent: str) -> bool:
    if campaign_intent == "recommend_best_campaign":
        return True
    target_intents = {_norm(value).replace(" ", "_") for value in _split(row.get("target_intents", ""))}
    return row["campaign_intent"] == campaign_intent or campaign_intent in target_intents


def _offer_rank_score(offer: Offer, campaign_intent: str) -> float:
    if campaign_intent in {"increase_data_usage", "increase_activity"}:
        lift = offer.estimated_data_lift_gb
    elif campaign_intent == "reduce_churn":
        lift = offer.estimated_save_rate * 100
    else:
        lift = offer.estimated_arpu_lift
    margin_bonus = 1.0 if offer.margin_impact == "positive" else 0.5 if offer.margin_impact == "neutral" else 0.0
    cost_efficiency = lift / offer.cost_per_user if offer.cost_per_user > 0 else lift
    return (lift * 0.55) + (cost_efficiency * 0.30) + margin_bonus
