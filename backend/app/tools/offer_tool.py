from __future__ import annotations

import csv

from app.schemas.objective import ParsedObjective
from app.schemas.segment import Offer, Segment
from app.tools.data_paths import data_dir


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def load_offer_catalog() -> list[Offer]:
    path = data_dir() / "offer_catalog.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        offers = []
        for row in csv.DictReader(handle):
            offers.append(
                Offer(
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
            )
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
            if segment.data_usage_segment not in offer.eligible_usage_segment:
                continue
            if segment.rfm_segment not in offer.eligible_rfm_segment:
                continue
            matching.append(offer)
        if not matching:
            matching = [offer for offer in offers if offer.campaign_intent == parsed_objective.campaign_intent][:1]
        if not matching:
            matching = [offers[0]]
        by_segment[segment.segment_id] = matching
    return by_segment
