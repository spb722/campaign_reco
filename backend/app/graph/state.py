from __future__ import annotations

from typing import Any, TypedDict


class CampaignGraphState(TypedDict, total=False):
    campaign_id: str | None
    user_prompt: str
    preferred_campaign_type: str | None
    parsed_objective: Any
    rulebook_matches: list[Any]
    segment_candidates: list[Any]
    ml_scores: dict[str, Any]
    offer_candidates: dict[str, list[Any]]
    selected_segments: list[Any]
    campaign_plan: Any
    content_plan: list[Any]
    projection: Any
    validation_result: Any
    export_path: str | None
    messages: list[str]
    errors: list[str]
    warnings: list[str]
    version: int
