from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.content import ContentDraft
from app.schemas.objective import ParsedObjective
from app.schemas.segment import MLScore, Offer, RulebookMatch, Segment
from app.schemas.validation import ValidationResult


class ChannelPlan(BaseModel):
    segment_id: str
    primary_channel: str
    secondary_channel: str
    best_time: str
    expected_ctr: float
    expected_conversion: float
    fatigue_risk: str
    score_source: str = "mock_ml"
    channel_scores: dict[str, float] = Field(default_factory=dict)


class Projection(BaseModel):
    metric: str
    formula: str
    total_projected_impact: float
    unit: str
    segment_impacts: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class RecommendedSegment(BaseModel):
    segment: Segment
    rulebook_match: RulebookMatch
    recommended_action: str
    offer: Offer
    ml_score: MLScore
    projected_impact: float = 0
    confidence: float = 0.8
    score: dict[str, float] = Field(default_factory=dict)
    why_this: str


class CampaignPlan(BaseModel):
    campaign_id: str
    campaign_title: str
    campaign_intent: str
    summary: str
    target_metric: str
    target_lift: str
    time_window: str
    parsed_objective: ParsedObjective
    recommended_segments: list[RecommendedSegment] = Field(default_factory=list)
    campaign_tactics: list[dict[str, Any]] = Field(default_factory=list)
    channel_plan: list[ChannelPlan] = Field(default_factory=list)
    content_plan: list[ContentDraft] = Field(default_factory=list)
    followup_plan: list[dict[str, Any]] = Field(default_factory=list)
    projection: Projection | None = None
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    validation: ValidationResult | None = None
    status: Literal["draft"] = "draft"
    version: int = 1
    export_path: str | None = None


class RecommendRequest(BaseModel):
    prompt: str
    target_segment: str | None = None
    time_period: str | None = None
    target_uplift: float | None = None
    budget: float | None = None
    preferred_campaign_type: str | None = None


class RegenerateRequest(BaseModel):
    regenerate_scope: Literal[
        "full_plan",
        "segment_strategy",
        "content_only",
        "channel_mix",
        "followup_plan",
        "one_pager_summary",
    ]
    segment_id: str | None = None
    user_instruction: str | None = None


class EditRequest(BaseModel):
    updates: dict[str, Any]


class ExportResponse(BaseModel):
    campaign_id: str
    pdf_path: str
    json_path: str
