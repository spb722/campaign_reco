from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


CampaignIntent = Literal[
    "increase_arpu",
    "reduce_churn",
    "increase_data_usage",
    "upsell",
    "cross_sell",
    "increase_activity",
    "reactivate_inactive",
    "recommend_best_campaign",
]


class ParsedObjective(BaseModel):
    campaign_id: str
    raw_user_prompt: str
    campaign_intent: CampaignIntent
    target_segment_hint: str | None = None
    target_metric: str
    target_lift_value: float | None = None
    target_lift_unit: str | None = None
    time_window_value: int | None = None
    time_window_unit: str | None = None
    business_context: str = "prepaid"
    constraints: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    needs_user_clarification: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    alternative_intents: list[CampaignIntent] = Field(default_factory=list)


class ParseRequest(BaseModel):
    prompt: str
    target_segment: str | None = None
    time_period: str | None = None
    target_uplift: float | None = None
    budget: float | None = None
    preferred_campaign_type: CampaignIntent | None = None


class APIResponse(BaseModel):
    success: bool = True
    data: dict | list | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    request_id: str
