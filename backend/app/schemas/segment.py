from __future__ import annotations

from pydantic import BaseModel, Field


class Segment(BaseModel):
    segment_id: str
    segment_name: str
    rfm_segment: str
    data_usage_segment: str
    voice_usage_segment: str
    data_usage_trend: str
    voice_usage_trend: str
    customer_count: int
    avg_arpu: float
    avg_data_gb: float
    avg_voice_min: float
    recharge_frequency_days: int
    churn_risk_score: float
    activity_score: float
    inactive_days: int
    current_pack_type: str
    offer_affinity: str
    business_context: str


class RulebookMatch(BaseModel):
    trend: str
    trend_meaning: str
    typical_action: str
    allowed_action_families: list[str] = Field(default_factory=list)
    eligible_intents: list[str] = Field(default_factory=list)
    rulebook_fit_score: float = 0.8


class MLScore(BaseModel):
    segment_id: str
    channel_scores: dict[str, float]
    best_channel: str
    secondary_channel: str
    best_time_window: str
    expected_ctr: float
    expected_conversion: float
    fatigue_risk: str
    offer_affinity: str
    model_confidence: float
    fallback_used: bool = False
    fallback_reason: str | None = None


class Offer(BaseModel):
    offer_id: str
    offer_name: str
    offer_type: str
    campaign_intent: str
    price: float
    benefit: str
    validity_days: int
    eligible_usage_segment: list[str]
    eligible_rfm_segment: list[str]
    estimated_arpu_lift: float
    estimated_data_lift_gb: float
    estimated_save_rate: float
    cost_per_user: float
    margin_impact: str
    description: str
