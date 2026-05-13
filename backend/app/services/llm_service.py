from __future__ import annotations

import os
import re
from itertools import count
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.content import ContentDraft
from app.schemas.objective import CampaignIntent, ParsedObjective
from app.schemas.segment import Offer, Segment


_campaign_counter = count(1)
_ENV_LOADED = False


INTENT_KEYWORDS = [
    ("reduce_churn", ["churn", "retention", "save"]),
    ("increase_data_usage", ["data consumption", "data usage", "increase data", "gb"]),
    ("reactivate_inactive", ["inactive", "dormant", "sleeping", "reactivate", "winback"]),
    ("increase_arpu", ["arpu", "revenue"]),
    ("upsell", ["upsell", "upgrade", "premium"]),
    ("cross_sell", ["cross-sell", "cross sell", "addon", "add-on"]),
    ("increase_activity", ["activity", "engagement", "usage frequency"]),
    ("recommend_best_campaign", ["best campaign", "best opportunity", "recommend"]),
]


class ObjectiveLLMResult(BaseModel):
    campaign_intent: CampaignIntent
    target_segment_hint: str | None = None
    target_metric: str
    target_lift_value: float | None = None
    target_lift_unit: Literal["percent", "rupees", "absolute", "gb", "customers"] | None = None
    time_window_value: int | None = None
    time_window_unit: Literal["days", "weeks", "months", "quarter"] | None = None
    business_context: str = "prepaid"
    constraints: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.82, ge=0, le=1)
    needs_user_clarification: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    alternative_intents: list[CampaignIntent] = Field(default_factory=list)


class ContentDraftBatch(BaseModel):
    drafts: list[ContentDraft]


class StrategyText(BaseModel):
    campaign_summary: str
    segment_explanations: dict[str, str] = Field(default_factory=dict)


def parse_objective(prompt: str, preferred_campaign_type: str | None = None) -> ParsedObjective:
    """Parse objectives with OpenAI structured output, with deterministic fallback."""
    _load_runtime_env()
    if _llm_enabled():
        try:
            parsed = _llm_parse_objective(prompt, preferred_campaign_type)
            parsed.assumptions.append("Objective parsed by OpenAI structured output.")
            return parsed
        except Exception as exc:
            fallback = _heuristic_parse(prompt, preferred_campaign_type)
            fallback.assumptions.append(f"OpenAI objective parser failed; heuristic fallback used: {exc.__class__.__name__}.")
            return fallback
    return _heuristic_parse(prompt, preferred_campaign_type)


def _llm_parse_objective(prompt: str, preferred_campaign_type: str | None = None) -> ParsedObjective:
    structured_llm = _chat_model("campaign_objective_parser").with_structured_output(ObjectiveLLMResult, method="function_calling")
    system = (
        "You parse telecom prepaid campaign objectives into strict JSON. "
        "Supported campaign_intent values are: increase_arpu, reduce_churn, increase_data_usage, "
        "upsell, cross_sell, increase_activity, reactivate_inactive, recommend_best_campaign. "
        "Map ARPU to increase_arpu; churn to reduce_churn; data consumption or data usage to increase_data_usage; "
        "inactive, dormant, sleeping, or winback to reactivate_inactive. "
        "If values are missing, infer reasonable demo defaults and list them in assumptions. "
        "Do not invent unsupported intent values."
    )
    user = (
        f"User prompt: {prompt}\n"
        f"Preferred campaign type override: {preferred_campaign_type or 'none'}\n"
        "Return only the structured object."
    )
    result: ObjectiveLLMResult = structured_llm.with_config(
        run_name="parse_objective",
        tags=["campaign_mvp", "objective_parser"],
    ).invoke([("system", system), ("user", user)])
    if preferred_campaign_type:
        result.campaign_intent = preferred_campaign_type  # type: ignore[assignment]
    fallback = _heuristic_parse(prompt, preferred_campaign_type)
    time_window_value = result.time_window_value or fallback.time_window_value
    time_window_unit = result.time_window_unit or fallback.time_window_unit
    if result.time_window_unit == "quarter":
        time_window_value = 90 if not result.time_window_value or result.time_window_value <= 4 else result.time_window_value
        time_window_unit = "days"
    return ParsedObjective(
        campaign_id=fallback.campaign_id,
        raw_user_prompt=prompt,
        campaign_intent=result.campaign_intent,
        target_segment_hint=_normalize_segment_hint(result.target_segment_hint),
        target_metric=result.target_metric or fallback.target_metric,
        target_lift_value=result.target_lift_value if result.target_lift_value is not None else fallback.target_lift_value,
        target_lift_unit=result.target_lift_unit or fallback.target_lift_unit,
        time_window_value=time_window_value,
        time_window_unit=time_window_unit,
        business_context=result.business_context or fallback.business_context,
        constraints=result.constraints,
        confidence=result.confidence,
        needs_user_clarification=result.needs_user_clarification,
        clarifying_questions=result.clarifying_questions,
        assumptions=result.assumptions,
        alternative_intents=result.alternative_intents,
    )


def _normalize_segment_hint(value: str | None) -> str | None:
    if not value:
        return None
    text = value.lower().replace("-", " ").replace("_", " ")
    if "mid" in text and "arpu" in text:
        return "mid_arpu"
    if any(token in text for token in ["inactive", "dormant", "sleeping", "winback"]):
        return "inactive"
    if text.strip() in {"prepaid", "prepaid customers", "all prepaid", "customers"}:
        return None
    return text.strip().replace(" ", "_")


def _heuristic_parse(prompt: str, preferred_campaign_type: str | None = None) -> ParsedObjective:
    text = prompt.lower()
    matched: list[str] = []
    for intent, keywords in INTENT_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            matched.append(intent)

    if preferred_campaign_type:
        intent = preferred_campaign_type
    elif matched:
        intent = matched[0]
    else:
        intent = "recommend_best_campaign"

    metric_by_intent = {
        "increase_arpu": "arpu",
        "reduce_churn": "churn",
        "increase_data_usage": "data_usage",
        "upsell": "arpu",
        "cross_sell": "attach_rate",
        "increase_activity": "activity",
        "reactivate_inactive": "reactivation",
        "recommend_best_campaign": "opportunity_score",
    }
    lift_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    days_match = re.search(r"(\d+)\s*(?:day|days)", text)
    quarter = "quarter" in text
    month = "month" in text

    target_segment_hint = None
    if "mid-arpu" in text or "mid arpu" in text:
        target_segment_hint = "mid_arpu"
    elif "inactive" in text or "dormant" in text:
        target_segment_hint = "inactive"

    assumptions: list[str] = []
    time_window_value = int(days_match.group(1)) if days_match else None
    time_window_unit = "days" if days_match else None
    if quarter and not time_window_value:
        time_window_value = 90
        time_window_unit = "days"
        assumptions.append("Interpreted next quarter as 90 days.")
    elif month and not time_window_value:
        time_window_value = 30
        time_window_unit = "days"
        assumptions.append("Interpreted this month as 30 days.")
    elif not time_window_value:
        time_window_value = 30
        time_window_unit = "days"
        assumptions.append("No time window supplied; defaulted to 30 days.")

    target_lift_value = float(lift_match.group(1)) if lift_match else None
    target_lift_unit = "percent" if lift_match else None
    if target_lift_value is None and intent != "recommend_best_campaign":
        target_lift_value = 5 if intent in {"reduce_churn", "increase_data_usage"} else 2
        target_lift_unit = "percent"
        assumptions.append(f"No target uplift supplied; defaulted to {target_lift_value:g}%.")

    business_context = "prepaid" if "prepaid" in text or "postpaid" not in text else "postpaid"
    campaign_id = f"CMP_{next(_campaign_counter):03d}"
    return ParsedObjective(
        campaign_id=campaign_id,
        raw_user_prompt=prompt,
        campaign_intent=intent,
        target_segment_hint=target_segment_hint,
        target_metric=metric_by_intent[intent],
        target_lift_value=target_lift_value,
        target_lift_unit=target_lift_unit,
        time_window_value=time_window_value,
        time_window_unit=time_window_unit,
        business_context=business_context,
        confidence=0.91 if matched else 0.72,
        needs_user_clarification=False,
        assumptions=assumptions,
        alternative_intents=[item for item in matched[1:3] if item != intent],
    )


def strategy_summary(intent: str, time_window: str) -> str:
    mapping = {
        "increase_arpu": f"A {time_window} segmented upsell campaign focused on prepaid ARPU growth.",
        "reduce_churn": f"A {time_window} retention campaign focused on declining and at-risk prepaid users.",
        "increase_data_usage": f"A {time_window} usage stimulation campaign for data adoption and frequency.",
        "upsell": f"A {time_window} premium migration campaign for growing, high-confidence segments.",
        "cross_sell": f"A {time_window} cross-sell campaign for segments with unused category potential.",
        "increase_activity": f"A {time_window} activity campaign to lift recharge and usage frequency.",
        "reactivate_inactive": f"A {time_window} winback campaign for dormant and low-activity prepaid users.",
        "recommend_best_campaign": f"A {time_window} ranked opportunity recommendation across eligible campaign families.",
    }
    return mapping.get(intent, f"A {time_window} campaign recommendation.")


def generate_strategy_text(
    intent: str,
    raw_prompt: str,
    time_window: str,
    recommendations: list,
) -> StrategyText:
    fallback = StrategyText(
        campaign_summary=strategy_summary(intent, time_window),
        segment_explanations={
            rec.segment.segment_id: rec.why_this
            for rec in recommendations
        },
    )
    _load_runtime_env()
    if not _llm_enabled():
        return fallback
    try:
        structured_llm = _chat_model("campaign_strategy_explainer").with_structured_output(StrategyText, method="function_calling")
        recommendation_context = [
            {
                "segment_id": rec.segment.segment_id,
                "segment_name": rec.segment.segment_name,
                "customer_count": rec.segment.customer_count,
                "rfm_segment": rec.segment.rfm_segment,
                "data_usage_segment": rec.segment.data_usage_segment,
                "data_usage_trend": rec.segment.data_usage_trend,
                "voice_usage_trend": rec.segment.voice_usage_trend,
                "rulebook_trend": rec.rulebook_match.trend,
                "rulebook_action": rec.rulebook_match.typical_action,
                "allowed_action_families": rec.rulebook_match.allowed_action_families,
                "offer_name": rec.offer.offer_name,
                "primary_channel": rec.ml_score.best_channel,
                "best_time": rec.ml_score.best_time_window,
                "expected_conversion": rec.ml_score.expected_conversion,
                "confidence": rec.confidence,
                "opportunity_score": rec.score.get("opportunity_score"),
            }
            for rec in recommendations
        ]
        system = (
            "You explain campaign recommendations for a telecom marketing demo. "
            "Use only the supplied rulebook, segment, offer, and mock ML fields. "
            "Do not invent categories, formulas, channels, or launch actions. "
            "Make the summary stakeholder-ready and each segment explanation concise."
        )
        user = (
            f"Original objective: {raw_prompt}\n"
            f"Campaign intent: {intent}\n"
            f"Time window: {time_window}\n"
            f"Deterministic recommendation context: {recommendation_context}\n"
            "Return a campaign_summary and segment_explanations keyed by segment_id."
        )
        result: StrategyText = structured_llm.with_config(
            run_name="generate_strategy_explanation",
            tags=["campaign_mvp", "strategy_explanation"],
        ).invoke([("system", system), ("user", user)])
        if not result.campaign_summary:
            return fallback
        return result
    except Exception:
        return fallback


def make_content_drafts(
    segment: Segment,
    offer: Offer,
    primary_channel: str,
    secondary_channel: str,
    user_instruction: str | None = None,
) -> list[ContentDraft]:
    _load_runtime_env()
    if _llm_enabled():
        try:
            return _llm_make_content_drafts(segment, offer, primary_channel, secondary_channel, user_instruction)
        except Exception:
            return _template_content_drafts(segment, offer, primary_channel, secondary_channel, user_instruction)
    return _template_content_drafts(segment, offer, primary_channel, secondary_channel, user_instruction)


def _llm_make_content_drafts(
    segment: Segment,
    offer: Offer,
    primary_channel: str,
    secondary_channel: str,
    user_instruction: str | None = None,
) -> list[ContentDraft]:
    channels = list(dict.fromkeys([primary_channel, secondary_channel]))
    structured_llm = _chat_model("campaign_content_generator").with_structured_output(ContentDraftBatch, method="function_calling")
    system = (
        "You generate draft telecom campaign copy. All copy is draft and requires approval. "
        "Never imply a message has been sent, never include a launch/send action, and never make guaranteed savings claims. "
        "Return one content draft per requested channel. Keep SMS and push concise; make WhatsApp conversational; "
        "make IVR and outbound_call scripts spoken and consent-aware."
    )
    user = (
        f"Segment: {segment.model_dump() if hasattr(segment, 'model_dump') else segment.dict()}\n"
        f"Offer: {offer.model_dump() if hasattr(offer, 'model_dump') else offer.dict()}\n"
        f"Channels: {channels}\n"
        f"Optional user instruction: {user_instruction or 'none'}\n"
        "Each draft must include segment_id, channel, draft_copy, tone, language, approval_required=true, "
        "approved=false, why_this_copy, and compliance_notes."
    )
    batch: ContentDraftBatch = structured_llm.with_config(
        run_name="generate_content_drafts",
        tags=["campaign_mvp", "content_generation", segment.segment_id],
    ).invoke([("system", system), ("user", user)])
    drafts = []
    for draft in batch.drafts:
        if draft.channel not in channels:
            continue
        draft.segment_id = segment.segment_id
        draft.approval_required = True
        draft.approved = False
        if not draft.compliance_notes:
            draft.compliance_notes = ["No guaranteed savings claim", "No misleading urgency", "Approval required before launch"]
        drafts.append(draft)
    return drafts or _template_content_drafts(segment, offer, primary_channel, secondary_channel, user_instruction)


def _template_content_drafts(
    segment: Segment,
    offer: Offer,
    primary_channel: str,
    secondary_channel: str,
    user_instruction: str | None = None,
) -> list[ContentDraft]:
    drafts = []
    channel_templates = {
        "sms": f"{offer.benefit} with {offer.offer_name} for INR {offer.price:.0f}. Valid {offer.validity_days} days. Activate now.",
        "whatsapp": f"Your usage pattern qualifies for {offer.offer_name}: {offer.benefit}. Reply YES to view details before activation.",
        "push": f"{offer.offer_name}: {offer.benefit} for {offer.validity_days} days.",
        "email": f"Based on your recent usage, {offer.offer_name} gives you {offer.benefit}. Review the plan details before activating.",
        "ivr": f"Hello. You are eligible for {offer.offer_name}, offering {offer.benefit}. Press 1 to hear details.",
        "outbound_call": f"Explain that {segment.segment_name} users qualify for {offer.offer_name}: {offer.benefit}. Confirm consent before activation.",
    }
    for channel in dict.fromkeys([primary_channel, secondary_channel]):
        drafts.append(
            ContentDraft(
                segment_id=segment.segment_id,
                channel=channel,
                draft_copy=_apply_copy_instruction(channel_templates.get(channel, channel_templates["sms"]), user_instruction),
                tone="direct" if channel in {"sms", "push"} else "consultative",
                why_this_copy=f"Matched to {segment.segment_name}, {offer.offer_type}, and {channel} engagement behavior.",
                compliance_notes=["No guaranteed savings claim", "No misleading urgency", "Approval required before launch"],
            )
        )
    return drafts


def _apply_copy_instruction(copy: str, user_instruction: str | None) -> str:
    if not user_instruction:
        return copy
    instruction = user_instruction.lower()
    if "short" in instruction:
        return copy.split(".")[0] + "."
    if "formal" in instruction:
        return f"Please review this offer: {copy}"
    if "conversation" in instruction or "conversational" in instruction:
        return f"You may like this: {copy}"
    if "premium" in instruction:
        return copy.replace("Activate now", "Review your premium option")
    return copy


def _chat_model(run_context: str):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        temperature=0.2,
        timeout=20,
        max_retries=1,
        metadata={"app": "campaign_mvp", "run_context": run_context},
    )


def _llm_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and os.getenv("CAMPAIGN_LLM_ENABLED", "true").lower() not in {"0", "false", "no"}


def _load_runtime_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        project_root = Path(__file__).resolve().parents[3]
        load_dotenv(project_root / ".env", override=False)
    except Exception:
        pass

    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_API_KEY", os.environ["LANGSMITH_API_KEY"])
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "campaign-recommendation-mvp")
    _ENV_LOADED = True
