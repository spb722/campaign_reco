from __future__ import annotations

from app.graph.state import CampaignGraphState
from app.schemas.campaign import CampaignPlan, ChannelPlan, RecommendedSegment
from app.schemas.objective import ParsedObjective
from app.schemas.segment import RulebookMatch, Segment
from app.services.llm_service import generate_strategy_text, make_content_drafts, parse_objective
from app.tools.ml_score_tool import load_mock_ml_scores
from app.tools.offer_tool import get_offer_candidates
from app.tools.projection_tool import estimate_campaign_impact, load_assumptions
from app.tools.rulebook_tool import get_rulebook_matches
from app.tools.segment_tool import get_segment_candidates
from app.tools.validation_tool import validate_campaign_plan


def parse_objective_node(state: CampaignGraphState) -> dict:
    parsed = parse_objective(state["user_prompt"], state.get("preferred_campaign_type"))
    return {"campaign_id": parsed.campaign_id, "parsed_objective": parsed, "warnings": parsed.assumptions}


def map_rulebook_node(state: CampaignGraphState) -> dict:
    parsed: ParsedObjective = state["parsed_objective"]
    matches = get_rulebook_matches(parsed)
    return {"rulebook_matches": matches}


def retrieve_segments_node(state: CampaignGraphState) -> dict:
    segments = get_segment_candidates(state["rulebook_matches"], state["parsed_objective"])
    return {"segment_candidates": segments}


def retrieve_ml_scores_node(state: CampaignGraphState) -> dict:
    segment_ids = [segment.segment_id for segment in state["segment_candidates"]]
    return {"ml_scores": load_mock_ml_scores(segment_ids)}


def retrieve_offer_candidates_node(state: CampaignGraphState) -> dict:
    return {"offer_candidates": get_offer_candidates(state["segment_candidates"], state["parsed_objective"])}


def plan_campaign_node(state: CampaignGraphState) -> dict:
    parsed: ParsedObjective = state["parsed_objective"]
    rulebook_matches: list[RulebookMatch] = state["rulebook_matches"]
    segments: list[Segment] = state["segment_candidates"]
    ml_scores = state["ml_scores"]
    offer_candidates = state["offer_candidates"]
    recommendations = []
    channel_plan = []
    tactics = []
    followup = []
    assumptions = list(parsed.assumptions)
    assumptions.extend(["Rulebook eligibility is deterministic.", "Channel and timing recommendations come from mock ML scores."])

    for segment in segments:
        rule = _best_rule_for_segment(segment, rulebook_matches)
        offer = _best_offer_for_segment(segment, offer_candidates.get(segment.segment_id, []), parsed.campaign_intent)
        ml_score = ml_scores[segment.segment_id]
        score = _opportunity_score(segment, rule, ml_score, offer)
        action = rule.allowed_action_families[0] if rule.allowed_action_families else rule.typical_action.lower()
        recommendations.append(
            RecommendedSegment(
                segment=segment,
                rulebook_match=rule,
                recommended_action=action,
                offer=offer,
                ml_score=ml_score,
                confidence=round((rule.rulebook_fit_score + ml_score.model_confidence) / 2, 2),
                score=score,
                why_this=f"{rule.trend} maps to {rule.typical_action}; {segment.segment_name} has {segment.customer_count:,} customers and {ml_score.best_channel} is the highest scoring channel.",
            )
        )
        channel_plan.append(
            ChannelPlan(
                segment_id=segment.segment_id,
                primary_channel=ml_score.best_channel,
                secondary_channel=ml_score.secondary_channel,
                best_time=ml_score.best_time_window,
                expected_ctr=ml_score.expected_ctr,
                expected_conversion=ml_score.expected_conversion,
                fatigue_risk=ml_score.fatigue_risk,
                score_source="rulebook_fallback" if ml_score.fallback_used else "mock_ml",
                channel_scores=ml_score.channel_scores,
            )
        )
        tactics.append(
            {
                "segment_id": segment.segment_id,
                "action_family": action,
                "offer_id": offer.offer_id,
                "offer_name": offer.offer_name,
                "why_this": f"{offer.description} Rulebook basis: {rule.trend} -> {rule.typical_action}.",
            }
        )
        followup.append(
            {
                "segment_id": segment.segment_id,
                "steps": [
                    f"Day 0: {ml_score.best_channel} primary message",
                    f"Day 3: {ml_score.secondary_channel} reminder if no conversion",
                    "Day 10: suppress converted users and review fatigue",
                    "Day 21: final low-frequency reminder within cap",
                ],
            }
        )

    recommendations = sorted(recommendations, key=lambda rec: rec.score["opportunity_score"], reverse=True)[:5]
    selected_ids = {rec.segment.segment_id for rec in recommendations}
    channel_plan = [item for item in channel_plan if item.segment_id in selected_ids]
    tactics = [item for item in tactics if item["segment_id"] in selected_ids]
    followup = [item for item in followup if item["segment_id"] in selected_ids]

    time_window = f"{parsed.time_window_value} {parsed.time_window_unit}"
    target_lift = (
        f"{parsed.target_lift_value:g}%" if parsed.target_lift_value is not None and parsed.target_lift_unit == "percent" else "Best available"
    )
    strategy_text = generate_strategy_text(parsed.campaign_intent, parsed.raw_user_prompt, time_window, recommendations)
    for recommendation in recommendations:
        recommendation.why_this = strategy_text.segment_explanations.get(recommendation.segment.segment_id, recommendation.why_this)
    plan = CampaignPlan(
        campaign_id=parsed.campaign_id,
        campaign_title=_campaign_title(parsed),
        campaign_intent=parsed.campaign_intent,
        summary=strategy_text.campaign_summary,
        target_metric=parsed.target_metric,
        target_lift=target_lift,
        time_window=time_window,
        parsed_objective=parsed,
        recommended_segments=recommendations,
        campaign_tactics=tactics,
        channel_plan=channel_plan,
        followup_plan=followup,
        assumptions=assumptions,
        risks=[
            "Mock ML confidence may not reflect production response.",
            "Frequency cap must be reviewed before real launch.",
            "Draft copy requires business and compliance approval.",
        ],
        version=state.get("version", 1),
    )
    return {"selected_segments": recommendations, "campaign_plan": plan}


def calculate_projection_node(state: CampaignGraphState) -> dict:
    plan: CampaignPlan = state["campaign_plan"]
    projection = estimate_campaign_impact(plan)
    plan.projection = projection
    return {"campaign_plan": plan, "projection": projection}


def generate_content_node(state: CampaignGraphState) -> dict:
    plan: CampaignPlan = state["campaign_plan"]
    drafts = []
    for rec in plan.recommended_segments:
        drafts.extend(
            make_content_drafts(
                rec.segment,
                rec.offer,
                rec.ml_score.best_channel,
                rec.ml_score.secondary_channel,
            )
        )
    plan.content_plan = drafts
    return {"campaign_plan": plan, "content_plan": drafts}


def validate_campaign_node(state: CampaignGraphState) -> dict:
    plan: CampaignPlan = state["campaign_plan"]
    validation = validate_campaign_plan(plan)
    plan.validation = validation
    warnings = list(dict.fromkeys(state.get("warnings", []) + validation.warnings))
    errors = list(dict.fromkeys(state.get("errors", []) + validation.errors))
    return {"campaign_plan": plan, "validation_result": validation, "warnings": warnings, "errors": errors}


def prepare_ui_response_node(state: CampaignGraphState) -> dict:
    return {"messages": ["Campaign plan prepared for Streamlit display."]}


def _best_rule_for_segment(segment: Segment, rules: list[RulebookMatch]) -> RulebookMatch:
    if segment.opportunity:
        for rule in rules:
            if rule.typical_action == segment.opportunity and rule.trend in {segment.data_usage_trend, segment.voice_usage_trend}:
                return rule
    for trend in (segment.data_usage_trend, segment.voice_usage_trend):
        for rule in rules:
            if rule.trend == trend:
                return rule
    return rules[0]


def _best_offer_for_segment(segment: Segment, offers: list, intent: str):
    if not offers:
        raise ValueError(f"No offer candidate found for {segment.segment_id}")
    if intent == "recommend_best_campaign":
        return sorted(offers, key=lambda offer: (offer.margin_impact == "positive", offer.estimated_arpu_lift), reverse=True)[0]
    affinity_match = [offer for offer in offers if offer.offer_type == segment.offer_affinity or offer.offer_id == segment.offer_affinity]
    return affinity_match[0] if affinity_match else offers[0]


def _opportunity_score(segment: Segment, rule: RulebookMatch, ml_score, offer) -> dict[str, float]:
    assumptions = load_assumptions()
    impact_proxy = min(1.0, (segment.customer_count * max(ml_score.expected_conversion, 0.01)) / 150000)
    rulebook_fit = rule.rulebook_fit_score
    ml_confidence = ml_score.model_confidence
    cost_efficiency = max(0.1, min(1.0, 1 - (offer.cost_per_user / 100)))
    execution_feasibility = 0.9 if ml_score.fatigue_risk == "low" else 0.75 if ml_score.fatigue_risk == "medium" else 0.58
    opportunity = (
        0.30 * impact_proxy
        + 0.25 * rulebook_fit
        + 0.20 * ml_confidence
        + 0.15 * cost_efficiency
        + 0.10 * execution_feasibility
    )
    if segment.recharge_frequency_days > assumptions["default_frequency_cap_per_30_days"] * 10:
        opportunity -= 0.03
    return {
        "projected_impact_score": round(impact_proxy, 2),
        "rulebook_fit_score": round(rulebook_fit, 2),
        "ml_confidence_score": round(ml_confidence, 2),
        "cost_efficiency_score": round(cost_efficiency, 2),
        "execution_feasibility_score": round(execution_feasibility, 2),
        "opportunity_score": round(max(0, min(1, opportunity)), 2),
    }


def _campaign_title(parsed: ParsedObjective) -> str:
    titles = {
        "increase_arpu": "Mid ARPU Growth Campaign",
        "reduce_churn": "Prepaid Churn Reduction Campaign",
        "increase_data_usage": "Prepaid Data Consumption Campaign",
        "upsell": "Premium Upsell Campaign",
        "cross_sell": "Cross-Sell Opportunity Campaign",
        "increase_activity": "Usage Activity Campaign",
        "reactivate_inactive": "Inactive Customer Winback Campaign",
        "recommend_best_campaign": "Best Campaign Opportunity Recommendation",
    }
    return titles[parsed.campaign_intent]
