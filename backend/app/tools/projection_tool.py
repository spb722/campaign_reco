from __future__ import annotations

import json

from app.schemas.campaign import CampaignPlan, Projection
from app.tools.data_paths import data_dir


def load_assumptions() -> dict:
    return json.loads((data_dir() / "campaign_assumptions.json").read_text(encoding="utf-8"))


def estimate_campaign_impact(campaign_plan: CampaignPlan) -> Projection:
    assumptions = load_assumptions()
    intent = campaign_plan.campaign_intent
    duration_days = _campaign_days(campaign_plan)
    duration_factor = duration_days / 30
    segment_impacts: list[dict] = []
    total = 0.0

    if intent in {"increase_arpu", "upsell", "cross_sell", "recommend_best_campaign"}:
        formula = "eligible_users x expected_conversion x expected_arpu_lift x duration_factor"
        unit = "OMR"
        metric = "incremental_revenue"
        for rec in campaign_plan.recommended_segments:
            conversion = rec.ml_score.expected_conversion or assumptions["arpu"]["default_conversion"]
            lift = rec.offer.estimated_arpu_lift or assumptions["arpu"].get("default_lift_omr", 3.5)
            impact = rec.segment.customer_count * conversion * lift * duration_factor
            total += impact
            segment_impacts.append(
                {
                    "segment_id": rec.segment.segment_id,
                    "eligible_users": rec.segment.customer_count,
                    "expected_conversion": conversion,
                    "expected_arpu_lift": lift,
                    "duration_days": duration_days,
                    "duration_factor": round(duration_factor, 4),
                    "projected_impact": round(impact, 2),
                }
            )
    elif intent == "reduce_churn":
        formula = "at_risk_users x expected_save_rate x duration_factor"
        unit = "customers"
        metric = "saved_customers"
        for rec in campaign_plan.recommended_segments:
            at_risk_users = int(rec.segment.customer_count * rec.segment.churn_risk_score)
            save_rate = rec.offer.estimated_save_rate or assumptions["churn"]["default_save_rate"]
            impact = min(at_risk_users, at_risk_users * save_rate * duration_factor)
            total += impact
            segment_impacts.append(
                {
                    "segment_id": rec.segment.segment_id,
                    "at_risk_users": at_risk_users,
                    "expected_save_rate": save_rate,
                    "duration_days": duration_days,
                    "duration_factor": round(duration_factor, 4),
                    "projected_impact": round(impact, 2),
                }
            )
    elif intent == "increase_data_usage":
        formula = "eligible_users x expected_conversion x expected_gb_lift x duration_factor"
        unit = "GB"
        metric = "incremental_gb"
        for rec in campaign_plan.recommended_segments:
            conversion = rec.ml_score.expected_conversion or assumptions["data_usage"]["default_conversion"]
            gb_lift = rec.offer.estimated_data_lift_gb or assumptions["data_usage"]["default_gb_lift"]
            impact = rec.segment.customer_count * conversion * gb_lift * duration_factor
            total += impact
            segment_impacts.append(
                {
                    "segment_id": rec.segment.segment_id,
                    "eligible_users": rec.segment.customer_count,
                    "expected_conversion": conversion,
                    "expected_gb_lift": gb_lift,
                    "duration_days": duration_days,
                    "duration_factor": round(duration_factor, 4),
                    "projected_impact": round(impact, 2),
                }
            )
    else:
        formula = "inactive_users x expected_reactivation_rate x duration_factor"
        unit = "customers"
        metric = "reactivated_users"
        for rec in campaign_plan.recommended_segments:
            inactive_users = rec.segment.customer_count if rec.segment.inactive_days > 0 else int(rec.segment.customer_count * 0.1)
            rate = assumptions["reactivation"]["default_reactivation_rate"]
            impact = min(inactive_users, inactive_users * rate * duration_factor)
            total += impact
            segment_impacts.append(
                {
                    "segment_id": rec.segment.segment_id,
                    "inactive_users": inactive_users,
                    "expected_reactivation_rate": rate,
                    "duration_days": duration_days,
                    "duration_factor": round(duration_factor, 4),
                    "projected_impact": round(impact, 2),
                }
            )

    for rec in campaign_plan.recommended_segments:
        matching = next((item for item in segment_impacts if item["segment_id"] == rec.segment.segment_id), None)
        if matching:
            rec.projected_impact = matching["projected_impact"]

    return Projection(
        metric=metric,
        formula=formula,
        total_projected_impact=round(total, 2),
        unit=unit,
        segment_impacts=segment_impacts,
        assumptions=[
            "Projection uses segment-level mock data, not customer-level records.",
            "Conversion, save, and lift values are MVP assumptions or mock ML outputs.",
            f"Duration factor is {duration_factor:.2f} based on {duration_days} days / 30-day baseline.",
            "Formula is deterministic and shown for auditability.",
        ],
    )


def _campaign_days(campaign_plan: CampaignPlan) -> int:
    value = campaign_plan.parsed_objective.time_window_value
    unit = campaign_plan.parsed_objective.time_window_unit
    if not value:
        return 30
    if unit == "weeks":
        return int(value * 7)
    if unit == "months":
        return int(value * 30)
    if unit == "quarter":
        return 90
    return int(value)
