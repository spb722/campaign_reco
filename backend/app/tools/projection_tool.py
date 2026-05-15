from __future__ import annotations

import json

from app.schemas.campaign import CampaignPlan, Projection
from app.tools.data_paths import data_dir


def load_assumptions() -> dict:
    return json.loads((data_dir() / "campaign_assumptions.json").read_text(encoding="utf-8"))


def estimate_campaign_impact(campaign_plan: CampaignPlan) -> Projection:
    assumptions = load_assumptions()
    intent = campaign_plan.campaign_intent
    segment_impacts: list[dict] = []
    total = 0.0

    if intent in {"increase_arpu", "upsell", "cross_sell", "recommend_best_campaign"}:
        formula = "eligible_users x expected_conversion x expected_arpu_lift"
        unit = "OMR"
        metric = "incremental_revenue"
        for rec in campaign_plan.recommended_segments:
            conversion = rec.ml_score.expected_conversion or assumptions["arpu"]["default_conversion"]
            lift = rec.offer.estimated_arpu_lift or assumptions["arpu"].get("default_lift_omr", 3.5)
            impact = rec.segment.customer_count * conversion * lift
            total += impact
            segment_impacts.append(
                {
                    "segment_id": rec.segment.segment_id,
                    "eligible_users": rec.segment.customer_count,
                    "expected_conversion": conversion,
                    "expected_arpu_lift": lift,
                    "projected_impact": round(impact, 2),
                }
            )
    elif intent == "reduce_churn":
        formula = "at_risk_users x expected_save_rate"
        unit = "customers"
        metric = "saved_customers"
        for rec in campaign_plan.recommended_segments:
            at_risk_users = int(rec.segment.customer_count * rec.segment.churn_risk_score)
            save_rate = rec.offer.estimated_save_rate or assumptions["churn"]["default_save_rate"]
            impact = at_risk_users * save_rate
            total += impact
            segment_impacts.append(
                {
                    "segment_id": rec.segment.segment_id,
                    "at_risk_users": at_risk_users,
                    "expected_save_rate": save_rate,
                    "projected_impact": round(impact, 2),
                }
            )
    elif intent == "increase_data_usage":
        formula = "eligible_users x expected_conversion x expected_gb_lift"
        unit = "GB"
        metric = "incremental_gb"
        for rec in campaign_plan.recommended_segments:
            conversion = rec.ml_score.expected_conversion or assumptions["data_usage"]["default_conversion"]
            gb_lift = rec.offer.estimated_data_lift_gb or assumptions["data_usage"]["default_gb_lift"]
            impact = rec.segment.customer_count * conversion * gb_lift
            total += impact
            segment_impacts.append(
                {
                    "segment_id": rec.segment.segment_id,
                    "eligible_users": rec.segment.customer_count,
                    "expected_conversion": conversion,
                    "expected_gb_lift": gb_lift,
                    "projected_impact": round(impact, 2),
                }
            )
    else:
        formula = "inactive_users x expected_reactivation_rate"
        unit = "customers"
        metric = "reactivated_users"
        for rec in campaign_plan.recommended_segments:
            inactive_users = rec.segment.customer_count if rec.segment.inactive_days > 0 else int(rec.segment.customer_count * 0.1)
            rate = assumptions["reactivation"]["default_reactivation_rate"]
            impact = inactive_users * rate
            total += impact
            segment_impacts.append(
                {
                    "segment_id": rec.segment.segment_id,
                    "inactive_users": inactive_users,
                    "expected_reactivation_rate": rate,
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
            "Formula is deterministic and shown for auditability.",
        ],
    )
