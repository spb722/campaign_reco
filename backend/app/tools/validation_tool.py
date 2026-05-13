from __future__ import annotations

from app.schemas.campaign import CampaignPlan
from app.schemas.validation import ValidationResult
from app.tools.ml_score_tool import SUPPORTED_CHANNELS
from app.tools.projection_tool import load_assumptions


def validate_campaign_plan(campaign_plan: CampaignPlan) -> ValidationResult:
    warnings = [
        "Uses mock data.",
        "Uses mock ML scores.",
        "Projection is simplified.",
        "Content requires approval.",
        "No production campaign launch integration.",
    ]
    errors: list[str] = []

    if not campaign_plan.parsed_objective:
        errors.append("Campaign plan is missing parsed objective.")
    if not campaign_plan.recommended_segments:
        errors.append("Campaign plan has no recommended segments.")

    for rec in campaign_plan.recommended_segments:
        if not rec.rulebook_match:
            errors.append(f"{rec.segment.segment_id} is missing rulebook match.")
        if rec.segment.customer_count < 0:
            errors.append(f"{rec.segment.segment_id} has negative customer count.")
        if rec.ml_score.fallback_used:
            warnings.append(f"{rec.segment.segment_id}: {rec.ml_score.fallback_reason}")

    for channel in campaign_plan.channel_plan:
        if channel.primary_channel not in SUPPORTED_CHANNELS:
            errors.append(f"Unsupported channel selected: {channel.primary_channel}")
        if channel.score_source not in {"mock_ml", "rulebook_fallback"}:
            errors.append(f"{channel.segment_id} channel source is not traceable.")

    if not campaign_plan.projection:
        errors.append("Campaign plan is missing deterministic projection.")
    elif not campaign_plan.projection.formula:
        errors.append("Projection formula is missing.")

    assumptions = load_assumptions()
    quiet_start = assumptions["quiet_hours"]["start"]
    quiet_end = assumptions["quiet_hours"]["end"]
    for channel in campaign_plan.channel_plan:
        if channel.best_time.startswith("22:") or channel.best_time.startswith("23:"):
            warnings.append(
                f"{channel.segment_id}: best time overlaps quiet-hour boundary {quiet_start}-{quiet_end}; review before approval."
            )

    for draft in campaign_plan.content_plan:
        if not draft.approval_required:
            errors.append(f"{draft.segment_id}/{draft.channel} copy is not marked approval_required.")
        if draft.approved:
            errors.append(f"{draft.segment_id}/{draft.channel} copy cannot be final in Version 1.")

    return ValidationResult(
        is_valid=not errors,
        warnings=warnings,
        errors=errors,
        rulebook_compliance="passed" if not any("rulebook" in error for error in errors) else "failed",
        projection_compliance="passed" if campaign_plan.projection and campaign_plan.projection.formula else "failed",
        content_compliance="passed" if not any("copy" in error for error in errors) else "failed",
    )
