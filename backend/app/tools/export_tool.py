from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.schemas.campaign import CampaignPlan
from app.tools.data_paths import output_dir


def _dump_model(model: Any) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def save_campaign_json(campaign_plan: CampaignPlan) -> str:
    path = output_dir("json") / f"{campaign_plan.campaign_id}_v{campaign_plan.version}.json"
    path.write_text(json.dumps(_dump_model(campaign_plan), indent=2, default=str), encoding="utf-8")
    return str(path)


def generate_one_pager_pdf(campaign_plan: CampaignPlan) -> str:
    path = output_dir("pdfs") / f"{campaign_plan.campaign_id}_v{campaign_plan.version}.pdf"
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        doc = SimpleDocTemplate(str(path), pagesize=landscape(A4), rightMargin=24, leftMargin=24, topMargin=20, bottomMargin=20)
        styles = getSampleStyleSheet()
        story = []
        generated = datetime.now().strftime("%Y-%m-%d %H:%M")
        story.append(Paragraph(campaign_plan.campaign_title, styles["Title"]))
        story.append(
            Paragraph(
                f"Objective: {campaign_plan.parsed_objective.raw_user_prompt} | Status: Draft | Owner: Campaign Team | Generated: {generated}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 8))

        projection = campaign_plan.projection
        projected = f"{projection.total_projected_impact:,.0f} {projection.unit}" if projection else "Pending"
        kpi_table = Table(
            [
                ["Target base", "Current metric", "Target metric", "Projected impact", "Estimated cost"],
                [
                    f"{sum(r.segment.customer_count for r in campaign_plan.recommended_segments):,} users",
                    campaign_plan.target_metric,
                    campaign_plan.target_lift,
                    projected,
                    f"OMR {sum(r.segment.customer_count * r.offer.cost_per_user * r.ml_score.expected_conversion for r in campaign_plan.recommended_segments):,.0f}",
                ],
            ],
            colWidths=[145, 145, 145, 145, 145],
        )
        kpi_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
        story.append(kpi_table)
        story.append(Spacer(1, 10))

        columns = [["Segment", "Profile", "Key metrics", "Tactic", "Channel mix", "Follow-up"]]
        for rec in campaign_plan.recommended_segments[:4]:
            followup = next((item for item in campaign_plan.followup_plan if item["segment_id"] == rec.segment.segment_id), {})
            columns.append(
                [
                    rec.segment.segment_name,
                    f"{rec.segment.rfm_segment}; {rec.segment.data_usage_trend}",
                    f"Users {rec.segment.customer_count:,}; ARPU {rec.segment.avg_arpu:.0f}; churn {rec.segment.churn_risk_score:.0%}",
                    f"{rec.offer.offer_name}: {rec.offer.benefit}",
                    f"{rec.ml_score.best_channel} + {rec.ml_score.secondary_channel}; {rec.ml_score.best_time_window}",
                    " -> ".join(followup.get("steps", [])),
                ]
            )
        segment_table = Table(columns, colWidths=[105, 120, 125, 155, 130, 180])
        segment_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEBD8")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(segment_table)
        story.append(Spacer(1, 10))
        validation = campaign_plan.validation
        story.append(
            Paragraph(
                "Footer governance: "
                f"Rulebook compliance {validation.rulebook_compliance if validation else 'pending'} | "
                f"System confidence {sum(r.confidence for r in campaign_plan.recommended_segments) / max(len(campaign_plan.recommended_segments), 1):.0%} | "
                "Frequency cap 4 per 30 days | Quiet hours 22:00-08:00 | Approval status Draft",
                styles["Normal"],
            )
        )
        doc.build(story)
    except Exception:
        _write_minimal_pdf(path, campaign_plan)
    return str(path)


def _write_minimal_pdf(path: Path, campaign_plan: CampaignPlan) -> None:
    # Minimal valid PDF fallback when ReportLab is unavailable.
    text = f"{campaign_plan.campaign_title}\\n{campaign_plan.summary}".replace("(", "[").replace(")", "]")
    stream = f"BT /F1 18 Tf 72 520 Td ({text[:300]}) Tj ET"
    pdf = (
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 842 595] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        f"5 0 obj << /Length {len(stream)} >> stream\n{stream}\nendstream endobj\n"
        "xref\n0 6\n0000000000 65535 f \n"
        "trailer << /Root 1 0 R /Size 6 >>\nstartxref\n0\n%%EOF\n"
    )
    path.write_text(pdf, encoding="latin-1")
