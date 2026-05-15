from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
DATA_DIR = Path(__file__).resolve().parents[1] / "backend" / "app" / "data"


st.set_page_config(page_title="Campaign Recommendation MVP", layout="wide")


def init_state() -> None:
    defaults = {
        "campaign_id": None,
        "user_prompt": "",
        "parsed_objective": None,
        "campaign_plan": None,
        "selected_segment_id": None,
        "edited_content": {},
        "validation_result": None,
        "export_path": None,
        "current_step": "Campaign Request",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def api(method: str, path: str, payload: dict | None = None) -> dict:
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", json=payload, timeout=200)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Backend request failed: {exc}")
        return {"success": False, "data": None, "warnings": [], "errors": [str(exc)], "request_id": "LOCAL"}


def local_sample_prompts() -> list[str]:
    path = DATA_DIR / "sample_prompts.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return [
        "Increase ARPU of mid-ARPU customers by 2% in 30 days.",
        "Reduce prepaid churn by 10% next quarter.",
        "Increase data consumption by 10% over the next quarter.",
        "Recommend the best campaign opportunity for this month.",
    ]


def generate_plan() -> None:
    if not st.session_state.user_prompt.strip():
        st.warning("Enter a campaign objective first.")
        return
    with st.spinner("Running objective parser, rulebook, mock ML scores, projection, and validation..."):
        result = api("POST", "/campaign/recommend", {"prompt": st.session_state.user_prompt})
    if result["data"]:
        st.session_state.campaign_plan = result["data"]
        st.session_state.campaign_id = result["data"]["campaign_id"]
        st.session_state.parsed_objective = result["data"]["parsed_objective"]
        st.session_state.validation_result = result["data"].get("validation")
        st.session_state.current_step = "Recommended Segments"
    for warning in result.get("warnings", []):
        st.warning(warning)
    for error in result.get("errors", []):
        st.error(error)


def render_business_header(plan: dict) -> None:
    st.write(plan.get("summary", ""))
    render_kpi_strip(plan)


def render_kpi_strip(plan: dict) -> None:
    kpis = build_kpis(plan)
    cols = st.columns(len(kpis))
    for col, (label, value) in zip(cols, kpis):
        with col.container(border=True):
            st.caption(label)
            st.markdown(f"## {value}")


def build_kpis(plan: dict) -> list[tuple[str, str]]:
    records = plan.get("recommended_segments", [])
    total_users = sum(rec["segment"]["customer_count"] for rec in records)
    projection = plan.get("projection") or {}
    impact = float(projection.get("total_projected_impact") or 0)
    est_cost = sum(
        rec["segment"]["customer_count"]
        * rec["offer"].get("cost_per_user", 0)
        * rec["ml_score"].get("expected_conversion", 0)
        for rec in records
    )
    intent = plan.get("campaign_intent", "")
    target_lift = plan.get("target_lift", "Target")
    if intent in {"increase_data_usage"}:
        current_usage = weighted_average(records, "avg_data_gb")
        lift_pct = parsed_lift_pct(plan)
        target_usage = current_usage * (1 + lift_pct / 100) if lift_pct else current_usage
        return [
            ("Prepaid base", f"{format_k(total_users)} users"),
            ("Current usage", f"{current_usage:.1f} GB/u/mo"),
            ("Target usage", f"{target_usage:.1f} GB ({target_lift})"),
            ("Quarter gap", f"{format_large_number(impact)} GB"),
            ("Budget cap", format_omr(est_cost)),
        ]
    if intent in {"reduce_churn"}:
        at_risk = sum(int(rec["segment"]["customer_count"] * rec["segment"].get("churn_risk_score", 0)) for rec in records)
        return [
            ("At-risk base", f"{format_k(at_risk)} users"),
            ("Avg churn risk", f"{weighted_average(records, 'churn_risk_score'):.0%}"),
            ("Target save", target_lift),
            ("Saved customers", f"{impact:,.0f}"),
            ("Est. cost", format_omr(est_cost)),
        ]
    if intent in {"reactivate_inactive", "increase_activity"}:
        inactive = sum(rec["segment"]["customer_count"] for rec in records if rec["segment"].get("inactive_days", 0) > 0)
        return [
            ("Inactive base", f"{format_k(inactive or total_users)} users"),
            ("Activity score", f"{weighted_average(records, 'activity_score'):.0%}"),
            ("Target lift", target_lift),
            ("Projected active", f"{impact:,.0f}"),
            ("Est. cost", format_omr(est_cost)),
        ]
    current_arpu = weighted_average(records, "avg_arpu")
    lift_pct = parsed_lift_pct(plan)
    target_arpu = current_arpu * (1 + lift_pct / 100) if lift_pct else current_arpu
    return [
        ("Target base", f"{format_k(total_users)} users"),
        ("Current ARPU", f"OMR {current_arpu:,.2f}/u"),
        ("Target ARPU", f"OMR {target_arpu:,.2f} ({target_lift})"),
        ("Projected impact", format_omr(impact)),
        ("Est. cost", format_omr(est_cost)),
    ]


def weighted_average(records: list[dict], metric: str) -> float:
    total_users = sum(rec["segment"]["customer_count"] for rec in records)
    if not total_users:
        return 0
    return sum(rec["segment"].get(metric, 0) * rec["segment"]["customer_count"] for rec in records) / total_users


def parsed_lift_pct(plan: dict) -> float:
    objective = plan.get("parsed_objective") or {}
    if objective.get("target_lift_unit") == "percent" and objective.get("target_lift_value") is not None:
        return float(objective["target_lift_value"])
    target = plan.get("target_lift", "")
    if target.endswith("%"):
        try:
            return float(target[:-1])
        except ValueError:
            return 0
    return 0


def format_k(value: float) -> str:
    return f"{value / 1_000:.1f}K"


def format_large_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return format_k(value)


def format_omr(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"RO {value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"RO {value / 1_000:.1f}K"
    return f"RO {value:,.2f}"


def render_segment_card(rec: dict) -> None:
    segment = rec["segment"]
    offer = rec["offer"]
    ml = rec["ml_score"]
    st.subheader(segment_display_name(segment))
    if segment.get("customer_signal"):
        st.caption(segment["segment_name"])
    cols = st.columns(4)
    cols[0].metric("Customers", f"{segment['customer_count']:,}")
    cols[1].metric("Expected conversion", f"{ml['expected_conversion']:.0%}")
    cols[2].metric("Projected impact", f"{rec.get('projected_impact', 0):,.0f}")
    cols[3].metric("Confidence", f"{rec['confidence']:.0%}")
    st.caption(f"Rulebook basis: {rec['rulebook_match']['trend']} -> {rec['rulebook_match']['typical_action']}")
    st.write(f"Recommended tactic: **{offer['offer_name']}** ({offer['benefit']})")
    st.write(f"Primary channel: **{ml['best_channel'].title()}**, best time: **{ml['best_time_window']}**")
    st.info(rec["why_this"])
    if st.button("Open drilldown", key=f"drill_{segment['segment_id']}"):
        st.session_state.selected_segment_id = segment["segment_id"]
        st.session_state.current_step = "Segment Drilldown"


def render_segment_dashboard(plan: dict) -> None:
    records = plan.get("recommended_segments", [])[:4]
    if not records:
        st.info("No recommended segments returned.")
        return
    total_users = sum(rec["segment"]["customer_count"] for rec in plan.get("recommended_segments", []))
    total_impact = sum(float(rec.get("projected_impact") or 0) for rec in plan.get("recommended_segments", []))
    cols = st.columns(len(records))
    for index, rec in enumerate(records):
        with cols[index]:
            render_native_segment_card(plan, rec, index, total_users, total_impact)


def render_native_segment_card(plan: dict, rec: dict, index: int, total_users: int, total_impact: float) -> None:
    segment = rec["segment"]
    offer = rec["offer"]
    ml = rec["ml_score"]
    impact = float(rec.get("projected_impact") or 0)
    base_share = segment["customer_count"] / total_users if total_users else 0
    impact_share = impact / total_impact if total_impact else 0
    segment_id = segment["segment_id"]
    with st.container(border=True):
        st.markdown(f"### {segment_display_name(segment)}")
        st.caption(f"{format_k(segment['customer_count'])} users · {base_share:.0%} base · {impact_share:.0%} impact")

        st.markdown("**Profile**")
        for line in segment_profile_lines(segment):
            st.write(line)

        st.markdown("**Key metrics**")
        metric_cols = st.columns(2)
        for metric_index, (label, value) in enumerate(segment_metric_pairs(plan, rec)):
            with metric_cols[metric_index % 2].container(border=True):
                st.caption(label)
                st.markdown(f"**{value}**")

        st.markdown("**Recommended tactic**")
        st.write(f"**{offer['offer_name']}**")
        st.caption(f"{offer.get('benefit', '')} · {offer.get('description', '')}")

        st.markdown("**Channels & send window**")
        st.dataframe(channel_table(ml), use_container_width=True, hide_index=True)

        st.markdown("**Drip schedule**")
        for step in followup_steps(plan, segment_id):
            st.write(f"- {step}")

        with st.expander("Why this?", expanded=False):
            st.write(rec.get("why_this", ""))

        if st.button("Open drilldown", key=f"drill_compact_{segment_id}", use_container_width=True):
            st.session_state.selected_segment_id = segment_id
            st.session_state.current_step = "Segment Drilldown"


def segment_profile_lines(segment: dict) -> list[str]:
    parts = [
        f"Original segment: {segment.get('segment_name', 'Segment')}",
        f"{segment.get('rfm_segment', 'Segment')} · {segment.get('opportunity', 'Opportunity')}",
        f"Meaning: {segment.get('customer_meaning')}" if segment.get("customer_meaning") else "",
        f"{segment.get('data_usage_segment', '').title()} data · {segment.get('voice_usage_segment', '').title()} voice",
        f"{segment.get('data_usage_trend', '')} data trend",
        f"NBO action: {segment.get('nbo_action')}" if segment.get("nbo_action") else "",
        f"{segment.get('current_pack_type', '').replace('_', ' ').title()} pack",
    ]
    if segment.get("inactive_days", 0):
        parts.append(f"{segment['inactive_days']} inactive days")
    return [part for part in parts if part.strip()]


def segment_display_name(segment: dict) -> str:
    return segment.get("customer_signal") or segment.get("segment_name") or segment.get("segment_id", "Segment")


def segment_metric_pairs(plan: dict, rec: dict) -> list[tuple[str, str]]:
    segment = rec["segment"]
    offer = rec["offer"]
    ml = rec["ml_score"]
    intent = plan.get("campaign_intent", "")
    if intent == "increase_data_usage":
        metrics = [
            ("Current usage", f"{segment['avg_data_gb']:.0f} GB/mo"),
            ("Lift / user", f"+{offer.get('estimated_data_lift_gb', 0):.0f} GB"),
            ("Target conv", f"{ml['expected_conversion']:.0%}"),
            ("Quarter gap", f"{format_large_number(rec.get('projected_impact', 0))} GB"),
        ]
    elif intent == "reduce_churn":
        at_risk = int(segment["customer_count"] * segment.get("churn_risk_score", 0))
        metrics = [
            ("At-risk users", f"{format_k(at_risk)}"),
            ("Save rate", f"{offer.get('estimated_save_rate', 0):.0%}"),
            ("Churn risk", f"{segment.get('churn_risk_score', 0):.0%}"),
            ("Saved", f"{rec.get('projected_impact', 0):,.0f}"),
        ]
    elif intent in {"reactivate_inactive", "increase_activity"}:
        metrics = [
            ("Inactive days", f"{segment.get('inactive_days', 0)}"),
            ("Activity", f"{segment.get('activity_score', 0):.0%}"),
            ("Target conv", f"{ml['expected_conversion']:.0%}"),
            ("Projected", f"{rec.get('projected_impact', 0):,.0f}"),
        ]
    else:
        metrics = [
            ("Current ARPU", f"OMR {segment['avg_arpu']:.2f}"),
            ("Lift / user", f"OMR {offer.get('estimated_arpu_lift', 0):.2f}"),
            ("Target conv", f"{ml['expected_conversion']:.0%}"),
            ("Revenue lift", format_omr(rec.get("projected_impact", 0))),
        ]
    return metrics


def channel_table(ml: dict) -> pd.DataFrame:
    scores = sorted(ml.get("channel_scores", {}).items(), key=lambda item: item[1], reverse=True)[:3]
    rows = []
    for channel, score in scores:
        window = ml.get("best_time_window", "") if channel == ml.get("best_channel") else "Fallback"
        rows.append({"Channel": channel.replace("_", " ").title(), "Window": window, "Score": f"{score:.0%}"})
    return pd.DataFrame(rows)


def followup_steps(plan: dict, segment_id: str) -> list[str]:
    followup = next((item for item in plan.get("followup_plan", []) if item["segment_id"] == segment_id), {})
    return followup.get("steps", [])[:3]


def find_selected_segment(plan: dict) -> dict | None:
    selected = st.session_state.selected_segment_id
    records = plan.get("recommended_segments", [])
    if not selected and records:
        st.session_state.selected_segment_id = records[0]["segment"]["segment_id"]
        selected = st.session_state.selected_segment_id
    return next((record for record in records if record["segment"]["segment_id"] == selected), None)


def render_drilldown(plan: dict) -> None:
    rec = find_selected_segment(plan)
    if not rec:
        st.info("Select a segment from Recommended Segments.")
        return
    segment = rec["segment"]
    st.subheader(segment_display_name(segment))
    st.caption(segment["segment_name"])
    tab_names = [
        "Profile",
        "Rulebook",
        "Mock Metrics",
        "ML Scores",
        "Tactic",
        "Channel",
        "Content",
        "Follow-up",
        "Assumptions",
        "Guardrails",
    ]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        profile_rows = {
            "display_name": segment_display_name(segment),
            "original_segment": segment.get("segment_name"),
            "customer_signal": segment.get("customer_signal"),
            "customer_meaning": segment.get("customer_meaning"),
            "rfm_segment": segment.get("rfm_segment"),
            "data_usage_segment": segment.get("data_usage_segment"),
            "voice_usage_segment": segment.get("voice_usage_segment"),
            "data_usage_trend": segment.get("data_usage_trend"),
            "voice_usage_trend": segment.get("voice_usage_trend"),
            "opportunity": segment.get("opportunity"),
            "nbo_action": segment.get("nbo_action"),
        }
        st.dataframe(pd.DataFrame([profile_rows]), use_container_width=True, hide_index=True)
        with st.expander("Raw segment record", expanded=False):
            st.json(segment)
    with tabs[1]:
        st.json(rec["rulebook_match"])
    with tabs[2]:
        metrics = {
            "avg_arpu": segment["avg_arpu"],
            "avg_data_gb": segment["avg_data_gb"],
            "avg_voice_min": segment["avg_voice_min"],
            "recharge_frequency_days": segment["recharge_frequency_days"],
            "churn_risk_score": segment["churn_risk_score"],
            "activity_score": segment["activity_score"],
            "inactive_days": segment["inactive_days"],
        }
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
    with tabs[3]:
        scores = rec["ml_score"]["channel_scores"]
        st.caption("Channel/timing are ML-score based for MVP.")
        st.bar_chart(pd.Series(scores))
        st.json(rec["ml_score"])
    with tabs[4]:
        st.json(rec["offer"])
    with tabs[5]:
        channel_rows = [item for item in plan["channel_plan"] if item["segment_id"] == segment["segment_id"]]
        st.dataframe(pd.DataFrame(channel_rows), use_container_width=True)
    with tabs[6]:
        drafts = [draft for draft in plan["content_plan"] if draft["segment_id"] == segment["segment_id"]]
        for draft in drafts:
            st.markdown(f"**{draft['channel'].title()}**")
            new_copy = st.text_area("Draft copy", draft["draft_copy"], key=f"edit_{segment['segment_id']}_{draft['channel']}")
            st.session_state.edited_content[f"{segment['segment_id']}:{draft['channel']}"] = new_copy
            controls = st.columns(5)
            if controls[0].button("Regenerate", key=f"regen_{segment['segment_id']}_{draft['channel']}"):
                regenerate("content_only", segment["segment_id"], "Refresh this draft.")
            if controls[1].button("Shorter", key=f"short_{segment['segment_id']}_{draft['channel']}"):
                regenerate("content_only", segment["segment_id"], "Make it shorter.")
            if controls[2].button("Formal", key=f"formal_{segment['segment_id']}_{draft['channel']}"):
                regenerate("content_only", segment["segment_id"], "Make it more formal.")
            if controls[3].button("Conversational", key=f"conv_{segment['segment_id']}_{draft['channel']}"):
                regenerate("content_only", segment["segment_id"], "Make it more conversational.")
            controls[4].button("Approve draft", disabled=True, key=f"approve_{segment['segment_id']}_{draft['channel']}")
            st.caption("Approval is intentionally disabled for Version 1 exports; all copy remains draft.")
    with tabs[7]:
        st.json(next((item for item in plan["followup_plan"] if item["segment_id"] == segment["segment_id"]), {}))
    with tabs[8]:
        st.json(plan.get("projection", {}).get("assumptions", []))
    with tabs[9]:
        st.json({"risks": plan.get("risks", []), "validation": plan.get("validation", {})})


def regenerate(scope: str, segment_id: str | None = None, instruction: str | None = None) -> None:
    campaign_id = st.session_state.campaign_id
    if not campaign_id:
        return
    result = api("POST", f"/campaign/{campaign_id}/regenerate", {"regenerate_scope": scope, "segment_id": segment_id, "user_instruction": instruction})
    if result["data"]:
        st.session_state.campaign_plan = result["data"]
        st.session_state.validation_result = result["data"].get("validation")


def render_validation(plan: dict) -> None:
    projection = plan.get("projection") or {}
    st.subheader("Projection")
    st.caption("Projection is deterministic and not generated by LLM prose.")
    st.json(projection)
    with st.expander("View assumptions", expanded=True):
        st.write(plan.get("assumptions", []))
        st.write(projection.get("assumptions", []))

    st.subheader("Validation")
    validation = plan.get("validation") or {}
    if validation.get("is_valid"):
        st.success("Validation passed for draft/export readiness.")
    else:
        st.error("Validation has blocking errors.")
    for warning in validation.get("warnings", []):
        st.warning(warning)
    for error in validation.get("errors", []):
        st.error(error)


def render_export() -> None:
    campaign_id = st.session_state.campaign_id
    if not campaign_id:
        st.info("Generate a campaign plan before exporting.")
        return
    if st.button("Generate one-pager PDF and JSON"):
        result = api("POST", f"/campaign/{campaign_id}/export")
        if result["success"]:
            st.session_state.export_path = result["data"]["pdf_path"]
            st.success("One-pager export generated.")
        else:
            st.error("Export blocked by validation.")
    if st.session_state.export_path:
        st.code(st.session_state.export_path)
        st.link_button("Download PDF", f"{BACKEND_URL}/campaign/{campaign_id}/download")


init_state()

st.title("Campaign Recommendation MVP")
st.caption("Objective -> Segment -> Plan -> Channels & Content -> Follow-up -> Live Report. Mock data and mock ML scores are clearly flagged.")

for prompt in local_sample_prompts():
    if st.button(prompt, key=f"prompt_{prompt}"):
        st.session_state.user_prompt = prompt

st.session_state.user_prompt = st.text_area("Campaign objective", value=st.session_state.user_prompt, height=100)
if st.button("Generate Campaign Plan", type="primary"):
    generate_plan()

plan = st.session_state.campaign_plan
if plan:
    render_business_header(plan)
    sections = st.tabs(
        [
            "Parsed Objective",
            "Recommended Segments",
            "Campaign Plan",
            "Segment Drilldown",
            "Content Drafts",
            "Assumptions & Validation",
            "One-Pager Export",
        ]
    )
    with sections[0]:
        st.json(plan["parsed_objective"])
    with sections[1]:
        render_segment_dashboard(plan)
    with sections[2]:
        st.subheader(plan["campaign_title"])
        st.write(plan["summary"])
        st.dataframe(pd.DataFrame(plan["campaign_tactics"]), use_container_width=True)
        if st.button("Regenerate segment strategy"):
            regenerate("segment_strategy")
    with sections[3]:
        render_drilldown(plan)
    with sections[4]:
        by_segment = {}
        for draft in plan["content_plan"]:
            by_segment.setdefault(draft["segment_id"], []).append(draft)
        for segment_id, drafts in by_segment.items():
            with st.expander(segment_id, expanded=True):
                for draft in drafts:
                    st.markdown(f"**{draft['channel'].title()}**")
                    st.write(draft["draft_copy"])
                    st.caption(draft["why_this_copy"])
    with sections[5]:
        render_validation(plan)
    with sections[6]:
        render_export()
else:
    st.info("Start with an example prompt or enter a custom objective.")
