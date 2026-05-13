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


def render_badges(plan: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Status", plan.get("status", "draft").title())
    cols[1].metric("Version", plan.get("version", 1))
    cols[2].metric("Intent", plan.get("campaign_intent", "").replace("_", " ").title())
    cols[3].metric("Target", plan.get("target_lift", ""))
    cols[4].metric("Window", plan.get("time_window", ""))


def render_segment_card(rec: dict) -> None:
    segment = rec["segment"]
    offer = rec["offer"]
    ml = rec["ml_score"]
    st.subheader(segment["segment_name"])
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
    st.subheader(segment["segment_name"])
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
    render_badges(plan)
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
        for rec in plan["recommended_segments"]:
            with st.container(border=True):
                render_segment_card(rec)
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
