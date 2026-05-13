from __future__ import annotations

from app.graph.nodes import (
    calculate_projection_node,
    generate_content_node,
    map_rulebook_node,
    parse_objective_node,
    plan_campaign_node,
    prepare_ui_response_node,
    retrieve_ml_scores_node,
    retrieve_offer_candidates_node,
    retrieve_segments_node,
    validate_campaign_node,
)
from app.graph.state import CampaignGraphState
from app.agents.campaign_deep_agent import deep_agents_enabled, run_campaign_deep_agent_workflow
from app.services.campaign_store import save_campaign_version


NODE_ORDER = [
    "parse_objective",
    "map_rulebook",
    "retrieve_segments",
    "retrieve_ml_scores",
    "retrieve_offer_candidates",
    "plan_campaign",
    "calculate_projection",
    "generate_content",
    "validate_campaign",
    "prepare_ui_response",
]

NODE_FUNCTIONS = {
    "parse_objective": parse_objective_node,
    "map_rulebook": map_rulebook_node,
    "retrieve_segments": retrieve_segments_node,
    "retrieve_ml_scores": retrieve_ml_scores_node,
    "retrieve_offer_candidates": retrieve_offer_candidates_node,
    "plan_campaign": plan_campaign_node,
    "calculate_projection": calculate_projection_node,
    "generate_content": generate_content_node,
    "validate_campaign": validate_campaign_node,
    "prepare_ui_response": prepare_ui_response_node,
}


class DeterministicCampaignRunner:
    def invoke(self, initial_state: CampaignGraphState) -> CampaignGraphState:
        state = dict(initial_state)
        for node_name in NODE_ORDER:
            state.update(NODE_FUNCTIONS[node_name](state))
        return state


def build_campaign_runner():
    return DeterministicCampaignRunner()


def run_campaign_workflow(prompt: str, preferred_campaign_type: str | None = None, version: int = 1):
    if deep_agents_enabled():
        try:
            return run_campaign_deep_agent_workflow(prompt, preferred_campaign_type, version)
        except Exception as exc:
            deep_agent_fallback_warning = f"Deep Agent tool flow failed; deterministic fallback used: {exc.__class__.__name__}."
        else:
            deep_agent_fallback_warning = ""
    else:
        deep_agent_fallback_warning = "Deep Agents disabled; deterministic fallback used."

    runner = build_campaign_runner()
    state = runner.invoke(
        {
            "user_prompt": prompt,
            "preferred_campaign_type": preferred_campaign_type,
            "warnings": [],
            "errors": [],
            "messages": [],
            "version": version,
        }
    )
    plan = state["campaign_plan"]
    state["campaign_plan"] = plan
    state["validation_result"] = plan.validation
    fallback_warnings = [deep_agent_fallback_warning] if deep_agent_fallback_warning else []
    state["warnings"] = list(dict.fromkeys(state.get("warnings", []) + fallback_warnings + (plan.validation.warnings if plan.validation else [])))
    state["errors"] = list(dict.fromkeys(state.get("errors", []) + (plan.validation.errors if plan.validation else [])))
    state["messages"] = state.get("messages", []) + ["Deterministic campaign runner completed."]
    save_campaign_version(plan)
    return state
