from __future__ import annotations

import json
import os
import re
import inspect
from uuid import uuid4
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.campaign import CampaignPlan
from app.schemas.content import ContentDraft
from app.services.llm_service import _chat_model, _llm_enabled, _load_runtime_env
from app.services.campaign_store import save_campaign_version
from app.tools.ml_score_tool import load_mock_ml_scores
from app.tools.offer_tool import get_offer_candidates
from app.tools.projection_tool import estimate_campaign_impact
from app.tools.rulebook_tool import get_rulebook_matches
from app.tools.segment_tool import get_segment_candidates
from app.tools.validation_tool import validate_campaign_plan


class DeepAgentContentEdit(BaseModel):
    segment_id: str
    channel: str
    draft_copy: str
    tone: str = "direct"
    why_this_copy: str
    compliance_notes: list[str] = Field(default_factory=list)


class DeepAgentPlanEdits(BaseModel):
    campaign_summary: str | None = None
    segment_explanations: dict[str, str] = Field(default_factory=dict)
    content_edits: list[DeepAgentContentEdit] = Field(default_factory=list)
    agent_notes: list[str] = Field(default_factory=list)


_RUN_CONTEXTS: dict[str, dict[str, Any]] = {}


def deep_agents_enabled() -> bool:
    _load_runtime_env()
    if not _llm_enabled():
        return False
    if os.getenv("CAMPAIGN_DEEP_AGENTS_ENABLED", "true").lower() in {"0", "false", "no"}:
        return False
    return True


def run_campaign_deep_agent_workflow(
    prompt: str,
    preferred_campaign_type: str | None = None,
    version: int = 1,
) -> dict[str, Any]:
    """Run the campaign recommendation through a Deep Agent tool flow.

    The agent chooses and calls tools. The backend still validates the final
    typed CampaignPlan before returning it.
    """
    if not deep_agents_enabled():
        raise RuntimeError("Deep Agents are disabled or the configured LLM provider API key is missing.")

    context_id = f"campaign_run_{uuid4().hex}"
    _RUN_CONTEXTS[context_id] = {
        "prompt": prompt,
        "preferred_campaign_type": preferred_campaign_type,
        "version": version,
        "warnings": [],
        "errors": [],
        "messages": [],
    }
    agent = _create_campaign_deep_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Run a campaign recommendation for context_id={context_id}.\n"
                        f"User objective: {prompt}\n"
                        f"Preferred campaign type: {preferred_campaign_type or 'none'}\n\n"
                        "Call the tools needed to complete the plan. For a new campaign recommendation, call these tools in order unless a tool result clearly makes the next one irrelevant: "
                        "parse_objective_tool, lookup_rulebook_tool, find_segments_tool, load_ml_scores_tool, select_offer_candidates_tool, "
                        "assemble_campaign_plan_tool, calculate_projection_tool, generate_content_tool, validate_campaign_tool, get_final_campaign_plan_tool. "
                        "Do not invent rulebook, segment, offer, ML, projection, or validation outputs. Use the tools."
                    ),
                }
            ]
        },
        config={"run_name": "campaign_deep_agent_orchestrator", "tags": ["campaign_mvp", "deepagents", "tool_flow"]},
    )
    context = _RUN_CONTEXTS[context_id]
    plan = context.get("campaign_plan")
    if not isinstance(plan, CampaignPlan):
        raise RuntimeError(f"Deep Agent did not produce a campaign plan. Last response: {_message_text(result)[:500]}")

    plan.validation = validate_campaign_plan(plan)
    context["validation_result"] = plan.validation
    context["warnings"] = list(dict.fromkeys(context.get("warnings", []) + plan.validation.warnings))
    context["errors"] = list(dict.fromkeys(context.get("errors", []) + plan.validation.errors))
    context["messages"] = context.get("messages", []) + ["Campaign Deep Agent completed tool-driven recommendation flow."]
    save_campaign_version(plan)
    return {
        "campaign_id": plan.campaign_id,
        "user_prompt": prompt,
        "parsed_objective": context.get("parsed_objective"),
        "rulebook_matches": context.get("rulebook_matches", []),
        "segment_candidates": context.get("segment_candidates", []),
        "ml_scores": context.get("ml_scores", {}),
        "offer_candidates": context.get("offer_candidates", {}),
        "selected_segments": plan.recommended_segments,
        "campaign_plan": plan,
        "content_plan": plan.content_plan,
        "projection": plan.projection,
        "validation_result": plan.validation,
        "export_path": plan.export_path,
        "messages": context["messages"],
        "warnings": context["warnings"],
        "errors": context["errors"],
        "version": version,
    }


def _create_campaign_deep_agent():
    from deepagents import create_deep_agent

    model = _chat_model(
        "campaign_deepagents_tool_flow",
        temperature=0.1,
        timeout=45,
        max_retries=1,
        metadata={"orchestrator": "deepagents_tool_flow"},
    )
    system_prompt = (
        "You are the Campaign Recommendation Deep Agent. You orchestrate a telecom campaign planning workflow by calling tools. "
        "Use tools for data and business logic; do not invent rulebook rows, segments, ML scores, offers, formulas, validation results, or final plan fields. "
        "For recommendation requests, call the planning tools in a sensible sequence and return a concise final note after get_final_campaign_plan_tool. "
        "You may delegate analysis to subagents, but the final campaign data must come from tools."
    )
    subagents = _subagent_specs(
        [
            {
                "name": "rulebook-agent",
                "description": "Decides when rulebook lookup is needed and explains allowed campaign actions.",
                "prompt": "Use rulebook tool outputs only. Do not create unsupported campaign action families.",
            },
            {
                "name": "segment-analyst",
                "description": "Reviews segment, ML score, and offer tool outputs for campaign fit.",
                "prompt": "Use segment and mock ML outputs only. Explain fit without changing deterministic values.",
            },
            {
                "name": "copywriter",
                "description": "Reviews draft content for tone and approval guardrails.",
                "prompt": "Keep all copy draft-only, approval-required, and compliant.",
            },
        ]
    )
    kwargs = {
        "model": model,
        "tools": [
            parse_objective_tool,
            lookup_rulebook_tool,
            find_segments_tool,
            load_ml_scores_tool,
            select_offer_candidates_tool,
            assemble_campaign_plan_tool,
            calculate_projection_tool,
            generate_content_tool,
            validate_campaign_tool,
            get_final_campaign_plan_tool,
        ],
        "subagents": subagents,
    }
    prompt_param = _deep_agent_prompt_param(create_deep_agent)
    kwargs[prompt_param] = system_prompt
    return create_deep_agent(**kwargs)


def parse_objective_tool(context_id: str) -> str:
    """Parse the user campaign objective into a structured objective and store it for this context_id."""
    from app.services.llm_service import parse_objective

    context = _context(context_id)
    parsed = parse_objective(context["prompt"], context.get("preferred_campaign_type"))
    context["parsed_objective"] = parsed
    context["campaign_id"] = parsed.campaign_id
    context["warnings"] = list(dict.fromkeys(context.get("warnings", []) + parsed.assumptions))
    return _json(parsed)


def lookup_rulebook_tool(context_id: str) -> str:
    """Look up deterministic rulebook matches for the parsed objective in this context_id."""
    context = _context(context_id)
    _ensure_parsed(context_id)
    parsed = context["parsed_objective"]
    matches = get_rulebook_matches(parsed)
    context["rulebook_matches"] = matches
    return _json(matches)


def find_segments_tool(context_id: str) -> str:
    """Find deterministic mock segment candidates from parsed objective and rulebook matches."""
    context = _context(context_id)
    _ensure_rulebook(context_id)
    segments = get_segment_candidates(context["rulebook_matches"], context["parsed_objective"])
    context["segment_candidates"] = segments
    return _json(segments)


def load_ml_scores_tool(context_id: str) -> str:
    """Load mock ML channel, timing, conversion, fatigue, and confidence scores for selected segment candidates."""
    context = _context(context_id)
    _ensure_segments(context_id)
    segment_ids = [segment.segment_id for segment in context["segment_candidates"]]
    scores = load_mock_ml_scores(segment_ids)
    context["ml_scores"] = scores
    return _json(scores)


def select_offer_candidates_tool(context_id: str) -> str:
    """Filter deterministic offer catalog candidates for segment candidates and parsed campaign intent."""
    context = _context(context_id)
    _ensure_segments(context_id)
    offers = get_offer_candidates(context["segment_candidates"], context["parsed_objective"])
    context["offer_candidates"] = offers
    return _json(offers)


def assemble_campaign_plan_tool(context_id: str) -> str:
    """Assemble a draft campaign plan from parsed objective, rulebook, segments, ML scores, and offers."""
    from app.graph.nodes import plan_campaign_node

    context = _context(context_id)
    _ensure_offers(context_id)
    _ensure_ml_scores(context_id)
    state = {
        "parsed_objective": context["parsed_objective"],
        "rulebook_matches": context["rulebook_matches"],
        "segment_candidates": context["segment_candidates"],
        "ml_scores": context["ml_scores"],
        "offer_candidates": context["offer_candidates"],
        "warnings": context.get("warnings", []),
        "errors": context.get("errors", []),
        "version": context.get("version", 1),
    }
    state.update(plan_campaign_node(state))
    context["campaign_plan"] = state["campaign_plan"]
    context["selected_segments"] = state["selected_segments"]
    return _json(context["campaign_plan"])


def calculate_projection_tool(context_id: str) -> str:
    """Calculate deterministic projection formulas and attach projection to the campaign plan."""
    context = _context(context_id)
    _ensure_plan(context_id)
    plan: CampaignPlan = context["campaign_plan"]
    plan.projection = estimate_campaign_impact(plan)
    context["campaign_plan"] = plan
    context["projection"] = plan.projection
    return _json(plan.projection)


def generate_content_tool(context_id: str) -> str:
    """Generate approval-required draft content and attach it to the campaign plan."""
    from app.services.llm_service import make_content_drafts

    context = _context(context_id)
    _ensure_plan(context_id)
    plan: CampaignPlan = context["campaign_plan"]
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
    context["campaign_plan"] = plan
    context["content_plan"] = drafts
    return _json(drafts)


def validate_campaign_tool(context_id: str) -> str:
    """Validate campaign plan for rulebook, projection, channel, content, frequency, quiet hours, and export readiness."""
    context = _context(context_id)
    _ensure_content(context_id)
    _ensure_projection(context_id)
    plan: CampaignPlan = context["campaign_plan"]
    plan.validation = validate_campaign_plan(plan)
    context["campaign_plan"] = plan
    context["validation_result"] = plan.validation
    context["warnings"] = list(dict.fromkeys(context.get("warnings", []) + plan.validation.warnings))
    context["errors"] = list(dict.fromkeys(context.get("errors", []) + plan.validation.errors))
    return _json(plan.validation)


def get_final_campaign_plan_tool(context_id: str) -> str:
    """Return the final typed campaign plan after all tool calls for this context_id."""
    context = _context(context_id)
    _ensure_validation(context_id)
    return _json(context["campaign_plan"])


def enrich_campaign_plan_with_deep_agent(campaign_plan: CampaignPlan) -> tuple[CampaignPlan, list[str]]:
    """Use Deep Agents for strategy/copy enrichment, then re-run validation.

    The agent can rewrite narrative and draft copy only. It cannot change rulebook
    eligibility, segments, offers, ML scores, projection, or validation results.
    """
    if not deep_agents_enabled():
        return campaign_plan, ["Deep Agents disabled; deterministic planner output used."]

    try:
        edits = _run_deep_agent(campaign_plan)
    except Exception as exc:
        return campaign_plan, [f"Deep Agent enrichment failed; deterministic output used: {exc.__class__.__name__}."]

    warnings: list[str] = []
    if edits.campaign_summary:
        campaign_plan.summary = edits.campaign_summary

    for recommendation in campaign_plan.recommended_segments:
        explanation = edits.segment_explanations.get(recommendation.segment.segment_id)
        if explanation:
            recommendation.why_this = explanation

    if edits.content_edits:
        _apply_content_edits(campaign_plan, edits.content_edits)

    if edits.agent_notes:
        campaign_plan.assumptions.extend([f"Deep Agent note: {note}" for note in edits.agent_notes])
    campaign_plan.assumptions.append("Strategy and copy were enriched by LangChain Deep Agents; deterministic guardrails were re-validated.")
    campaign_plan.validation = validate_campaign_plan(campaign_plan)
    return campaign_plan, warnings


def _run_deep_agent(campaign_plan: CampaignPlan) -> DeepAgentPlanEdits:
    from deepagents import create_deep_agent

    def get_campaign_context() -> str:
        """Return deterministic campaign context: objective, segments, rulebook, offers, ML scores, projections, and draft copy."""
        return json.dumps(_campaign_context(campaign_plan), indent=2)

    def get_guardrails() -> str:
        """Return non-negotiable campaign MVP guardrails."""
        return (
            "You may change only narrative summary, segment explanations, and draft copy. "
            "Do not change selected segments, rulebook action families, channels, offers, projection formulas, validation, "
            "frequency caps, quiet hours, status, or any launch/send behavior. "
            "All customer-facing content must remain approval_required=true and approved=false. "
            "Do not claim guaranteed savings or imply urgency that is not in the offer."
        )

    model = _chat_model(
        "campaign_deepagents_enrichment",
        temperature=0.2,
        timeout=30,
        max_retries=1,
        metadata={"orchestrator": "deepagents"},
    )
    system_prompt = (
        "You are the Campaign Recommendation Deep Agent. Use the available tools before answering. "
        "Your job is to polish strategy explanation and customer-facing draft copy for a telecom campaign MVP. "
        "Return ONLY valid JSON matching this schema: "
        "{"
        '"campaign_summary": string|null, '
        '"segment_explanations": {"segment_id": "short explanation"}, '
        '"content_edits": [{"segment_id": string, "channel": string, "draft_copy": string, "tone": string, '
        '"why_this_copy": string, "compliance_notes": [string]}], '
        '"agent_notes": [string]'
        "}. "
        "Never wrap the JSON in markdown."
    )
    kwargs = {
        "model": model,
        "tools": [get_campaign_context, get_guardrails],
        "subagents": _subagent_specs(
            [
                {
                    "name": "strategy-reviewer",
                    "description": "Reviews campaign strategy and segment explanations against deterministic rulebook context.",
                    "prompt": (
                        "Review the supplied campaign context. Improve stakeholder-facing explanations without changing "
                        "segments, rulebook logic, offers, scores, or projections."
                    ),
                },
                {
                    "name": "copywriter",
                    "description": "Improves draft telecom campaign copy while preserving compliance guardrails.",
                    "prompt": (
                        "Rewrite draft copy for the requested channels. Keep copy concise, approval-required, and compliant. "
                        "Do not imply that any campaign is launched or sent."
                    ),
                },
            ]
        ),
    }
    kwargs[_deep_agent_prompt_param(create_deep_agent)] = system_prompt
    agent = create_deep_agent(**kwargs)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use get_campaign_context and get_guardrails, then return the JSON edits for this campaign. "
                        "Keep the same segment IDs and channel names."
                    ),
                }
            ]
        },
        config={"run_name": "campaign_deep_agent", "tags": ["campaign_mvp", "deepagents"]},
    )
    text = _message_text(result)
    data = _extract_json_object(text)
    return DeepAgentPlanEdits(**data)


def _campaign_context(campaign_plan: CampaignPlan) -> dict[str, Any]:
    return {
        "campaign_id": campaign_plan.campaign_id,
        "objective": campaign_plan.parsed_objective.model_dump()
        if hasattr(campaign_plan.parsed_objective, "model_dump")
        else campaign_plan.parsed_objective.dict(),
        "campaign_title": campaign_plan.campaign_title,
        "current_summary": campaign_plan.summary,
        "time_window": campaign_plan.time_window,
        "target_metric": campaign_plan.target_metric,
        "target_lift": campaign_plan.target_lift,
        "recommended_segments": [
            {
                "segment_id": rec.segment.segment_id,
                "segment_name": rec.segment.segment_name,
                "customer_count": rec.segment.customer_count,
                "rfm_segment": rec.segment.rfm_segment,
                "data_usage_segment": rec.segment.data_usage_segment,
                "voice_usage_segment": rec.segment.voice_usage_segment,
                "data_usage_trend": rec.segment.data_usage_trend,
                "voice_usage_trend": rec.segment.voice_usage_trend,
                "rulebook_basis": {
                    "trend": rec.rulebook_match.trend,
                    "meaning": rec.rulebook_match.trend_meaning,
                    "typical_action": rec.rulebook_match.typical_action,
                    "allowed_action_families": rec.rulebook_match.allowed_action_families,
                },
                "recommended_action": rec.recommended_action,
                "offer": {
                    "offer_id": rec.offer.offer_id,
                    "offer_name": rec.offer.offer_name,
                    "benefit": rec.offer.benefit,
                    "price": rec.offer.price,
                    "validity_days": rec.offer.validity_days,
                    "description": rec.offer.description,
                },
                "mock_ml": {
                    "best_channel": rec.ml_score.best_channel,
                    "secondary_channel": rec.ml_score.secondary_channel,
                    "best_time_window": rec.ml_score.best_time_window,
                    "expected_conversion": rec.ml_score.expected_conversion,
                    "fatigue_risk": rec.ml_score.fatigue_risk,
                    "model_confidence": rec.ml_score.model_confidence,
                },
                "current_why_this": rec.why_this,
                "opportunity_score": rec.score,
            }
            for rec in campaign_plan.recommended_segments
        ],
        "projection": campaign_plan.projection.model_dump() if campaign_plan.projection and hasattr(campaign_plan.projection, "model_dump") else None,
        "content_plan": [
            draft.model_dump() if hasattr(draft, "model_dump") else draft.dict()
            for draft in campaign_plan.content_plan
        ],
        "risks": campaign_plan.risks,
    }


def _apply_content_edits(campaign_plan: CampaignPlan, edits: list[DeepAgentContentEdit]) -> None:
    by_key = {(draft.segment_id, draft.channel): draft for draft in campaign_plan.content_plan}
    valid_segment_ids = {rec.segment.segment_id for rec in campaign_plan.recommended_segments}
    valid_channels = {(draft.segment_id, draft.channel) for draft in campaign_plan.content_plan}
    for edit in edits:
        key = (edit.segment_id, edit.channel)
        if edit.segment_id not in valid_segment_ids or key not in valid_channels:
            continue
        by_key[key] = ContentDraft(
            segment_id=edit.segment_id,
            channel=edit.channel,
            draft_copy=edit.draft_copy,
            tone=edit.tone,
            language="English",
            approval_required=True,
            approved=False,
            why_this_copy=edit.why_this_copy,
            compliance_notes=edit.compliance_notes
            or ["No guaranteed savings claim", "No misleading urgency", "Approval required before launch"],
        )
    campaign_plan.content_plan = list(by_key.values())


def _message_text(result: Any) -> str:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        return str(result)
    content = getattr(messages[-1], "content", messages[-1].get("content") if isinstance(messages[-1], dict) else str(messages[-1]))
    if isinstance(content, list):
        return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _context(context_id: str) -> dict[str, Any]:
    if context_id not in _RUN_CONTEXTS:
        raise KeyError(f"Unknown campaign Deep Agent context_id: {context_id}")
    return _RUN_CONTEXTS[context_id]


def _json(value: Any) -> str:
    def convert(item: Any) -> Any:
        if hasattr(item, "model_dump"):
            return item.model_dump()
        if isinstance(item, dict):
            return {key: convert(val) for key, val in item.items()}
        if isinstance(item, list):
            return [convert(val) for val in item]
        return item

    return json.dumps(convert(value), indent=2, default=str)


def _ensure_parsed(context_id: str) -> None:
    if "parsed_objective" not in _context(context_id):
        parse_objective_tool(context_id)


def _ensure_rulebook(context_id: str) -> None:
    context = _context(context_id)
    _ensure_parsed(context_id)
    if "rulebook_matches" not in context:
        lookup_rulebook_tool(context_id)


def _ensure_segments(context_id: str) -> None:
    context = _context(context_id)
    _ensure_rulebook(context_id)
    if "segment_candidates" not in context:
        find_segments_tool(context_id)


def _ensure_ml_scores(context_id: str) -> None:
    context = _context(context_id)
    _ensure_segments(context_id)
    if "ml_scores" not in context:
        load_ml_scores_tool(context_id)


def _ensure_offers(context_id: str) -> None:
    context = _context(context_id)
    _ensure_segments(context_id)
    if "offer_candidates" not in context:
        select_offer_candidates_tool(context_id)


def _ensure_plan(context_id: str) -> None:
    context = _context(context_id)
    _ensure_ml_scores(context_id)
    _ensure_offers(context_id)
    if "campaign_plan" not in context:
        assemble_campaign_plan_tool(context_id)


def _ensure_projection(context_id: str) -> None:
    context = _context(context_id)
    _ensure_plan(context_id)
    plan = context["campaign_plan"]
    if not plan.projection:
        calculate_projection_tool(context_id)


def _ensure_content(context_id: str) -> None:
    context = _context(context_id)
    _ensure_plan(context_id)
    plan = context["campaign_plan"]
    if not plan.content_plan:
        generate_content_tool(context_id)


def _ensure_validation(context_id: str) -> None:
    context = _context(context_id)
    _ensure_content(context_id)
    _ensure_projection(context_id)
    plan = context["campaign_plan"]
    if not plan.validation:
        validate_campaign_tool(context_id)


def _deep_agent_prompt_param(create_deep_agent_fn: Any) -> str:
    params = inspect.signature(create_deep_agent_fn).parameters
    return "instructions" if "instructions" in params else "system_prompt"


def _subagent_specs(subagents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from deepagents.types import SubAgent  # type: ignore
    except Exception:
        from deepagents.middleware.subagents import SubAgent  # type: ignore

    annotations = getattr(SubAgent, "__annotations__", {})
    prompt_key = "prompt" if "prompt" in annotations else "system_prompt"
    return [
        {
            "name": item["name"],
            "description": item["description"],
            prompt_key: item["prompt"],
        }
        for item in subagents
    ]
