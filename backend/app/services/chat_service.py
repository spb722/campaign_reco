from __future__ import annotations

import copy
import inspect
import json
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any
from uuid import uuid4

from fastapi.encoders import jsonable_encoder

from app.graph.workflow import run_campaign_workflow
from app.schemas.campaign import CampaignPlan, RecommendedSegment
from app.schemas.chat import ChatResponse, PendingClarification
from app.services.campaign_store import load_campaign_version, next_version, save_campaign_version
from app.services.llm_service import _chat_model, _llm_enabled, _load_runtime_env, make_content_drafts, parse_objective
from app.tools.export_tool import generate_one_pager_pdf, save_campaign_json
from app.tools.offer_tool import get_next_best_offer_candidates
from app.tools.projection_tool import estimate_campaign_impact
from app.tools.validation_tool import validate_campaign_plan


CAMPAIGN_KEYWORDS = {
    "arpu",
    "revenue",
    "churn",
    "retention",
    "save",
    "data",
    "usage",
    "upsell",
    "cross-sell",
    "cross sell",
    "inactive",
    "dormant",
    "reactivate",
    "winback",
    "activity",
    "engagement",
    "campaign",
    "offer",
    "segment",
}
TIME_KEYWORDS = {"day", "days", "week", "weeks", "month", "months", "quarter"}
GREETINGS = {"hi", "hello", "hey", "hai", "good morning", "good afternoon", "good evening"}
SUPPORTED_CHANNELS = {"sms", "whatsapp", "push", "email", "ivr", "outbound_call"}
CHANNEL_ORDER = ["whatsapp", "push", "sms", "email", "ivr", "outbound_call"]


@dataclass
class ChatSession:
    session_id: str
    current_campaign_id: str | None = None
    pending_clarification: PendingClarification | None = None
    last_referenced_segment_id: str | None = None
    last_suggested_offer_id: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)


_CHAT_SESSIONS: dict[str, ChatSession] = {}
_CHAT_AGENT_CONTEXTS: dict[str, dict[str, Any]] = {}


def handle_chat_message(session_id: str, message: str, campaign_id: str | None, request_id: str) -> ChatResponse:
    session = _CHAT_SESSIONS.setdefault(session_id, ChatSession(session_id=session_id))
    text = message.strip()
    if campaign_id and load_campaign_version(campaign_id):
        session.current_campaign_id = campaign_id
    session.messages.append({"role": "user", "content": text})

    if chat_deep_agent_enabled():
        try:
            return _run_chat_deep_agent(session, text, request_id)
        except Exception as exc:
            response = _handle_chat_message_deterministic(session, text, request_id)
            response.warnings = list(
                dict.fromkeys([f"Chat Deep Agent failed; deterministic chat fallback used: {exc.__class__.__name__}."] + response.warnings)
            )
            return response

    return _handle_chat_message_deterministic(session, text, request_id)


def chat_deep_agent_enabled() -> bool:
    _load_runtime_env()
    if not _llm_enabled():
        return False
    if os.getenv("CAMPAIGN_DEEP_AGENTS_ENABLED", "true").lower() in {"0", "false", "no"}:
        return False
    return True


def _handle_chat_message_deterministic(session: ChatSession, text: str, request_id: str) -> ChatResponse:
    lower = text.lower()

    if _is_greeting(lower):
        return _remember(
            session,
            ChatResponse(
                request_id=request_id,
                response_type="conversation",
                message="Hi. What campaign objective would you like to work on? For example, reduce churn by 10% next quarter or increase ARPU for prepaid customers.",
                data=None,
            ),
        )

    if session.pending_clarification:
        combined = _combine_pending(session.pending_clarification, text)
        session.pending_clarification = None
        return _run_new_campaign(session, combined, request_id)

    if _is_export_request(lower):
        return _export_current_campaign(session, request_id)

    if _is_modification_request(lower):
        return _modify_current_campaign(session, text, request_id)

    if session.current_campaign_id and _is_followup_question(lower):
        return _answer_from_current_campaign(session, text, request_id)

    if _looks_like_campaign_request(lower):
        parsed = parse_objective(text)
        missing = _missing_fields(lower, parsed)
        if missing:
            pending = PendingClarification(
                missing_fields=missing,
                partial_objective=jsonable_encoder(parsed),
                original_message=text,
            )
            session.pending_clarification = pending
            return _remember(
                session,
                ChatResponse(
                    request_id=request_id,
                    response_type="clarification",
                    message=_clarification_message(missing, parsed.campaign_intent),
                    data=None,
                    pending_clarification=pending,
                    ui_action={"set_active_view": "chat"},
                ),
            )
        return _run_new_campaign(session, text, request_id)

    return _remember(
        session,
        ChatResponse(
            request_id=request_id,
            response_type="conversation",
            message="I can help create or explain a campaign plan. Please share a campaign objective, such as 'reduce churn by 10% next quarter' or 'increase data usage this month'.",
            data=None,
        ),
    )


def _run_chat_deep_agent(session: ChatSession, message: str, request_id: str) -> ChatResponse:
    from deepagents import create_deep_agent

    context_id = f"chat_run_{uuid4().hex}"
    _CHAT_AGENT_CONTEXTS[context_id] = {
        "session": session,
        "message": message,
        "request_id": request_id,
    }
    model = _chat_model(
        "chat_deep_agent",
        temperature=0.1,
        timeout=30,
        max_retries=1,
        metadata={"orchestrator": "chat_deep_agent"},
    )
    system_prompt = (
        "You are the user-facing Campaign Chat Deep Agent. Decide which tool to call for each user message. "
        "Do not create campaign plans or answers from free text yourself; call exactly one primary tool, then call get_final_chat_response_tool. "
        "Use reply_conversation_tool for greetings or general conversation. "
        "Use create_campaign_plan_tool for campaign requests or when the user answers a pending clarification. "
        "Use ask_clarification_tool for vague campaign-like requests that lack a clear objective or time window. "
        "Use answer_from_campaign_tool for questions about the current campaign. "
        "Use modify_campaign_tool for requests to change/regenerate/rewrite copy or plan fields such as duration, target lift, channel, or content tone. "
        "Use export_campaign_tool for PDF, one-pager, export, or download requests. "
        "The tools return the exact API response that the UI will consume."
    )
    kwargs = {
        "model": model,
        "tools": [
            reply_conversation_tool,
            ask_clarification_tool,
            create_campaign_plan_tool,
            answer_from_campaign_tool,
            modify_campaign_tool,
            export_campaign_tool,
            get_final_chat_response_tool,
        ],
    }
    kwargs[_deep_agent_prompt_param(create_deep_agent)] = system_prompt
    agent = create_deep_agent(**kwargs)
    pending = session.pending_clarification.model_dump() if session.pending_clarification else None
    agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"context_id={context_id}\n"
                        f"session_id={session.session_id}\n"
                        f"current_campaign_id={session.current_campaign_id or 'none'}\n"
                        f"pending_clarification={json.dumps(pending, default=str)}\n"
                        f"user_message={message}\n\n"
                        "Choose the right tool. If pending_clarification is present and the user appears to answer it, use create_campaign_plan_tool."
                    ),
                }
            ]
        },
        config={"run_name": "chat_deep_agent_orchestrator", "tags": ["campaign_mvp", "chat_deep_agent", "tool_flow"]},
    )
    response = _CHAT_AGENT_CONTEXTS[context_id].get("response")
    if not isinstance(response, ChatResponse):
        raise RuntimeError("Chat Deep Agent did not produce a ChatResponse.")
    return response


def reply_conversation_tool(context_id: str) -> str:
    """Return a conversational response without creating or changing a campaign."""
    context = _chat_agent_context(context_id)
    session: ChatSession = context["session"]
    text = context["message"].strip()
    lower = text.lower()
    if _is_greeting(lower):
        message = "Hi. What campaign objective would you like to work on? For example, reduce churn by 10% next quarter or increase ARPU for prepaid customers."
    else:
        message = "I can help create, explain, modify, or export a campaign plan. Please share a campaign objective or ask a question about the current campaign."
    response = _remember(
        session,
        ChatResponse(
            request_id=context["request_id"],
            response_type="conversation",
            message=message,
            data=None,
        ),
    )
    context["response"] = response
    return _json_response(response)


def ask_clarification_tool(context_id: str) -> str:
    """Ask for missing campaign details before running the campaign planner."""
    context = _chat_agent_context(context_id)
    session: ChatSession = context["session"]
    text = context["message"].strip()
    parsed = parse_objective(text)
    missing = _missing_fields(text.lower(), parsed) or ["campaign_intent"]
    pending = PendingClarification(
        missing_fields=missing,
        partial_objective=jsonable_encoder(parsed),
        original_message=text,
    )
    session.pending_clarification = pending
    response = _remember(
        session,
        ChatResponse(
            request_id=context["request_id"],
            response_type="clarification",
            message=_clarification_message(missing, parsed.campaign_intent),
            data=None,
            pending_clarification=pending,
            ui_action={"set_active_view": "chat"},
        ),
    )
    context["response"] = response
    return _json_response(response)


def create_campaign_plan_tool(context_id: str) -> str:
    """Create a campaign plan, or return clarification if required campaign fields are missing."""
    context = _chat_agent_context(context_id)
    session: ChatSession = context["session"]
    text = context["message"].strip()
    if session.pending_clarification:
        prompt = _combine_pending(session.pending_clarification, text)
        session.pending_clarification = None
    else:
        prompt = text
    parsed = parse_objective(prompt)
    missing = _missing_fields(prompt.lower(), parsed)
    if missing:
        pending = PendingClarification(
            missing_fields=missing,
            partial_objective=jsonable_encoder(parsed),
            original_message=prompt,
        )
        session.pending_clarification = pending
        response = _remember(
            session,
            ChatResponse(
                request_id=context["request_id"],
                response_type="clarification",
                message=_clarification_message(missing, parsed.campaign_intent),
                data=None,
                pending_clarification=pending,
                ui_action={"set_active_view": "chat"},
            ),
        )
    else:
        response = _run_new_campaign(session, prompt, context["request_id"])
    context["response"] = response
    return _json_response(response)


def answer_from_campaign_tool(context_id: str) -> str:
    """Answer a user question from the current saved campaign plan."""
    context = _chat_agent_context(context_id)
    response = _answer_from_current_campaign(context["session"], context["message"], context["request_id"])
    context["response"] = response
    return _json_response(response)


def modify_campaign_tool(context_id: str) -> str:
    """Modify the current campaign, usually content or summary, without changing deterministic business logic."""
    context = _chat_agent_context(context_id)
    response = _modify_current_campaign(context["session"], context["message"], context["request_id"])
    context["response"] = response
    return _json_response(response)


def export_campaign_tool(context_id: str) -> str:
    """Export the current campaign one-pager and JSON."""
    context = _chat_agent_context(context_id)
    response = _export_current_campaign(context["session"], context["request_id"])
    context["response"] = response
    return _json_response(response)


def get_final_chat_response_tool(context_id: str) -> str:
    """Return the final API ChatResponse created by the selected chat tool."""
    context = _chat_agent_context(context_id)
    response = context.get("response")
    if not isinstance(response, ChatResponse):
        response = _remember(
            context["session"],
            ChatResponse(
                request_id=context["request_id"],
                response_type="conversation",
                message="I can help create or explain a campaign plan. Please share a campaign objective.",
                data=None,
            ),
        )
        context["response"] = response
    return _json_response(response)


def _run_new_campaign(session: ChatSession, prompt: str, request_id: str) -> ChatResponse:
    state = run_campaign_workflow(prompt)
    plan: CampaignPlan = state["campaign_plan"]
    session.current_campaign_id = plan.campaign_id
    message = f"I created a draft {plan.campaign_title.lower()} for {plan.time_window}."
    return _remember(
        session,
        ChatResponse(
            request_id=request_id,
            response_type="campaign_plan",
            message=message,
            data={"campaign_plan": jsonable_encoder(plan)},
            ui_action={"set_active_view": "recommended_segments"},
            warnings=state.get("warnings", []),
            errors=state.get("errors", []),
            success=not bool(state.get("errors")),
        ),
    )


def _answer_from_current_campaign(session: ChatSession, message: str, request_id: str) -> ChatResponse:
    plan = _current_plan(session)
    if not plan:
        return _no_current_campaign(session, request_id)

    lower = message.lower()
    rec = plan.recommended_segments[0] if plan.recommended_segments else None
    ui_action: dict[str, Any] = {"set_active_view": "campaign_plan"}
    if rec:
        ui_action["highlight_segment_id"] = rec.segment.segment_id

    if _is_next_best_offer_question(lower):
        segment_match = _find_segment_for_question(plan, message)
        if isinstance(segment_match, list):
            names = ", ".join(f"{item.segment.segment_id} ({item.segment.segment_name})" for item in segment_match[:5])
            answer = f"I found multiple matching segments: {names}. Please specify the full segment ID before I compare alternate offers."
            ui_action["set_active_view"] = "recommended_segments"
        else:
            selected = segment_match or rec
            if selected:
                answer = _next_best_offer_answer(plan, selected, session)
                ui_action["set_active_view"] = "segment_drilldown"
                ui_action["highlight_segment_id"] = selected.segment.segment_id
            else:
                answer = "I could not find a matching segment in the current campaign plan, so I cannot compare alternate offers yet."
                ui_action["set_active_view"] = "recommended_segments"
    elif _is_segment_profile_question(lower):
        segment_match = _find_segment_for_question(plan, message)
        if isinstance(segment_match, list):
            names = ", ".join(f"{item.segment.segment_id} ({item.segment.segment_name})" for item in segment_match[:5])
            answer = f"I found multiple matching segments: {names}. Please specify the full segment ID."
            ui_action["set_active_view"] = "recommended_segments"
        else:
            selected = segment_match or rec
            if selected:
                session.last_referenced_segment_id = selected.segment.segment_id
                answer = _segment_profile_answer(plan, selected, message)
                ui_action["set_active_view"] = "segment_drilldown"
                ui_action["highlight_segment_id"] = selected.segment.segment_id
            else:
                answer = "I could not find a matching segment in the current campaign plan."
                ui_action["set_active_view"] = "recommended_segments"
    elif "why" in lower and rec:
        answer = (
            f"{rec.segment.customer_signal or rec.segment.segment_name} was selected because the rulebook maps "
            f"{rec.rulebook_match.trend} to {rec.rulebook_match.typical_action}, the segment has "
            f"{rec.segment.customer_count:,} eligible users, and {rec.ml_score.best_channel} has the strongest channel score."
        )
        ui_action["set_active_view"] = "segment_drilldown"
    elif "projection" in lower or "impact" in lower or "formula" in lower:
        projection = plan.projection
        answer = (
            f"The projection uses this deterministic formula: {projection.formula}. "
            f"The current total projected impact is {projection.total_projected_impact:,.2f} {projection.unit}."
            if projection
            else "Projection is not available yet for this campaign."
        )
        ui_action["set_active_view"] = "assumptions_validation"
    elif "channel" in lower and rec:
        answer = (
            f"The primary channel for the first segment is {rec.ml_score.best_channel}, with "
            f"{rec.ml_score.secondary_channel} as secondary. The recommended time window is {rec.ml_score.best_time_window}."
        )
        ui_action["set_active_view"] = "segment_drilldown"
    elif "offer" in lower and rec:
        answer = (
            f"The first recommended offer is {rec.offer.offer_name}: {rec.offer.benefit}, valid for "
            f"{rec.offer.validity_days} days. It was selected from the local offer catalog."
        )
        ui_action["set_active_view"] = "segment_drilldown"
    else:
        answer = plan.summary

    return _remember(
        session,
        ChatResponse(
            request_id=request_id,
            response_type="answer",
            message=answer,
            data={"campaign_plan": jsonable_encoder(plan)},
            ui_action=ui_action,
            warnings=plan.validation.warnings if plan.validation else [],
        ),
    )


def _is_segment_profile_question(text: str) -> bool:
    return (
        "segment" in text
        and any(term in text for term in ("profile", "description", "describe", "what kind", "what type", "people", "customers", "more about", "tell me about"))
    )


def _is_next_best_offer_question(text: str) -> bool:
    return "offer" in text and any(term in text for term in ("next best", "alternate", "alternative", "another", "other option", "second best"))


def _find_segment_for_question(plan: CampaignPlan, message: str) -> RecommendedSegment | list[RecommendedSegment] | None:
    tokens = _segment_reference_tokens(message)
    if tokens:
        matches = [
            rec
            for rec in plan.recommended_segments
            if any(_segment_token_matches(rec, token) for token in tokens)
        ]
        unique = {rec.segment.segment_id: rec for rec in matches}
        if len(unique) == 1:
            return next(iter(unique.values()))
        if len(unique) > 1:
            return list(unique.values())

    lower = message.lower()
    name_matches = [
        rec
        for rec in plan.recommended_segments
        if rec.segment.segment_name.lower() in lower
        or (rec.segment.customer_signal and rec.segment.customer_signal.lower() in lower)
    ]
    if len(name_matches) == 1:
        return name_matches[0]
    if len(name_matches) > 1:
        return name_matches
    return None


def _segment_reference_tokens(message: str) -> list[str]:
    lower = message.lower()
    tokens = re.findall(r"\b(?:segment|seg)[-\s_]*([a-z0-9-]+)\b", lower)
    tokens.extend(re.findall(r"\bseg[-_]?(\d+)\b", lower))
    return list(dict.fromkeys(token.strip("-_") for token in tokens if len(token.strip("-_")) >= 2))


def _segment_token_matches(rec: RecommendedSegment, token: str) -> bool:
    segment_id = rec.segment.segment_id.lower()
    compact_id = re.sub(r"[^a-z0-9]", "", segment_id)
    compact_token = re.sub(r"[^a-z0-9]", "", token.lower())
    return compact_token == compact_id or compact_token in compact_id


def _segment_profile_answer(plan: CampaignPlan, rec: RecommendedSegment, question: str) -> str:
    if _llm_enabled():
        try:
            return _llm_segment_profile_answer(plan, rec, question)
        except Exception:
            pass
    return _template_segment_profile_answer(rec)


def _next_best_offer_answer(plan: CampaignPlan, rec: RecommendedSegment, session: ChatSession) -> str:
    candidates = get_next_best_offer_candidates(rec.segment, plan.parsed_objective, rec.offer.offer_id)
    if not candidates:
        return (
            f"The current offer for {rec.segment.segment_id} is {rec.offer.offer_name}. "
            "I could not find another eligible offer for this segment in the current offer catalog."
        )

    next_offer = candidates[0]
    session.last_referenced_segment_id = rec.segment.segment_id
    session.last_suggested_offer_id = next_offer.offer_id
    if _llm_enabled():
        try:
            return _llm_next_best_offer_answer(plan, rec, next_offer, candidates[:3])
        except Exception:
            pass
    return _template_next_best_offer_answer(rec, next_offer, len(candidates))


def _llm_next_best_offer_answer(plan: CampaignPlan, rec: RecommendedSegment, next_offer, top_candidates: list) -> str:
    model = _chat_model(
        "campaign_next_best_offer_answer",
        temperature=0.2,
        timeout=20,
        max_retries=1,
        metadata={"orchestrator": "chat_answer_tool"},
    )
    context = {
        "campaign_intent": plan.campaign_intent,
        "segment": jsonable_encoder(rec.segment),
        "current_offer": jsonable_encoder(rec.offer),
        "next_best_offer": jsonable_encoder(next_offer),
        "other_top_alternates": [jsonable_encoder(offer) for offer in top_candidates[1:]],
    }
    system = (
        "Explain the next best offer for a telecom campaign segment. "
        "Use only the supplied JSON facts. Do not invent offers, prices, eligibility, or lift values. "
        "Mention that the current offer is excluded from the alternate ranking. Keep the answer concise."
    )
    result = model.with_config(
        run_name="answer_next_best_offer",
        tags=["campaign_mvp", "chat_answer", "next_best_offer", rec.segment.segment_id],
    ).invoke([("system", system), ("user", json.dumps(context, indent=2, default=str))])
    return _message_content(result) or _template_next_best_offer_answer(rec, next_offer, len(top_candidates))


def _template_next_best_offer_answer(rec: RecommendedSegment, next_offer, candidate_count: int) -> str:
    return (
        f"The current offer for {rec.segment.segment_id} is {rec.offer.offer_name}. "
        f"Excluding that, the next best eligible offer in the catalog is {next_offer.offer_name}. "
        f"It provides {next_offer.benefit}, is valid for {next_offer.validity_days} days, costs OMR {next_offer.price:.2f}, "
        f"and has estimated ARPU lift of OMR {next_offer.estimated_arpu_lift:.2f}, estimated data lift of {next_offer.estimated_data_lift_gb:.2f} GB, "
        f"cost per user of OMR {next_offer.cost_per_user:.2f}, and {next_offer.margin_impact} margin impact. "
        f"I found {candidate_count} alternate eligible offer{'s' if candidate_count != 1 else ''} for this segment."
    )


def _llm_segment_profile_answer(plan: CampaignPlan, rec: RecommendedSegment, question: str) -> str:
    model = _chat_model(
        "campaign_segment_profile_answer",
        temperature=0.2,
        timeout=20,
        max_retries=1,
        metadata={"orchestrator": "chat_answer_tool"},
    )
    context = {
        "user_question": question,
        "campaign_title": plan.campaign_title,
        "campaign_intent": plan.campaign_intent,
        "segment": jsonable_encoder(rec.segment),
        "rulebook_match": jsonable_encoder(rec.rulebook_match),
        "recommended_action": rec.recommended_action,
        "offer": jsonable_encoder(rec.offer),
        "ml_score": jsonable_encoder(rec.ml_score),
        "projected_impact": rec.projected_impact,
        "confidence": rec.confidence,
        "why_this": rec.why_this,
    }
    system = (
        "Explain a telecom campaign segment in simple business language. "
        "Use only the provided JSON facts. Do not invent demographics, income, geography, age, or behavior not present in the data. "
        "If a detail is not present, do not mention it. Keep the answer concise and useful for a campaign manager."
    )
    user = f"Segment context JSON:\n{json.dumps(context, indent=2, default=str)}"
    result = model.with_config(
        run_name="answer_segment_profile",
        tags=["campaign_mvp", "chat_answer", "segment_profile", rec.segment.segment_id],
    ).invoke([("system", system), ("user", user)])
    return _message_content(result) or _template_segment_profile_answer(rec)


def _template_segment_profile_answer(rec: RecommendedSegment) -> str:
    segment = rec.segment
    signal = segment.customer_signal or segment.segment_name
    meaning = segment.customer_meaning or "No extra signal description is available"
    return (
        f"{segment.segment_id} is {segment.segment_name}. These are {segment.rfm_segment} prepaid customers with "
        f"{segment.data_usage_segment} data usage and {segment.voice_usage_segment} voice usage. "
        f"The customer signal is {signal}: {meaning}. The segment has {segment.customer_count:,} customers, "
        f"average ARPU of OMR {segment.avg_arpu:.2f}, about {segment.avg_data_gb:.2f} GB data usage, and "
        f"{segment.avg_voice_min:.0f} voice minutes. The recommended action is {rec.recommended_action} because "
        f"the rulebook maps {rec.rulebook_match.trend} to {rec.rulebook_match.typical_action}."
    )


def _message_content(result: Any) -> str:
    content = getattr(result, "content", "")
    if isinstance(content, list):
        return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content).strip()
    return str(content).strip()


def _modify_current_campaign(session: ChatSession, instruction: str, request_id: str) -> ChatResponse:
    plan = _current_plan(session)
    if not plan:
        return _no_current_campaign(session, request_id)

    updated = copy.deepcopy(plan)
    updated.version = next_version(updated)
    lower = instruction.lower()
    structured_changes = _apply_offer_change(updated, instruction, session)
    structured_changes.extend(_apply_structured_plan_changes(updated, instruction))
    if structured_changes:
        changed = structured_changes
        if "channel_plan" in changed or "content_plan" in changed or "offer" in changed:
            updated.content_plan = _regenerate_all_content(updated, instruction if "content_plan" in changed else None)
            if "content_plan" not in changed:
                changed.append("content_plan")
        updated.projection = estimate_campaign_impact(updated)
        updated.validation = validate_campaign_plan(updated)
        if "projection" not in changed:
            changed.append("projection")
        if "validation" not in changed:
            changed.append("validation")
        message = _modification_message(changed, updated)
        active_view = "segment_drilldown" if "offer" in changed else "assumptions_validation" if "projection" in changed else "campaign_plan"
    elif any(term in lower for term in ["copy", "sms", "whatsapp", "push", "shorter", "formal", "conversational", "premium"]):
        updated.content_plan = _regenerate_all_content(updated, instruction)
        changed = ["content_plan"]
        message = "I updated the draft content. Segment selection, rulebook logic, offers, channels, and projections were unchanged."
        active_view = "content_drafts"
        updated.validation = validate_campaign_plan(updated)
    else:
        updated.summary = f"{updated.summary} Update note: {instruction}"
        changed = ["summary"]
        message = "I updated the campaign summary while keeping the deterministic plan unchanged."
        active_view = "campaign_plan"
        updated.validation = validate_campaign_plan(updated)

    save_campaign_version(updated)
    session.current_campaign_id = updated.campaign_id
    return _remember(
        session,
        ChatResponse(
            request_id=request_id,
            response_type="plan_updated",
            message=message,
            data={"campaign_plan": jsonable_encoder(updated)},
            ui_action={"set_active_view": active_view, "changed_sections": changed, **_highlight_for_changes(updated, session, changed)},
            warnings=updated.validation.warnings if updated.validation else [],
            errors=updated.validation.errors if updated.validation else [],
            success=updated.validation.is_valid if updated.validation else True,
        ),
    )


def _apply_offer_change(plan: CampaignPlan, instruction: str, session: ChatSession) -> list[str]:
    if not _is_offer_replacement_request(instruction):
        return []
    target_rec = _target_segment_for_offer_change(plan, instruction, session)
    if not target_rec:
        return []

    candidates = get_next_best_offer_candidates(target_rec.segment, plan.parsed_objective, target_rec.offer.offer_id)
    replacement = _resolve_replacement_offer(instruction, candidates, session.last_suggested_offer_id)
    if not replacement or replacement.offer_id == target_rec.offer.offer_id:
        return []

    old_offer_name = target_rec.offer.offer_name
    target_rec.offer = replacement
    target_rec.why_this = (
        f"User replaced {old_offer_name} with {replacement.offer_name}. "
        f"The replacement is eligible for {target_rec.segment.segment_name} and has estimated ARPU lift of OMR {replacement.estimated_arpu_lift:.2f}."
    )
    for tactic in plan.campaign_tactics:
        if tactic.get("segment_id") == target_rec.segment.segment_id:
            tactic["offer_id"] = replacement.offer_id
            tactic["offer_name"] = replacement.offer_name
            tactic["why_this"] = f"{replacement.description} User-selected replacement for {old_offer_name}."
    plan.summary = _updated_summary(plan, instruction)
    plan.assumptions = list(dict.fromkeys(plan.assumptions + [f"Offer for {target_rec.segment.segment_id} changed from {old_offer_name} to {replacement.offer_name} by user request."]))
    session.last_referenced_segment_id = target_rec.segment.segment_id
    session.last_suggested_offer_id = replacement.offer_id
    return ["offer", "recommended_segments", "campaign_tactics", "summary", "assumptions"]


def _is_offer_replacement_request(text: str) -> bool:
    lower = text.lower()
    return "offer" in lower or "pack" in lower or any(term in lower for term in ("rather than", "instead of", "replace", "switch", "select", "give"))


def _target_segment_for_offer_change(plan: CampaignPlan, instruction: str, session: ChatSession) -> RecommendedSegment | None:
    segment_match = _find_segment_for_question(plan, instruction)
    if isinstance(segment_match, RecommendedSegment):
        return segment_match
    if session.last_referenced_segment_id:
        for rec in plan.recommended_segments:
            if rec.segment.segment_id == session.last_referenced_segment_id:
                return rec
    return plan.recommended_segments[0] if plan.recommended_segments else None


def _resolve_replacement_offer(instruction: str, candidates: list, last_suggested_offer_id: str | None):
    if not candidates:
        return None
    lower = instruction.lower()
    if last_suggested_offer_id and any(term in lower for term in ("that", "this", "next best", "suggested")):
        for offer in candidates:
            if offer.offer_id == last_suggested_offer_id:
                return offer
    scored = sorted(
        ((offer, _offer_name_match_score(instruction, offer.offer_name)) for offer in candidates),
        key=lambda item: item[1],
        reverse=True,
    )
    if scored and scored[0][1] >= 0.42:
        return scored[0][0]
    if last_suggested_offer_id:
        for offer in candidates:
            if offer.offer_id == last_suggested_offer_id:
                return offer
    return None


def _offer_name_match_score(instruction: str, offer_name: str) -> float:
    left = re.sub(r"[^a-z0-9]+", " ", instruction.lower()).strip()
    right = re.sub(r"[^a-z0-9]+", " ", offer_name.lower()).strip()
    right_tokens = [token for token in right.split() if len(token) > 2]
    token_overlap = sum(1 for token in right_tokens if token in left) / max(len(right_tokens), 1)
    ratio = SequenceMatcher(None, left, right).ratio()
    return max(token_overlap, ratio)


def _highlight_for_changes(plan: CampaignPlan, session: ChatSession, changed: list[str]) -> dict[str, str]:
    if "offer" not in changed or not session.last_referenced_segment_id:
        return {}
    return {"highlight_segment_id": session.last_referenced_segment_id}


def _apply_structured_plan_changes(plan: CampaignPlan, instruction: str) -> list[str]:
    changed: list[str] = []

    duration_days = _extract_duration_days(instruction)
    if duration_days and duration_days != _campaign_days(plan):
        plan.parsed_objective.time_window_value = duration_days
        plan.parsed_objective.time_window_unit = "days"
        plan.time_window = f"{duration_days} days"
        _refresh_followup_plan(plan, duration_days)
        changed.extend(["parsed_objective", "time_window", "followup_plan", "summary"])

    target_lift = _extract_target_lift_percent(instruction)
    if target_lift is not None and target_lift != plan.parsed_objective.target_lift_value:
        old_lift = plan.parsed_objective.target_lift_value
        plan.parsed_objective.target_lift_value = target_lift
        plan.parsed_objective.target_lift_unit = "percent"
        plan.target_lift = f"{target_lift:g}%"
        _scale_offer_lifts(plan, old_lift, target_lift)
        changed.extend(["parsed_objective", "target_lift", "recommended_segments", "campaign_tactics", "summary"])

    channel = _extract_requested_channel(instruction)
    if channel:
        _apply_channel_override(plan, channel)
        changed.extend(["channel_plan", "recommended_segments", "assumptions", "summary"])

    if changed:
        plan.summary = _updated_summary(plan, instruction)
        plan.assumptions = list(dict.fromkeys(plan.assumptions + ["User-requested plan modification applied; deterministic projection and validation were rerun."]))
    return list(dict.fromkeys(changed))


def _extract_duration_days(text: str) -> int | None:
    lower = text.lower()
    matches = re.findall(r"(\d+)\s*(day|days|week|weeks|month|months)", lower)
    if matches:
        value, unit = matches[-1]
        number = int(value)
        if unit.startswith("week"):
            return number * 7
        if unit.startswith("month"):
            return number * 30
        return number
    if "quarter" in lower:
        return 90
    return None


def _extract_target_lift_percent(text: str) -> float | None:
    to_matches = re.findall(r"\bto\s+(\d+(?:\.\d+)?)\s*%", text, flags=re.IGNORECASE)
    if to_matches:
        return float(to_matches[-1])
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
    if not matches:
        return None
    return float(matches[-1])


def _extract_requested_channel(text: str) -> str | None:
    plain = text.lower()
    if any(term in plain for term in ("copy", "shorter", "formal", "conversational", "rewrite", "tone")):
        return None
    lower = plain.replace(" ", "_")
    for pattern in (r"\buse\s+([a-z_]+)", r"\bto\s+([a-z_]+)", r"\bprimary channel\s+([a-z_]+)"):
        match = re.search(pattern, lower)
        if match and match.group(1) in SUPPORTED_CHANNELS:
            return match.group(1)
    for channel in CHANNEL_ORDER:
        if channel in lower:
            return channel
    return None


def _campaign_days(plan: CampaignPlan) -> int:
    value = plan.parsed_objective.time_window_value
    unit = plan.parsed_objective.time_window_unit
    if not value:
        match = re.search(r"(\d+)", plan.time_window or "")
        return int(match.group(1)) if match else 30
    if unit == "weeks":
        return value * 7
    if unit == "months":
        return value * 30
    if unit == "quarter":
        return 90
    return int(value)


def _scale_offer_lifts(plan: CampaignPlan, old_lift: float | None, new_lift: float) -> None:
    if not old_lift or old_lift <= 0:
        return
    factor = new_lift / old_lift
    seen_offer_objects: set[int] = set()
    for rec in plan.recommended_segments:
        object_id = id(rec.offer)
        if object_id in seen_offer_objects:
            continue
        seen_offer_objects.add(object_id)
        if plan.campaign_intent in {"increase_arpu", "upsell", "cross_sell", "recommend_best_campaign"}:
            rec.offer.estimated_arpu_lift = round(rec.offer.estimated_arpu_lift * factor, 4)
        elif plan.campaign_intent == "increase_data_usage":
            rec.offer.estimated_data_lift_gb = round(rec.offer.estimated_data_lift_gb * factor, 4)
        elif plan.campaign_intent == "reduce_churn":
            rec.offer.estimated_save_rate = min(round(rec.offer.estimated_save_rate * factor, 4), 1.0)


def _apply_channel_override(plan: CampaignPlan, channel: str) -> None:
    for rec in plan.recommended_segments:
        previous_primary = rec.ml_score.best_channel
        if previous_primary != channel:
            rec.ml_score.best_channel = channel
            if rec.ml_score.secondary_channel == channel:
                rec.ml_score.secondary_channel = previous_primary
    for item in plan.channel_plan:
        previous_primary = item.primary_channel
        if previous_primary != channel:
            item.primary_channel = channel
            item.score_source = "user_override"
            if item.secondary_channel == channel:
                item.secondary_channel = previous_primary
    plan.assumptions = list(dict.fromkeys(plan.assumptions + [f"Primary channel overridden by user to {channel}."]))


def _refresh_followup_plan(plan: CampaignPlan, duration_days: int) -> None:
    review_day = max(7, min(14, duration_days // 3))
    final_day = max(review_day + 1, min(duration_days - 3, 21 if duration_days <= 30 else duration_days - 7))
    refreshed = []
    for rec in plan.recommended_segments:
        primary = rec.ml_score.best_channel
        secondary = rec.ml_score.secondary_channel
        refreshed.append(
            {
                "segment_id": rec.segment.segment_id,
                "steps": [
                    f"Day 0: {primary} primary message",
                    f"Day 3: {secondary} reminder if no conversion",
                    f"Day {review_day}: suppress converted users and review fatigue",
                    f"Day {final_day}: final low-frequency reminder within cap",
                ],
            }
        )
    plan.followup_plan = refreshed


def _regenerate_all_content(plan: CampaignPlan, instruction: str | None = None):
    drafts = []
    for rec in plan.recommended_segments:
        drafts.extend(
            make_content_drafts(
                rec.segment,
                rec.offer,
                rec.ml_score.best_channel,
                rec.ml_score.secondary_channel,
                instruction,
            )
        )
    return drafts


def _updated_summary(plan: CampaignPlan, instruction: str) -> str:
    return (
        f"Updated plan for {plan.campaign_title}: target {plan.target_lift} over {plan.time_window}. "
        f"The latest user instruction was: {instruction}"
    )


def _modification_message(changed: list[str], plan: CampaignPlan) -> str:
    parts = []
    if "offer" in changed:
        changed_rec = next((rec for rec in plan.recommended_segments if rec.segment.segment_id), None)
        if changed_rec:
            parts.append(f"the selected offer to {changed_rec.offer.offer_name}")
    if "time_window" in changed:
        parts.append(f"duration to {plan.time_window}")
    if "target_lift" in changed:
        parts.append(f"target lift to {plan.target_lift}")
    if "channel_plan" in changed and plan.recommended_segments:
        parts.append(f"primary channel to {plan.recommended_segments[0].ml_score.best_channel}")
    changed_text = ", ".join(parts) if parts else "the requested fields"
    return f"I updated {changed_text} and recalculated the affected campaign plan sections."


def _export_current_campaign(session: ChatSession, request_id: str) -> ChatResponse:
    plan = _current_plan(session)
    if not plan:
        return _no_current_campaign(session, request_id)
    plan.validation = validate_campaign_plan(plan)
    if not plan.validation.is_valid:
        return _remember(
            session,
            ChatResponse(
                request_id=request_id,
                response_type="error",
                success=False,
                message="Export is blocked because validation has errors.",
                data={"validation": jsonable_encoder(plan.validation)},
                warnings=plan.validation.warnings,
                errors=plan.validation.errors,
            ),
        )
    pdf_path = generate_one_pager_pdf(plan)
    json_path = save_campaign_json(plan)
    plan.export_path = pdf_path
    save_campaign_version(plan)
    return _remember(
        session,
        ChatResponse(
            request_id=request_id,
            response_type="export_ready",
            message="The one-page PDF is ready.",
            data={
                "campaign_id": plan.campaign_id,
                "pdf_path": pdf_path,
                "json_path": json_path,
                "download_path": f"/campaign/{plan.campaign_id}/download",
            },
            ui_action={"set_active_view": "export"},
            warnings=plan.validation.warnings,
        ),
    )


def _current_plan(session: ChatSession) -> CampaignPlan | None:
    if not session.current_campaign_id:
        return None
    return load_campaign_version(session.current_campaign_id)


def _no_current_campaign(session: ChatSession, request_id: str) -> ChatResponse:
    return _remember(
        session,
        ChatResponse(
            request_id=request_id,
            response_type="conversation",
            message="I do not have an active campaign in this session yet. Please create a campaign objective first.",
            data=None,
        ),
    )


def _remember(session: ChatSession, response: ChatResponse) -> ChatResponse:
    session.messages.append({"role": "assistant", "content": response.message})
    return response


def _is_greeting(text: str) -> bool:
    cleaned = re.sub(r"[^a-z ]", "", text).strip()
    return cleaned in GREETINGS


def _looks_like_campaign_request(text: str) -> bool:
    return any(keyword in text for keyword in CAMPAIGN_KEYWORDS)


def _is_followup_question(text: str) -> bool:
    return (
        "?" in text
        or _is_segment_profile_question(text)
        or _is_next_best_offer_question(text)
        or any(text.startswith(prefix) for prefix in ("why", "what", "how", "which", "show", "explain", "tell me", "i want to know"))
    )


def _is_modification_request(text: str) -> bool:
    terms = ("make", "change", "changed", "update", "regenerate", "shorter", "formal", "conversational", "premium", "rewrite", "instead", "should be", "rather than", "replace", "switch", "give")
    return any(term in text for term in terms)


def _is_export_request(text: str) -> bool:
    return any(term in text for term in ("export", "pdf", "one pager", "one-pager", "download"))


def _missing_fields(text: str, parsed) -> list[str]:
    missing = []
    if parsed.campaign_intent == "recommend_best_campaign" and not any(
        term in text for term in ("best campaign", "best opportunity", "recommend")
    ):
        missing.append("campaign_intent")
    if not any(term in text for term in TIME_KEYWORDS):
        missing.append("time_window")
    return missing


def _clarification_message(missing: list[str], intent: str) -> str:
    if "campaign_intent" in missing:
        return "What campaign outcome should I optimize: ARPU, churn, data usage, activity, inactive recovery, or best opportunity?"
    if "time_window" in missing:
        return f"I can build the {intent.replace('_', ' ')} campaign. What time window should I use?"
    return "I need one more detail before creating the campaign plan."


def _combine_pending(pending: PendingClarification, text: str) -> str:
    original = pending.original_message or ""
    return f"{original}. {text}".strip()


def _chat_agent_context(context_id: str) -> dict[str, Any]:
    if context_id not in _CHAT_AGENT_CONTEXTS:
        raise KeyError(f"Unknown chat Deep Agent context_id: {context_id}")
    return _CHAT_AGENT_CONTEXTS[context_id]


def _json_response(response: ChatResponse) -> str:
    return json.dumps(jsonable_encoder(response), indent=2, default=str)


def _deep_agent_prompt_param(create_deep_agent_fn: Any) -> str:
    params = inspect.signature(create_deep_agent_fn).parameters
    return "instructions" if "instructions" in params else "system_prompt"
