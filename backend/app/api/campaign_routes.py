from __future__ import annotations

import copy
from itertools import count
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse

from app.graph.workflow import run_campaign_workflow
from app.schemas.campaign import EditRequest, ExportResponse, RecommendRequest, RegenerateRequest
from app.schemas.objective import APIResponse, ParseRequest
from app.services.campaign_store import load_campaign_version, next_version, save_campaign_version
from app.services.llm_service import make_content_drafts, parse_objective
from app.tools.export_tool import generate_one_pager_pdf, save_campaign_json
from app.tools.validation_tool import validate_campaign_plan


router = APIRouter(prefix="/campaign")
_request_counter = count(1)


def _request_id() -> str:
    return f"REQ_{next(_request_counter):03d}"


def _model_data(value: Any) -> Any:
    return jsonable_encoder(value)


@router.post("/parse", response_model=APIResponse)
def parse_campaign(request: ParseRequest) -> APIResponse:
    parsed = parse_objective(request.prompt, request.preferred_campaign_type)
    return APIResponse(
        request_id=_request_id(),
        data=_model_data(parsed),
        warnings=parsed.assumptions,
    )


@router.post("/recommend", response_model=APIResponse)
def recommend_campaign(request: RecommendRequest) -> APIResponse:
    state = run_campaign_workflow(request.prompt, request.preferred_campaign_type)
    return APIResponse(
        request_id=_request_id(),
        data=_model_data(state["campaign_plan"]),
        warnings=state.get("warnings", []),
        errors=state.get("errors", []),
        success=not bool(state.get("errors")),
    )


@router.get("/{campaign_id}", response_model=APIResponse)
def get_campaign(campaign_id: str) -> APIResponse:
    plan = load_campaign_version(campaign_id)
    if not plan:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")
    return APIResponse(request_id=_request_id(), data=_model_data(plan), warnings=plan.validation.warnings if plan.validation else [])


@router.post("/{campaign_id}/regenerate", response_model=APIResponse)
def regenerate_campaign(campaign_id: str, request: RegenerateRequest) -> APIResponse:
    current = load_campaign_version(campaign_id)
    if not current:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")

    if request.regenerate_scope in {"full_plan", "segment_strategy"}:
        state = run_campaign_workflow(
            current.parsed_objective.raw_user_prompt,
            current.parsed_objective.campaign_intent,
            version=next_version(current),
        )
        plan = state["campaign_plan"]
    else:
        plan = copy.deepcopy(current)
        plan.version = next_version(current)
        if request.regenerate_scope == "content_only":
            _regenerate_content(plan, request.segment_id, request.user_instruction)
        elif request.regenerate_scope == "channel_mix":
            _regenerate_channel_mix(plan, request.segment_id)
        elif request.regenerate_scope == "followup_plan":
            _regenerate_followup(plan, request.segment_id)
        elif request.regenerate_scope == "one_pager_summary":
            plan.summary = f"{plan.summary} Updated summary focus: {request.user_instruction or 'stakeholder one-pager clarity'}."
        plan.validation = validate_campaign_plan(plan)
        save_campaign_version(plan)

    return APIResponse(
        request_id=_request_id(),
        data=_model_data(plan),
        warnings=plan.validation.warnings if plan.validation else [],
        errors=plan.validation.errors if plan.validation else [],
        success=plan.validation.is_valid if plan.validation else True,
    )


@router.post("/{campaign_id}/edit", response_model=APIResponse)
def edit_campaign(campaign_id: str, request: EditRequest) -> APIResponse:
    plan = load_campaign_version(campaign_id)
    if not plan:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")
    plan = copy.deepcopy(plan)
    plan.version = next_version(plan)
    for key, value in request.updates.items():
        if hasattr(plan, key):
            setattr(plan, key, value)
    plan.validation = validate_campaign_plan(plan)
    save_campaign_version(plan)
    return APIResponse(request_id=_request_id(), data=_model_data(plan), warnings=plan.validation.warnings, errors=plan.validation.errors, success=plan.validation.is_valid)


@router.post("/{campaign_id}/validate", response_model=APIResponse)
def validate_campaign(campaign_id: str) -> APIResponse:
    plan = load_campaign_version(campaign_id)
    if not plan:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")
    plan.validation = validate_campaign_plan(plan)
    save_campaign_version(plan)
    return APIResponse(request_id=_request_id(), data=_model_data(plan.validation), warnings=plan.validation.warnings, errors=plan.validation.errors, success=plan.validation.is_valid)


@router.post("/{campaign_id}/export", response_model=APIResponse)
def export_campaign(campaign_id: str) -> APIResponse:
    plan = load_campaign_version(campaign_id)
    if not plan:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")
    plan.validation = validate_campaign_plan(plan)
    if not plan.validation.is_valid:
        return APIResponse(request_id=_request_id(), success=False, data=_model_data(plan.validation), warnings=plan.validation.warnings, errors=plan.validation.errors)
    pdf_path = generate_one_pager_pdf(plan)
    json_path = save_campaign_json(plan)
    plan.export_path = pdf_path
    save_campaign_version(plan)
    payload = ExportResponse(campaign_id=campaign_id, pdf_path=pdf_path, json_path=json_path)
    return APIResponse(request_id=_request_id(), data=_model_data(payload), warnings=plan.validation.warnings)


@router.get("/{campaign_id}/download")
def download_campaign(campaign_id: str) -> FileResponse:
    plan = load_campaign_version(campaign_id)
    if not plan or not plan.export_path:
        raise HTTPException(status_code=404, detail="Export not found; generate PDF first")
    path = Path(plan.export_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export file missing")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


def _regenerate_content(plan, segment_id: str | None, instruction: str | None) -> None:
    targets = [rec for rec in plan.recommended_segments if not segment_id or rec.segment.segment_id == segment_id]
    keep = [draft for draft in plan.content_plan if segment_id and draft.segment_id != segment_id]
    new_drafts = []
    for rec in targets:
        drafts = make_content_drafts(rec.segment, rec.offer, rec.ml_score.best_channel, rec.ml_score.secondary_channel, instruction)
        if instruction:
            for draft in drafts:
                draft.why_this_copy = f"{draft.why_this_copy} Regenerated with instruction: {instruction}."
        new_drafts.extend(drafts)
    plan.content_plan = keep + new_drafts


def _regenerate_channel_mix(plan, segment_id: str | None) -> None:
    for item in plan.channel_plan:
        if segment_id and item.segment_id != segment_id:
            continue
        item.primary_channel, item.secondary_channel = item.secondary_channel, item.primary_channel
        item.score_source = "mock_ml"


def _regenerate_followup(plan, segment_id: str | None) -> None:
    for item in plan.followup_plan:
        if segment_id and item["segment_id"] != segment_id:
            continue
        item["steps"] = [
            "Day 0: primary message",
            "Day 5: value-led reminder for non-converters",
            "Day 14: suppress fatigued users and send final reminder",
        ]
