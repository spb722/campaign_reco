from __future__ import annotations

from fastapi import APIRouter

from app.schemas.objective import APIResponse
from app.tools.rulebook_tool import rulebook_summary
from app.tools.segment_tool import load_mock_segments


router = APIRouter()


@router.get("/rulebook/summary", response_model=APIResponse)
def get_summary() -> APIResponse:
    return APIResponse(request_id="REQ_RULEBOOK", data=rulebook_summary())


@router.get("/segments/mock", response_model=APIResponse)
def get_segments() -> APIResponse:
    return APIResponse(
        request_id="REQ_SEGMENTS",
        data=[segment.model_dump() if hasattr(segment, "model_dump") else segment.dict() for segment in load_mock_segments()],
        warnings=["Mock/anonymized segment-level data only."],
    )
