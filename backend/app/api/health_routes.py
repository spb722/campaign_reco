from __future__ import annotations

from fastapi import APIRouter

from app.schemas.objective import APIResponse


router = APIRouter()


@router.get("/health", response_model=APIResponse)
def health() -> APIResponse:
    return APIResponse(request_id="REQ_HEALTH", data={"status": "ok", "service": "campaign-mvp"})
