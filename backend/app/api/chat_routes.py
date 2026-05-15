from __future__ import annotations

from itertools import count

from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import handle_chat_message


router = APIRouter()
_request_counter = count(1)


def _request_id() -> str:
    return f"REQ_CHAT_{next(_request_counter):03d}"


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return handle_chat_message(
        session_id=request.session_id,
        message=request.message,
        campaign_id=request.campaign_id,
        request_id=_request_id(),
    )
