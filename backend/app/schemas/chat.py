from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ChatResponseType = Literal[
    "conversation",
    "clarification",
    "campaign_plan",
    "answer",
    "plan_updated",
    "export_ready",
    "error",
]


class ChatRequest(BaseModel):
    session_id: str
    message: str
    campaign_id: str | None = None


class PendingClarification(BaseModel):
    missing_fields: list[str] = Field(default_factory=list)
    partial_objective: dict[str, Any] = Field(default_factory=dict)
    original_message: str | None = None


class ChatResponse(BaseModel):
    success: bool = True
    response_type: ChatResponseType
    message: str
    data: dict[str, Any] | None = None
    ui_action: dict[str, Any] | None = None
    pending_clarification: PendingClarification | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    request_id: str
