from __future__ import annotations

from pydantic import BaseModel, Field


class ContentDraft(BaseModel):
    segment_id: str
    channel: str
    draft_copy: str
    tone: str = "direct"
    language: str = "English"
    approval_required: bool = True
    approved: bool = False
    why_this_copy: str
    compliance_notes: list[str] = Field(default_factory=list)
