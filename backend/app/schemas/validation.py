from __future__ import annotations

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    is_valid: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    rulebook_compliance: str = "pending"
    projection_compliance: str = "pending"
    content_compliance: str = "pending"
