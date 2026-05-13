from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas.campaign import CampaignPlan
from app.tools.data_paths import output_dir


_MEMORY_STORE: dict[str, list[CampaignPlan]] = {}


def _dump_model(model: Any) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def save_campaign_version(campaign_plan: CampaignPlan) -> CampaignPlan:
    versions = _MEMORY_STORE.setdefault(campaign_plan.campaign_id, [])
    versions.append(campaign_plan)
    path = Path(output_dir("json")) / f"{campaign_plan.campaign_id}_v{campaign_plan.version}.json"
    path.write_text(json.dumps(_dump_model(campaign_plan), indent=2, default=str), encoding="utf-8")
    return campaign_plan


def load_campaign_version(campaign_id: str, version: int | None = None) -> CampaignPlan | None:
    versions = _MEMORY_STORE.get(campaign_id, [])
    if version is None and versions:
        return versions[-1]
    for plan in versions:
        if plan.version == version:
            return plan

    json_dir = output_dir("json")
    candidates = sorted(json_dir.glob(f"{campaign_id}_v*.json"))
    if not candidates:
        return None
    path = candidates[-1] if version is None else json_dir / f"{campaign_id}_v{version}.json"
    if not path.exists():
        return None
    return CampaignPlan(**json.loads(path.read_text(encoding="utf-8")))


def next_version(campaign_plan: CampaignPlan) -> int:
    versions = _MEMORY_STORE.get(campaign_plan.campaign_id, [])
    return max([plan.version for plan in versions] + [campaign_plan.version]) + 1
