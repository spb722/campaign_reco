from __future__ import annotations

import json

from app.schemas.segment import MLScore
from app.tools.data_paths import data_dir


SUPPORTED_CHANNELS = {"sms", "whatsapp", "push", "email", "ivr", "outbound_call"}


def load_mock_ml_scores(segment_ids: list[str] | None = None) -> dict[str, MLScore]:
    path = data_dir() / "mock_ml_scores.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    scores = {record["segment_id"]: MLScore(**record) for record in records}
    if segment_ids is None:
        return scores
    return {segment_id: scores.get(segment_id) or fallback_ml_score(segment_id) for segment_id in segment_ids}


def fallback_ml_score(segment_id: str) -> MLScore:
    return MLScore(
        segment_id=segment_id,
        channel_scores={"sms": 0.55, "whatsapp": 0.5, "push": 0.4, "email": 0.25, "ivr": 0.2, "outbound_call": 0.2},
        best_channel="sms",
        secondary_channel="whatsapp",
        best_time_window="10:00-18:00",
        expected_ctr=0.08,
        expected_conversion=0.04,
        fatigue_risk="unknown",
        offer_affinity="default",
        model_confidence=0.55,
        fallback_used=True,
        fallback_reason="No mock ML score found; using rulebook/channel defaults.",
    )
