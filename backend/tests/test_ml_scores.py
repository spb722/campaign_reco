from app.tools.ml_score_tool import load_mock_ml_scores


def test_mock_ml_scores_load_for_segment():
    scores = load_mock_ml_scores(["S001"])
    assert scores["S001"].best_channel == "push"
    assert scores["S001"].channel_scores["push"] == 0.81


def test_mock_ml_score_fallback_is_flagged():
    scores = load_mock_ml_scores(["UNKNOWN"])
    assert scores["UNKNOWN"].fallback_used is True
    assert scores["UNKNOWN"].fallback_reason
