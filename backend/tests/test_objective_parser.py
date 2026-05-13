from app.services.llm_service import parse_objective


def test_parser_maps_arpu_prompt():
    parsed = parse_objective("Increase ARPU of mid-ARPU customers by 2% in 30 days")
    assert parsed.campaign_intent == "increase_arpu"
    assert parsed.target_metric == "arpu"
    assert parsed.target_lift_value == 2
    assert parsed.time_window_value == 30


def test_parser_maps_churn_prompt():
    parsed = parse_objective("Reduce prepaid churn by 10% next quarter")
    assert parsed.campaign_intent == "reduce_churn"
    assert parsed.time_window_value == 90


def test_parser_maps_inactive_prompt():
    parsed = parse_objective("Engage inactive prepaid customers this month")
    assert parsed.campaign_intent == "reactivate_inactive"
    assert parsed.target_segment_hint == "inactive"
