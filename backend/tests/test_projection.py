from app.graph.workflow import run_campaign_workflow


def test_arpu_projection_uses_deterministic_formula():
    state = run_campaign_workflow("Increase ARPU of mid-ARPU customers by 2% in 30 days")
    projection = state["campaign_plan"].projection
    assert projection.metric == "incremental_revenue"
    assert projection.formula == "eligible_users x expected_conversion x expected_arpu_lift"
    assert projection.total_projected_impact > 0
    first = projection.segment_impacts[0]
    expected = first["eligible_users"] * first["expected_conversion"] * first["expected_arpu_lift"]
    assert first["projected_impact"] == round(expected, 2)
