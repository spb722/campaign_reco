from app.graph.workflow import run_campaign_workflow
from app.tools.validation_tool import validate_campaign_plan


def test_validator_passes_complete_draft_with_warnings():
    state = run_campaign_workflow("Increase data consumption by 10% over the next quarter")
    plan = state["campaign_plan"]
    result = validate_campaign_plan(plan)
    assert result.is_valid
    assert "Uses mock data." in result.warnings
    assert result.content_compliance == "passed"


def test_validator_catches_missing_projection():
    state = run_campaign_workflow("Engage inactive prepaid customers this month")
    plan = state["campaign_plan"]
    plan.projection = None
    result = validate_campaign_plan(plan)
    assert not result.is_valid
    assert "Campaign plan is missing deterministic projection." in result.errors
