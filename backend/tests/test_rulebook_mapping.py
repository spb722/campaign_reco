from app.services.llm_service import parse_objective
from app.tools.offer_tool import get_offer_candidates
from app.tools.rulebook_tool import get_rulebook_matches
from app.tools.segment_tool import get_segment_candidates


def test_rulebook_mapper_returns_growth_action_for_arpu():
    parsed = parse_objective("Increase ARPU of mid-ARPU customers by 2% in 30 days")
    matches = get_rulebook_matches(parsed)
    trends = {match.trend for match in matches}
    assert "Gradual Growth" in trends
    assert any("upsell" in match.allowed_action_families for match in matches)


def test_offer_catalog_filters_by_intent():
    parsed = parse_objective("Reduce prepaid churn by 10% next quarter")
    matches = get_rulebook_matches(parsed)
    segments = get_segment_candidates(matches, parsed)
    offers = get_offer_candidates(segments, parsed)
    assert offers
    assert all(candidate.campaign_intent == "reduce_churn" for values in offers.values() for candidate in values)
