from __future__ import annotations

import csv

from app.schemas.objective import ParsedObjective
from app.schemas.segment import RulebookMatch
from app.tools.data_paths import data_dir


def load_rulebook() -> list[RulebookMatch]:
    path = data_dir() / "rulebook.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return [
            RulebookMatch(
                trend=row["trend"],
                trend_meaning=row["trend_meaning"],
                typical_action=row["typical_action"],
                allowed_action_families=row["allowed_action_families"].split("|"),
                eligible_intents=row["eligible_intents"].split("|"),
                rulebook_fit_score=float(row["rulebook_fit_score"]),
            )
            for row in rows
        ]


def get_rulebook_matches(parsed_objective: ParsedObjective) -> list[RulebookMatch]:
    matches = [
        rule
        for rule in load_rulebook()
        if parsed_objective.campaign_intent in rule.eligible_intents
        or parsed_objective.campaign_intent == "recommend_best_campaign"
    ]
    if matches:
        return matches
    return [rule for rule in load_rulebook() if "gentle_nbo" in rule.allowed_action_families]


def rulebook_summary() -> dict:
    rules = load_rulebook()
    return {
        "rfm_categories_supported_in_seed_data": [
            "Champions",
            "Loyal Customers",
            "Potential Loyalists",
            "At Risk",
            "About to Sleep",
            "Hibernating",
            "Lost",
        ],
        "trend_dimensions": [rule.trend for rule in rules],
        "total_combinations_reference": 14700,
        "actions_by_trend": {
            rule.trend: {
                "meaning": rule.trend_meaning,
                "typical_action": rule.typical_action,
                "allowed_action_families": rule.allowed_action_families,
            }
            for rule in rules
        },
    }
