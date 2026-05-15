from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_chat_greeting_does_not_create_campaign():
    response = client.post("/chat", json={"session_id": "chat_greeting", "message": "hi"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "conversation"
    assert payload["data"] is None
    assert "campaign objective" in payload["message"].lower()


def test_chat_unclear_campaign_asks_clarification():
    response = client.post(
        "/chat",
        json={"session_id": "chat_clarify", "message": "I want to increase campaign output by 10%"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "clarification"
    assert payload["pending_clarification"]
    assert "campaign_intent" in payload["pending_clarification"]["missing_fields"]


def test_chat_campaign_after_clarification_returns_plan():
    client.post(
        "/chat",
        json={"session_id": "chat_plan_after_clarify", "message": "I want to reduce churn by 10%"},
    )
    response = client.post(
        "/chat",
        json={"session_id": "chat_plan_after_clarify", "message": "next quarter"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "campaign_plan"
    assert payload["data"]["campaign_plan"]["campaign_intent"] == "reduce_churn"


def test_chat_followup_answers_from_existing_campaign():
    created = client.post(
        "/chat",
        json={"session_id": "chat_followup", "message": "Reduce churn by 10% next quarter"},
    ).json()
    campaign_id = created["data"]["campaign_plan"]["campaign_id"]
    response = client.post(
        "/chat",
        json={
            "session_id": "chat_followup",
            "campaign_id": campaign_id,
            "message": "Why did you choose the first segment?",
        },
    )
    payload = response.json()
    assert payload["response_type"] == "answer"
    assert payload["data"]["campaign_plan"]["campaign_id"] == campaign_id
    assert payload["ui_action"]["highlight_segment_id"]


def test_chat_segment_profile_question_uses_segment_lookup():
    created = client.post(
        "/chat",
        json={"session_id": "chat_segment_profile", "message": "Increase ARPU of mid-ARPU customers by 2% in 30 days"},
    ).json()
    campaign_id = created["data"]["campaign_plan"]["campaign_id"]
    first_segment_id = created["data"]["campaign_plan"]["recommended_segments"][0]["segment"]["segment_id"]
    partial_id = first_segment_id.replace("Seg-", "")[:3]

    response = client.post(
        "/chat",
        json={
            "session_id": "chat_segment_profile",
            "campaign_id": campaign_id,
            "message": f"I want to know more about Segment {partial_id}. What kind of people are there in that segment?",
        },
    )

    payload = response.json()
    assert payload["response_type"] == "answer"
    assert first_segment_id in payload["message"]
    assert "customers" in payload["message"].lower()
    assert payload["ui_action"]["set_active_view"] == "segment_drilldown"
    assert payload["ui_action"]["highlight_segment_id"] == first_segment_id


def test_chat_next_best_offer_uses_offer_catalog():
    created = client.post(
        "/chat",
        json={"session_id": "chat_next_best_offer", "message": "Increase ARPU of mid-ARPU customers by 2% in 30 days"},
    ).json()
    campaign_id = created["data"]["campaign_plan"]["campaign_id"]
    first_segment = created["data"]["campaign_plan"]["recommended_segments"][0]
    first_segment_id = first_segment["segment"]["segment_id"]
    current_offer_name = first_segment["offer"]["offer_name"]

    response = client.post(
        "/chat",
        json={
            "session_id": "chat_next_best_offer",
            "campaign_id": campaign_id,
            "message": f"For {first_segment_id}, what is the next best offer available for this segment?",
        },
    )

    payload = response.json()
    assert payload["response_type"] == "answer"
    assert "next best" in payload["message"].lower()
    assert "eligible offer" in payload["message"].lower()
    assert current_offer_name in payload["message"]
    assert payload["ui_action"]["set_active_view"] == "segment_drilldown"
    assert payload["ui_action"]["highlight_segment_id"] == first_segment_id


def test_chat_can_replace_segment_offer_after_next_best_offer():
    created = client.post(
        "/chat",
        json={"session_id": "chat_replace_offer", "message": "Increase ARPU of mid-ARPU customers by 2% in 30 days"},
    ).json()
    campaign_id = created["data"]["campaign_plan"]["campaign_id"]
    first_segment = created["data"]["campaign_plan"]["recommended_segments"][0]
    first_segment_id = first_segment["segment"]["segment_id"]
    original_offer = first_segment["offer"]["offer_name"]

    client.post(
        "/chat",
        json={
            "session_id": "chat_replace_offer",
            "campaign_id": campaign_id,
            "message": f"For segment {first_segment_id}, I need the next best offer that can come for that segment.",
        },
    )
    response = client.post(
        "/chat",
        json={
            "session_id": "chat_replace_offer",
            "campaign_id": campaign_id,
            "message": "Okay, then for that segment, I think it is better to give hayyak voice onlu pack rather than 10 GB weekly RO4.",
        },
    )

    payload = response.json()
    updated_plan = payload["data"]["campaign_plan"]
    updated_first = updated_plan["recommended_segments"][0]
    assert payload["response_type"] == "plan_updated"
    assert updated_first["segment"]["segment_id"] == first_segment_id
    assert updated_first["offer"]["offer_name"] == "Hayyak Voice Only"
    assert updated_first["offer"]["offer_name"] != original_offer
    assert "offer" in payload["ui_action"]["changed_sections"]
    assert "content_plan" in payload["ui_action"]["changed_sections"]
    assert payload["ui_action"]["highlight_segment_id"] == first_segment_id


def test_chat_modification_updates_plan_content():
    created = client.post(
        "/chat",
        json={"session_id": "chat_modify", "message": "Increase ARPU by 2% in 30 days"},
    ).json()
    campaign_id = created["data"]["campaign_plan"]["campaign_id"]
    response = client.post(
        "/chat",
        json={"session_id": "chat_modify", "campaign_id": campaign_id, "message": "Make the SMS copy shorter"},
    )
    payload = response.json()
    assert payload["response_type"] == "plan_updated"
    assert payload["data"]["campaign_plan"]["campaign_id"] == campaign_id
    assert "content_plan" in payload["ui_action"]["changed_sections"]


def test_chat_duration_modification_updates_projection():
    created = client.post(
        "/chat",
        json={"session_id": "chat_duration_modify", "message": "Increase ARPU of mid-ARPU customers by 2% in 30 days"},
    ).json()
    campaign_id = created["data"]["campaign_plan"]["campaign_id"]
    original_projection = created["data"]["campaign_plan"]["projection"]["total_projected_impact"]

    response = client.post(
        "/chat",
        json={
            "session_id": "chat_duration_modify",
            "campaign_id": campaign_id,
            "message": "What if we changed from 30 days to 45 days?",
        },
    )

    payload = response.json()
    updated_plan = payload["data"]["campaign_plan"]
    assert payload["response_type"] == "plan_updated"
    assert updated_plan["campaign_id"] == campaign_id
    assert updated_plan["time_window"] == "45 days"
    assert updated_plan["parsed_objective"]["time_window_value"] == 45
    assert updated_plan["projection"]["total_projected_impact"] == round(original_projection * 1.5, 2)
    assert "projection" in payload["ui_action"]["changed_sections"]


def test_chat_target_lift_modification_updates_projection():
    created = client.post(
        "/chat",
        json={"session_id": "chat_lift_modify", "message": "Increase data consumption by 10% over the next quarter"},
    ).json()
    campaign_id = created["data"]["campaign_plan"]["campaign_id"]
    original_projection = created["data"]["campaign_plan"]["projection"]["total_projected_impact"]

    response = client.post(
        "/chat",
        json={
            "session_id": "chat_lift_modify",
            "campaign_id": campaign_id,
            "message": "I need to change the data consumption to 20%. It is not 10%.",
        },
    )

    payload = response.json()
    updated_plan = payload["data"]["campaign_plan"]
    assert payload["response_type"] == "plan_updated"
    assert updated_plan["target_lift"] == "20%"
    assert updated_plan["time_window"] == "90 days"
    assert abs(updated_plan["projection"]["total_projected_impact"] - round(original_projection * 2, 2)) <= 0.02
    assert "target_lift" in payload["ui_action"]["changed_sections"]
