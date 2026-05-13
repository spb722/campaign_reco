from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_recommend_endpoint_returns_full_plan():
    response = client.post("/campaign/recommend", json={"prompt": "Increase ARPU of mid-ARPU customers by 2% in 30 days"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["campaign_id"].startswith("CMP_")
    assert len(data["recommended_segments"]) >= 1
    assert data["projection"]["formula"]
    assert data["validation"]["is_valid"] is True


def test_regenerate_content_updates_version_only_selected_scope():
    created = client.post("/campaign/recommend", json={"prompt": "Engage inactive prepaid customers this month"}).json()["data"]
    campaign_id = created["campaign_id"]
    response = client.post(
        f"/campaign/{campaign_id}/regenerate",
        json={"regenerate_scope": "content_only", "segment_id": created["recommended_segments"][0]["segment"]["segment_id"], "user_instruction": "Make it more premium."},
    )
    assert response.status_code == 200
    updated = response.json()["data"]
    assert updated["parsed_objective"] == created["parsed_objective"]
    assert updated["version"] == created["version"] + 1


def test_export_creates_pdf():
    created = client.post("/campaign/recommend", json={"prompt": "Reduce prepaid churn by 10% next quarter"}).json()["data"]
    response = client.post(f"/campaign/{created['campaign_id']}/export")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["pdf_path"].endswith(".pdf")
