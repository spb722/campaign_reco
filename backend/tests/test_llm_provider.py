from app.services.llm_service import _chat_model, _llm_enabled


def test_openrouter_provider_uses_openai_compatible_base_url(monkeypatch):
    monkeypatch.setenv("CAMPAIGN_LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_MODEL_NAME", "openai/gpt-4.1-mini")
    monkeypatch.setenv("OPENROUTER_SITE_URL", "http://localhost:3000")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "campaign-reco-mvp")

    model = _chat_model("test_openrouter")

    assert _llm_enabled()
    assert model.model_name == "openai/gpt-4.1-mini"
    assert str(model.openai_api_base) == "https://openrouter.ai/api/v1"
    assert model.openai_api_key.get_secret_value() == "sk-test"
    assert model.default_headers == {
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "campaign-reco-mvp",
    }


def test_openai_provider_uses_openai_key_and_model(monkeypatch):
    monkeypatch.setenv("CAMPAIGN_LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("MODEL_NAME", "gpt-4.1-mini")

    model = _chat_model("test_openai")

    assert _llm_enabled()
    assert model.model_name == "gpt-4.1-mini"
    assert model.openai_api_key.get_secret_value() == "sk-openai-test"
