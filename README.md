# Campaign Recommendation MVP

Streamlit + FastAPI demo that converts a free-text campaign objective into a structured, rulebook-backed campaign plan.

## What is included

- Streamlit demo UI
- FastAPI typed API
- LangChain Deep Agents enrichment layer for strategy explanation and copy refinement
- Deterministic preparation pipeline for objective, rulebook, segment, score, offer, projection, and validation steps
- Local deterministic tools for rulebook matching, mock segments, mock ML scores, offer filtering, projection, validation, and one-pager export
- Optional LLM hook for structured parsing/copy, with deterministic fallback for repeatable demos
- Mock/anonymized CSV and JSON seed data
- PDF and JSON export

Version 1 intentionally does not include real campaign launch, SMS/WhatsApp sending, production warehouse integration, MCP servers, authentication, or A/B execution.

## Run

From this directory:

```bash
pip install -r requirements.txt
uvicorn app.main:app --app-dir backend --reload --port 8000
```

In a second terminal:

```bash
BACKEND_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
```

Open:

- API docs: http://localhost:8000/docs
- Streamlit app: the URL printed by Streamlit, usually http://localhost:8501

## LLM Provider and LangSmith

Create a local `.env` file from `.env.example` and set:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=...
MODEL_NAME=gpt-4.1-mini
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=campaign-recommendation-mvp
```

To use OpenRouter through the OpenAI-compatible API, switch the provider:

```bash
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=...
OPENROUTER_MODEL_NAME=openai/gpt-4.1-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=http://localhost:3000
OPENROUTER_APP_NAME=campaign-reco-mvp
```

When the selected provider has an API key and `CAMPAIGN_LLM_ENABLED` is not false, objective parsing and content draft generation use structured outputs through LangChain. When `CAMPAIGN_DEEP_AGENTS_ENABLED` is not false, LangChain Deep Agents route chat requests and campaign planning tool calls. LangSmith traces are emitted for those LLM and Deep Agent calls when tracing is enabled.

## Test

```bash
PYTHONPATH=backend pytest backend/tests
```

## Demo prompts

- Increase ARPU of mid-ARPU customers by 2% in 30 days.
- Reduce prepaid churn by 10% next quarter.
- Increase data consumption by 10% over the next quarter.
- Engage inactive prepaid customers this month.
- Recommend the best campaign opportunity for this month.
