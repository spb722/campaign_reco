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

## OpenAI and LangSmith

Create a local `.env` file from `.env.example` and set:

```bash
OPENAI_API_KEY=...
MODEL_NAME=gpt-4.1-mini
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=campaign-recommendation-mvp
```

When `OPENAI_API_KEY` is present and `CAMPAIGN_LLM_ENABLED` is not false, objective parsing and content draft generation use OpenAI structured outputs through LangChain. When `CAMPAIGN_DEEP_AGENTS_ENABLED` is not false, LangChain Deep Agents enrich the campaign summary, segment explanations, and draft copy after deterministic guardrails run. LangSmith traces are emitted for those LLM and Deep Agent calls when tracing is enabled.

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
