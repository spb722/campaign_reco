# Campaign Recommendation MVP - Project Context

This document captures the full project context for the Campaign Recommendation MVP in `/Users/sachinpb/PycharmProjects/agents_flow/campaign_mvp`.

It is intended for developers, demo operators, and future agents working on this codebase. It explains the product objective, current architecture, implementation status, agent/tool flow, data files, LangSmith tracing behavior, known mock layers, and the reasoning behind key design decisions.

## 1. Product Goal

The application converts a free-text telecom campaign objective into a structured, rulebook-backed campaign recommendation plan.

Example input:

```text
Increase ARPU of mid-ARPU customers by 2% in 30 days.
```

The app should produce:

- Parsed campaign objective
- Rulebook-backed eligible actions
- Recommended customer segments
- Mock ML-backed channel/timing recommendations
- Offer/tactic recommendations
- Draft campaign content
- Projection calculations
- Validation warnings and guardrails
- Segment drilldown
- One-page PDF export

The target user is a business/campaign manager who wants to move from a natural-language business objective to a structured draft campaign plan.

The demo flow is:

```text
Objective
  -> Segment
  -> Plan
  -> Channels & Content
  -> Follow-up
  -> Validation
  -> One-Pager Export
```

## 2. Version 1 Scope

Version 1 is a demo application. It is not a production campaign execution system.

### In Scope

- Streamlit UI
- FastAPI backend
- LangChain Deep Agents orchestration
- OpenAI LLM calls for:
  - objective parsing
  - strategy explanation
  - content generation
  - agent-driven tool orchestration
- Local deterministic tools
- Rulebook CSV
- Mock/anonymized segment data
- Mock ML outputs
- Offer catalog
- Projection calculator
- Validator
- Campaign cards
- Segment drilldown
- Content drafts
- One-pager PDF export
- JSON campaign export
- LangSmith traces
- Jupyter notebook for step-by-step debugging

### Out of Scope

- Real campaign launch
- SMS/WhatsApp/email sending
- Production data warehouse integration
- Real ML model integration
- MCP servers
- User authentication
- Role-based approval workflow
- Real A/B test execution
- Customer-level production data

The app intentionally has no launch/send capability.

## 3. Current Project Structure

```text
campaign_mvp/
  backend/
    app/
      main.py
      api/
        campaign_routes.py
        health_routes.py
        rulebook_routes.py
      agents/
        campaign_deep_agent.py
      graph/
        state.py
        workflow.py
        nodes.py
      tools/
        data_paths.py
        export_tool.py
        ml_score_tool.py
        offer_tool.py
        projection_tool.py
        rulebook_tool.py
        segment_tool.py
        validation_tool.py
      schemas/
        campaign.py
        content.py
        objective.py
        segment.py
        validation.py
      services/
        campaign_store.py
        llm_service.py
      data/
        rulebook.csv
        mock_segments.csv
        mock_ml_scores.json
        offer_catalog.csv
        campaign_assumptions.json
        sample_prompts.json
    tests/
      conftest.py
      test_api_campaign.py
      test_ml_scores.py
      test_objective_parser.py
      test_projection.py
      test_rulebook_mapping.py
      test_validator.py

  frontend/
    streamlit_app.py
    components/
    pages/

  notebooks/
    deep_agent_tool_flow_debug.ipynb

  outputs/
    json/
    pdfs/

  README.md
  PROJECT_CONTEXT.md
  requirements.txt
  .env.example
  .gitignore
```

## 4. Current Architecture

At a high level:

```text
Streamlit UI
  |
  | HTTP
  v
FastAPI Backend
  |
  | /campaign/recommend
  v
Campaign Deep Agent Orchestrator
  |
  | tool calls
  v
Deterministic Local Tools
  |
  | validated typed output
  v
CampaignPlan Response
```

The current architecture combines dynamic agent orchestration with deterministic business logic.

The important principle is:

```text
The agent decides what tool to call.
The tool decides business truth.
```

That means the Deep Agent can decide that it needs rulebook lookup, segment lookup, offer lookup, or validation, but it cannot invent rulebook results, projections, or validation outcomes.

## 5. Why Deep Agents Were Added

The first implementation used a fixed LangGraph-like pipeline:

```text
parse_objective
  -> map_rulebook
  -> retrieve_segments
  -> retrieve_ml_scores
  -> retrieve_offer_candidates
  -> plan_campaign
  -> calculate_projection
  -> generate_content
  -> validate_campaign
```

This worked, but it was too rigid for a future demo where a user might ask open-ended questions such as:

```text
Why did you choose push?
Show inactive opportunities only.
What is the best campaign this month?
Explain the rulebook basis.
Regenerate only content for S001.
```

The project was migrated so `/campaign/recommend` first tries a Deep Agent tool-flow.

The Deep Agent can call tools dynamically. The deterministic pipeline still exists as fallback and helper logic, but the preferred path is now:

```text
Campaign Deep Agent
  -> parse_objective_tool
  -> lookup_rulebook_tool
  -> find_segments_tool
  -> load_ml_scores_tool
  -> select_offer_candidates_tool
  -> assemble_campaign_plan_tool
  -> calculate_projection_tool
  -> generate_content_tool
  -> validate_campaign_tool
  -> get_final_campaign_plan_tool
```

## 6. Main Runtime Path

The key backend entry point is:

```python
run_campaign_workflow(...)
```

Location:

```text
backend/app/graph/workflow.py
```

Despite the folder name `graph`, this function now behaves as the public orchestration adapter.

Current behavior:

1. If Deep Agents are enabled and `OPENAI_API_KEY` exists:
   - call `run_campaign_deep_agent_workflow(...)`
2. If the Deep Agent tool-flow fails:
   - fallback to deterministic runner
   - include a warning
3. Save campaign version
4. Return state to FastAPI

Deep Agent enablement is controlled by:

```bash
CAMPAIGN_LLM_ENABLED=true
CAMPAIGN_DEEP_AGENTS_ENABLED=true
OPENAI_API_KEY=...
```

If either LLM or Deep Agents are disabled, the app falls back to deterministic mode.

## 7. Deep Agent Orchestration

Main file:

```text
backend/app/agents/campaign_deep_agent.py
```

Main public function:

```python
run_campaign_deep_agent_workflow(prompt, preferred_campaign_type=None, version=1)
```

This function:

1. Creates a unique `context_id`
2. Stores an in-memory run context in `_RUN_CONTEXTS`
3. Creates a Deep Agent
4. Instructs the agent to call planning tools
5. Extracts the final `CampaignPlan` from context
6. Re-runs deterministic validation
7. Saves the campaign version
8. Returns a state object compatible with the API layer

### Why an In-Memory Context Exists

Deep Agent tools are stateless function calls from the agent perspective. To preserve intermediate outputs across tool calls, the implementation uses:

```python
_RUN_CONTEXTS: dict[str, dict[str, Any]]
```

Each tool receives:

```python
context_id: str
```

The tool then reads/writes the shared run context.

This allows a sequence like:

```text
parse_objective_tool(context_id)
  stores parsed_objective

lookup_rulebook_tool(context_id)
  reads parsed_objective
  stores rulebook_matches

find_segments_tool(context_id)
  reads parsed_objective + rulebook_matches
  stores segment_candidates
```

### Self-Healing Tools

The tools are self-healing.

If the agent calls a later tool before an earlier one, missing prerequisites are filled automatically.

Example:

```text
Agent calls lookup_rulebook_tool first.
```

The tool detects that `parsed_objective` is missing and internally calls:

```python
parse_objective_tool(context_id)
```

This prevents fragile failures caused by agent tool-order mistakes while still preserving agent flexibility.

## 8. Deep Agent Tools

### 8.1 `parse_objective_tool`

Input:

```json
{
  "context_id": "campaign_run_..."
}
```

Reads from context:

```json
{
  "prompt": "Increase ARPU of mid-ARPU customers by 2% in 30 days."
}
```

Calls:

```python
parse_objective(...)
```

Writes:

```python
context["parsed_objective"]
context["campaign_id"]
context["warnings"]
```

Typical output:

```json
{
  "campaign_id": "CMP_001",
  "raw_user_prompt": "Increase ARPU of mid-ARPU customers by 2% in 30 days.",
  "campaign_intent": "increase_arpu",
  "target_segment_hint": "mid_arpu",
  "target_metric": "arpu",
  "target_lift_value": 2,
  "target_lift_unit": "percent",
  "time_window_value": 30,
  "time_window_unit": "days",
  "business_context": "prepaid"
}
```

### 8.2 `lookup_rulebook_tool`

Reads:

```python
context["parsed_objective"]
```

Calls:

```python
get_rulebook_matches(parsed_objective)
```

Data source:

```text
backend/app/data/rulebook.csv
```

Writes:

```python
context["rulebook_matches"]
```

For `increase_arpu`, typical matches include:

```text
Gradual Growth -> Upsell / cross-sell
Strong Growth -> Premium upsell / bundling
Rapid Expansion -> Protect value / avoid discounts
```

The LLM does not invent this. The code reads the CSV and filters deterministic rows.

### 8.3 `find_segments_tool`

Reads:

```python
context["parsed_objective"]
context["rulebook_matches"]
```

Calls:

```python
get_segment_candidates(rulebook_matches, parsed_objective)
```

Data source:

```text
backend/app/data/mock_segments.csv
```

Writes:

```python
context["segment_candidates"]
```

For the ARPU example, the current smoke test returns 3 segments.

### 8.4 `load_ml_scores_tool`

Reads:

```python
context["segment_candidates"]
```

Calls:

```python
load_mock_ml_scores(segment_ids)
```

Data source:

```text
backend/app/data/mock_ml_scores.json
```

Writes:

```python
context["ml_scores"]
```

Typical output for `S001`:

```json
{
  "best_channel": "push",
  "secondary_channel": "whatsapp",
  "best_time_window": "18:00-21:00",
  "expected_conversion": 0.08,
  "fatigue_risk": "medium",
  "model_confidence": 0.83
}
```

### 8.5 `select_offer_candidates_tool`

Reads:

```python
context["segment_candidates"]
context["parsed_objective"]
```

Calls:

```python
get_offer_candidates(segments, parsed_objective)
```

Data source:

```text
backend/app/data/offer_catalog.csv
```

Writes:

```python
context["offer_candidates"]
```

### 8.6 `assemble_campaign_plan_tool`

Reads:

```python
context["parsed_objective"]
context["rulebook_matches"]
context["segment_candidates"]
context["ml_scores"]
context["offer_candidates"]
```

Calls:

```python
plan_campaign_node(state)
```

Writes:

```python
context["campaign_plan"]
context["selected_segments"]
```

This creates the draft `CampaignPlan`, including:

- campaign title
- summary
- recommended segment cards
- tactics
- channel plan
- follow-up plan
- risks
- assumptions

### 8.7 `calculate_projection_tool`

Reads:

```python
context["campaign_plan"]
```

Calls:

```python
estimate_campaign_impact(plan)
```

Formula for ARPU:

```text
incremental_revenue =
eligible_users × expected_conversion × expected_arpu_lift
```

Writes:

```python
context["projection"]
context["campaign_plan"].projection
```

Example:

```text
1,200,000 users × 0.08 conversion × ₹35 lift = ₹3,360,000
```

### 8.8 `generate_content_tool`

Reads:

```python
context["campaign_plan"]
```

Calls:

```python
make_content_drafts(...)
```

Writes:

```python
context["content_plan"]
context["campaign_plan"].content_plan
```

Every content draft is marked:

```json
{
  "approval_required": true,
  "approved": false
}
```

### 8.9 `validate_campaign_tool`

Reads:

```python
context["campaign_plan"]
```

Calls:

```python
validate_campaign_plan(plan)
```

Writes:

```python
context["validation_result"]
context["campaign_plan"].validation
context["warnings"]
context["errors"]
```

Validation checks:

- parsed objective exists
- segment has rulebook match
- projection formula exists
- no negative customer counts
- supported channels only
- content remains draft
- export data is complete
- no send/launch behavior

### 8.10 `get_final_campaign_plan_tool`

Reads:

```python
context["campaign_plan"]
```

Ensures:

- content exists
- projection exists
- validation exists

Returns the final typed plan as JSON.

## 9. Example Query Story

Input:

```text
Increase ARPU of mid-ARPU customers by 2% in 30 days.
```

### Step-by-Step Story

1. Streamlit sends prompt to FastAPI:

```http
POST /campaign/recommend
```

2. FastAPI calls:

```python
run_campaign_workflow(prompt)
```

3. Workflow sees Deep Agents are enabled.

4. Workflow calls:

```python
run_campaign_deep_agent_workflow(prompt)
```

5. A new context is created:

```python
context_id = "campaign_run_..."
```

6. Deep Agent receives an instruction:

```text
Run a campaign recommendation for this context_id.
Call the tools needed to complete the plan.
```

7. Deep Agent calls tools.

8. The first tool parses:

```json
{
  "campaign_intent": "increase_arpu",
  "target_segment_hint": "mid_arpu",
  "target_metric": "arpu",
  "target_lift_value": 2,
  "time_window_value": 30
}
```

9. Rulebook tool maps `increase_arpu` to allowed trend/action rows.

10. Segment tool finds matching mock prepaid segments.

11. ML score tool loads channel/timing/conversion/fatigue scores.

12. Offer tool selects eligible offers.

13. Plan tool creates campaign cards.

14. Projection tool calculates impact.

15. Content tool drafts messages.

16. Validator checks guardrails.

17. Final tool returns the plan.

18. Backend validates once more.

19. Campaign is saved to local JSON store.

20. FastAPI returns the plan to Streamlit.

## 10. LLM vs Code Responsibilities

### LLM / OpenAI / Deep Agent Does

- Understand free-text objective
- Decide which tools to call
- Generate strategy wording
- Generate content drafts
- Explain why recommendations were selected
- Provide agent-level reasoning in LangSmith traces

### Code / Deterministic Tools Do

- Rulebook matching
- Segment filtering
- ML score lookup
- Offer eligibility
- Projection math
- Validation
- Export generation
- Guardrail enforcement
- Campaign storage

This split is deliberate.

The app should be flexible in conversation, but deterministic in business rules.

## 11. LangSmith Tracing

LangSmith should show:

- root Deep Agent orchestrator run
- model calls
- tool calls
- parser runs
- nested subagent/tool activity where applicable

The important root run name for full orchestration is:

```text
campaign_deep_agent_orchestrator
```

Other useful traced runs include:

```text
parse_objective
generate_strategy_explanation
generate_content_drafts
```

The project name is configured by:

```bash
LANGSMITH_PROJECT=campaign-recommendation-mvp
```

Required environment variables:

```bash
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=campaign-recommendation-mvp
```

## 12. Debugging Notebook

Notebook:

```text
notebooks/deep_agent_tool_flow_debug.ipynb
```

Purpose:

- Run the same tools step by step
- Inspect exact tool input/output
- Inspect run context after each call
- Run full Deep Agent workflow
- Stream Deep Agent chunks locally
- Compare notebook behavior with LangSmith trace

Recommended use:

1. Open notebook.
2. Run environment setup.
3. Run manual tool cells top to bottom.
4. Open LangSmith in parallel.
5. Run the full Deep Agent cell.
6. Compare local notebook outputs with the LangSmith trace tree.

## 13. API Endpoints

Main endpoints:

```text
GET  /health
POST /campaign/parse
POST /campaign/recommend
GET  /campaign/{campaign_id}
POST /campaign/{campaign_id}/regenerate
POST /campaign/{campaign_id}/edit
POST /campaign/{campaign_id}/validate
POST /campaign/{campaign_id}/export
GET  /campaign/{campaign_id}/download
GET  /rulebook/summary
GET  /segments/mock
```

All API responses follow:

```json
{
  "success": true,
  "data": {},
  "warnings": [],
  "errors": [],
  "request_id": "REQ_001"
}
```

## 14. Streamlit UI

Main file:

```text
frontend/streamlit_app.py
```

Key features:

- Prompt input
- Example prompts
- Generate campaign button
- Parsed objective tab
- Recommended segment cards
- Campaign plan tab
- Segment drilldown tab
- Content drafts tab
- Assumptions and validation tab
- One-pager export tab

The frontend calls FastAPI over HTTP.

Default backend URL:

```bash
BACKEND_URL=http://localhost:8000
```

In recent runs, Streamlit was started with:

```bash
BACKEND_URL=http://127.0.0.1:8000 venv/bin/streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

The Streamlit request timeout is set high enough for Deep Agent runs.

## 15. Data Files

### `rulebook.csv`

Contains trend/action mappings.

Important rows:

```text
Declining -> Retention
Dormant -> Reactivation / light NBO
Early Recovery -> Gentle NBO
Gradual Growth -> Upsell / cross-sell
Strong Growth -> Premium upsell / bundling
Rapid Expansion -> Protect value / avoid discounts
```

### `mock_segments.csv`

Contains anonymized segment-level records.

Important fields:

- `segment_id`
- `segment_name`
- `rfm_segment`
- `data_usage_segment`
- `voice_usage_segment`
- `data_usage_trend`
- `voice_usage_trend`
- `customer_count`
- `avg_arpu`
- `churn_risk_score`
- `activity_score`
- `inactive_days`
- `offer_affinity`
- `business_context`

### `mock_ml_scores.json`

Contains mock ML outputs per segment.

Important fields:

- channel scores
- best channel
- secondary channel
- best time window
- expected CTR
- expected conversion
- fatigue risk
- offer affinity
- model confidence

### `offer_catalog.csv`

Contains local offer definitions.

Important fields:

- offer type
- campaign intent
- price
- benefit
- validity
- eligible usage segment
- eligible RFM segment
- estimated ARPU lift
- estimated data lift
- estimated save rate
- cost per user
- margin impact

### `campaign_assumptions.json`

Contains default assumptions.

Examples:

```json
{
  "default_frequency_cap_per_30_days": 4,
  "quiet_hours": {
    "start": "22:00",
    "end": "08:00"
  },
  "arpu": {
    "default_conversion": 0.08,
    "default_lift_rupees": 35
  }
}
```

## 16. Mocked Components

Still mocked:

- Segment data
- ML scores
- Offer catalog
- Conversion estimates
- Fatigue risk
- Projection assumptions
- Local persistence
- PDF content richness
- Approval workflow
- Campaign launch integration

Real/functional:

- FastAPI backend
- Streamlit frontend
- Deep Agent orchestration
- OpenAI objective parsing
- OpenAI content/strategy generation
- LangSmith tracing
- Rulebook filtering
- Segment filtering
- Projection math
- Validation
- PDF generation
- JSON export

## 17. Environment Variables

Expected `.env` keys:

```bash
OPENAI_API_KEY=...
MODEL_NAME=gpt-4.1-mini
CAMPAIGN_LLM_ENABLED=true
CAMPAIGN_DEEP_AGENTS_ENABLED=true
BACKEND_URL=http://localhost:8000
APP_ENV=local
CAMPAIGN_DATA_DIR=backend/app/data
EXPORT_DIR=outputs/pdfs
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=campaign-recommendation-mvp
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Important toggles:

```bash
CAMPAIGN_LLM_ENABLED=false
```

Disables OpenAI-backed parsing/copy and uses deterministic fallback.

```bash
CAMPAIGN_DEEP_AGENTS_ENABLED=false
```

Disables Deep Agent tool-flow and uses deterministic fallback.

## 18. Run Commands

From:

```bash
/Users/sachinpb/PycharmProjects/agents_flow/campaign_mvp
```

Install:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Stop existing services:

```bash
lsof -tiTCP:8000 -sTCP:LISTEN | xargs -r kill
lsof -tiTCP:8501 -sTCP:LISTEN | xargs -r kill
```

Start backend:

```bash
CAMPAIGN_DEEP_AGENTS_ENABLED=true venv/bin/uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Start Streamlit:

```bash
BACKEND_URL=http://127.0.0.1:8000 venv/bin/streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
http://127.0.0.1:8000/docs
```

## 19. Test Commands

Run tests without live LLM calls:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=backend pytest backend/tests -q -p no:cacheprovider
```

Current validation status:

```text
13 passed, 1 warning
```

The warning is from ReportLab dependency internals and is not currently blocking.

## 20. Recent Issues and Fixes

### `ModuleNotFoundError`

Cause:

Backend was running from project venv, but the new dependencies were not installed there.

Fix:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### OpenAI `Invalid schema for response_format`

Cause:

Strict OpenAI `json_schema` response format rejected the generated Pydantic schema for a dictionary field.

Fix:

Changed structured output mode from:

```python
method="json_schema"
```

to:

```python
method="function_calling"
```

### Deep Agents `instructions` vs `system_prompt`

Cause:

Different installed Deep Agents versions use different parameter names.

Fix:

The code now detects the installed `create_deep_agent` signature and uses:

- `instructions` if available
- otherwise `system_prompt`

### Deep Agent Tool Order `KeyError`

Cause:

The agent called a later tool before an earlier prerequisite tool.

Example:

```text
lookup_rulebook_tool before parse_objective_tool
```

Fix:

Tools now self-heal with `_ensure_*` helpers.

## 21. Design Principles

### Keep Business Logic Deterministic

The LLM should not decide what is allowed.

Correct:

```text
LLM decides it needs rulebook lookup.
Tool returns allowed rulebook actions.
LLM explains the result.
```

Incorrect:

```text
LLM invents rulebook actions.
```

### Let the Agent Choose the Path

The agent should be able to respond to different types of user requests by choosing tools.

Examples:

```text
Recommend ARPU campaign
  -> full campaign planning tool flow

Why push over WhatsApp?
  -> ML score lookup and explanation

Show inactive opportunities
  -> objective parsing, rulebook lookup, segment filtering

Regenerate content only
  -> content tool and validation
```

### Validate After Creativity

Any time the LLM or agent modifies language, content, or explanation, deterministic validation should run afterward.

## 22. Known Limitations

- Deep Agent tool-flow currently targets campaign recommendation requests.
- Follow-up conversational questions may need additional dedicated tools.
- LangSmith trace will show tool calls, but internal self-healing prerequisite calls may not appear as separate agent-decided calls unless wrapped separately.
- Segment data is small and mock.
- ML scores are static.
- Rulebook is a simplified CSV representation.
- Campaign IDs are generated in memory and may reset across backend restarts.
- Store is local/in-memory plus JSON snapshots, not production persistence.

## 23. Recommended Next Steps

1. Add LangSmith `@traceable` wrappers around deterministic tool internals so self-healing prerequisite calls also appear clearly.
2. Add a `campaign_question_answer_tool` for follow-up questions.
3. Add a `channel_explanation_tool` for “why push/WhatsApp/SMS?” questions.
4. Add a `rulebook_explain_tool` for business users.
5. Add a `regenerate_content_tool` into the Deep Agent tool-flow.
6. Add SQLite campaign version store.
7. Add richer PDF preview/export controls.
8. Replace mock ML JSON with model-service adapter when available.
9. Replace mock segment CSV with warehouse adapter when available.
10. Add more demo prompts and regression tests.

## 24. Mental Model for Future Developers

Think of the system as three layers:

```text
Layer 1: Agent Orchestration
  Deep Agent chooses tools and produces a traceable reasoning path.

Layer 2: Deterministic Business Tools
  Python functions return trusted data, rules, projections, and validation.

Layer 3: Presentation
  FastAPI returns typed JSON; Streamlit renders cards, drilldowns, content, and exports.
```

The agent is the planner.

The tools are the source of truth.

The validator is the safety net.

The UI is the demo surface.

