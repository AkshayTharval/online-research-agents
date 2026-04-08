# CLAUDE.md — Online Research Agents

This file is read automatically by Claude Code at session start. It gives you the context needed to work in this repo without asking setup questions.

---

## What this project does

A multi-agent research pipeline that takes a topic, runs three parallel web-search workers, extracts factual claims, cross-verifies each claim on an independent domain, and writes a cited essay — all driven by Groq LLMs and DuckDuckGo (no paid API beyond Groq).

---

## Tech stack

| Layer | Library |
|---|---|
| Agent orchestration | `langgraph` — `StateGraph` with fan-out/fan-in |
| LLM | `langchain-groq` / `groq` — `llama-3.3-70b-versatile` (fallback: `llama-3.1-8b-instant`) |
| Web search | `duckduckgo-search` — `DDGS().text()` |
| Web scraping | `trafilatura` — `fetch_url()` + `extract()` |
| Retry logic | `tenacity` — `@llm_retry` (5×, exp backoff) and `@web_retry` (3×) |
| Data models | `pydantic` v2 — `BaseModel`, `with_structured_output()` |
| Config | `pydantic-settings` — reads `GROQ_API_KEY` etc. from `.env` |
| UI | `streamlit` ≥ 1.40 — `st.status()` for live streaming agent output |
| Tests | `pytest` + `pytest-mock` |

---

## One-time setup

```bash
# 1. Create venv and install (including dev deps)
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Add your Groq API key
cp .env.example .env
# edit .env → GROQ_API_KEY=gsk_...
```

There is no `.env.example` committed — create `.env` manually:
```
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile          # optional override
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant    # optional override
```

---

## Running things

```bash
# Start the Streamlit UI (primary way to use the project)
source .venv/bin/activate
streamlit run src/online_research_agents/ui.py
# → opens http://localhost:8501

# Run unit tests (fast, no API calls)
pytest

# Run integration test (hits real Groq API + DuckDuckGo, ~2–4 min)
pytest -m integration -v
```

---

## Source layout

```
src/online_research_agents/
├── config.py               # Settings (Groq keys, model names) via pydantic-settings
├── models.py               # Shared Pydantic models: ResearchState, Claim, VerifiedClaim, …
├── retry.py                # @llm_retry and @web_retry tenacity decorators
├── graph.py                # LangGraph graph + run_pipeline() orchestrator with retry loop
├── ui.py                   # Streamlit UI — StreamlitLogHandler routes logs to st.status() sections
└── agents/
    ├── search_agent.py     # DDG search + trafilatura scraping; EXCLUDED_DOMAINS list
    ├── extraction_agent.py # Structured LLM claim extraction via with_structured_output()
    ├── verification_agent.py # Two-pass DDG corroboration with named-entity relevance check
    └── essay_agent.py      # Cited essay writer; _build_references_section()

tests/
├── test_retry.py           # 11 tests
├── test_search_agent.py    # 10 tests
├── test_extraction_agent.py # 8 tests
├── test_verification_agent.py # 20 tests
├── test_essay_agent.py     # 16 tests
└── test_integration.py     # 15 assertions; skipped automatically without GROQ_API_KEY
```

---

## Architecture in one page

```
User (browser) → Streamlit UI
                      │ topic, num_claims
                      ▼
             graph.run_pipeline()
                      │
             ┌────────┴────────┐   up to 3 rounds until
             │  Boost loop     │   ≥50 % of claims VERIFIED
             │                 │
             │  _run_partial_pipeline()
             │        │
             │   LangGraph graph
             │        │
             │  START ──► search_news (thread)  ─┐
             │  START ──► search_academic (thread)├─► merge_sources ──► extract ──► verify
             │  START ──► search_general (thread) ─┘
             │
             │  (loop back if verified_rate < 0.50)
             └────────────────────────────────────┘
                      │ all accumulated verified claims
                      ▼
               essay_agent.run()
                      │
                      ▼
              ResearchState (essay + verified_claims)
```

**Key state type**: `ResearchState` (Pydantic model) — `topic`, `num_claims`, `raw_sources`, `claims`, `verified_claims`, `essay`.

**LangGraph internal state**: `_GraphState` (TypedDict) — `raw_sources` uses `Annotated[list, operator.add]` so the three parallel workers' outputs are concatenated automatically before `merge_sources` fires.

---

## Key behaviours to know

### Model fallback
Every LLM call tries `llama-3.3-70b-versatile` first (with tenacity retries). If all retries are exhausted (rate limit / quota), the same call is retried automatically with `llama-3.1-8b-instant`. This happens in `search_agent.run()`, `extraction_agent.run()`, and `essay_agent.run()`.

### Verification quality
`verification_agent._is_topically_relevant()` uses a **two-gate check** before accepting a corroborating URL:
1. At least one named entity (proper noun) from the claim must appear in the result snippet — prevents e.g. `mayoclinic.org` corroborating "MS Dhoni" claims via the "MS = Multiple Sclerosis" false match.
2. At least two meaningful content words must also match.

### Source diversity
`search_agent.EXCLUDED_DOMAINS` blocks Wikipedia and content farms. The extraction prompt explicitly instructs the LLM to use the `SOURCE_URL:` header — not any URL embedded in article body text.

### Live UI logging
`ui.StreamlitLogHandler` routes log records by logger name fragment to the matching `st.status()` section. The `_resolve()` method returns `None` for any unrecognised logger (httpx, trafilatura, etc.) so third-party HTTP logs never appear on screen.

---

## Common tasks

| Task | How |
|---|---|
| Add a new agent | Create `src/online_research_agents/agents/new_agent.py`, add a node function in `graph.py`, wire edges, add `"new_agent": "new"` to `StreamlitLogHandler._ROUTES` in `ui.py` |
| Change model | Set `GROQ_MODEL=` in `.env` — no code change needed |
| Change verification threshold | `graph.VERIFICATION_THRESHOLD` (default `0.50`) |
| Change max boost rounds | `graph.MAX_PIPELINE_ROUNDS` (default `3`) |
| Exclude a domain from search | Add to `search_agent.EXCLUDED_DOMAINS` |
| Add a stopword for relevance check | Add to `verification_agent._STOPWORDS` or `_GENERIC_CAPITALIZED` |
| Run only essay agent tests | `pytest tests/test_essay_agent.py -v` |

---

## What NOT to do

- Do not add Wikipedia or other aggregators back to `EXCLUDED_DOMAINS` removal — they drown out primary sources.
- Do not change `with_structured_output()` calls to plain `.invoke()` — the extraction agent depends on typed Pydantic output.
- Do not commit `.env` — it contains your Groq API key.
- Integration tests (`-m integration`) hit real external APIs and cost Groq quota; do not run them in CI without a budget cap.
- `get_settings()` is `@lru_cache` — if you change `.env` mid-process you must restart; do not call `lru_cache.cache_clear()` as a workaround.

---

## Further reading

- `LOW_LEVEL_DESIGN.md` — full architecture, per-agent step-by-step logic, debugging guide, log pattern table, retry config table, and common failure modes.
- `TASK_LIST.md` — original build task list showing what was built and in what order.
