# Multi-Agent Research System — Task List

## Task 1 — Scaffold project structure and dependencies
**Status:** DONE

**What's included:**
- `pyproject.toml` with all runtime dependencies: `langgraph`, `langchain-groq`, `langchain-core`, `groq`, `duckduckgo-search`, `trafilatura`, `tenacity`, `pydantic`, `pydantic-settings`, `python-dotenv`, `httpx`, `requests`
- Dev/test dependencies: `pytest`, `pytest-asyncio`, `pytest-mock`, `respx`
- `uv`-compatible build system (`hatchling`)
- `.env.example` listing required env vars (`GROQ_API_KEY`)
- `.env` with actual Groq API key (git-ignored)
- `.gitignore` excluding `.env`, `__pycache__`, `.venv`, build artifacts
- `src/online_research_agents/__init__.py` — package root
- `src/online_research_agents/agents/__init__.py` — agents sub-package
- `src/online_research_agents/models.py` — shared Pydantic models: `RawSource`, `Claim`, `VerifiedClaim`, `VerificationStatus`, `ResearchState`
- `src/online_research_agents/config.py` — `Settings` (pydantic-settings, loads `.env`) and `get_settings()` cached accessor
- `tests/__init__.py` — empty test package init

---

## Task 2 — Implement retry utility module
**Status:** DONE

**What's included:**
- `src/online_research_agents/retry.py`
- `llm_retry` decorator: exponential backoff, max 5 attempts, catches `groq.RateLimitError` and `groq.APIStatusError`, logs each retry with wait duration
- `web_retry` decorator: exponential backoff, max 3 attempts, catches `httpx.HTTPError`, `httpx.TimeoutException`, `requests.exceptions.RequestException`, logs each retry with wait duration
- Both decorators built with `tenacity` (`retry`, `stop_after_attempt`, `wait_exponential`, `before_sleep_log`)
- Module-level logger (`logging.getLogger(__name__)`)

---

## Task 3 — Implement Search Agent
**Status:** DONE

**What's included:**
- `src/online_research_agents/agents/search_agent.py`
- Accepts `ResearchState`, returns updated state with `raw_sources` populated
- Generates multiple DuckDuckGo queries from the topic (using the Groq LLM to diversify queries)
- Iterates pages of results until **at least 15 distinct URLs** are collected
- Scrapes each URL with `trafilatura.fetch_url` + `trafilatura.extract` for clean text
- Deduplicates by URL; extracts domain from URL for later cross-domain checks
- Applies `web_retry` decorator to the scraping call
- Applies `llm_retry` decorator to the LLM query-generation call
- Skips pages where `trafilatura` returns empty/None text
- Logs progress (number of sources collected so far)

---

## Task 4 — Implement Extraction Agent
**Status:** DONE

**What's included:**
- `src/online_research_agents/agents/extraction_agent.py`
- Accepts `ResearchState` (reads `raw_sources`, `num_claims`, `topic`), returns state with `claims` populated
- Concatenates source texts and sends to Groq LLM with a structured prompt requesting exactly `num_claims` factual claims
- Uses LangChain structured output / Pydantic output parser to guarantee a `list[Claim]` response
- Each `Claim` = `{claim: str, source_url: str, source_domain: str}`
- Applies `llm_retry` to the LLM call
- Truncates combined source text to fit within model context (configurable max chars)
- Logs extracted claim count

---

## Task 5 — Implement Verification Agent
**Status:** DONE

**What's included:**
- `src/online_research_agents/agents/verification_agent.py`
- Accepts `ResearchState` (reads `claims`), returns state with `verified_claims` populated
- For each `Claim`, runs a targeted DuckDuckGo search for corroborating evidence
- A claim is VERIFIED if a result is found whose domain **differs** from `claim.source_domain`
- A claim is UNVERIFIED if no cross-domain corroborating source is found
- Produces `list[VerifiedClaim]` with `status` and `corroboration_url` fields
- Applies `web_retry` to DuckDuckGo search calls
- Logs per-claim verification outcome

---

## Task 6 — Implement Essay Writer Agent
**Status:** DONE

**What's included:**
- `src/online_research_agents/agents/essay_agent.py`
- Accepts `ResearchState` (reads only `verified_claims` with status=VERIFIED, and `topic`)
- Builds a prompt listing only verified claims with their source URLs
- Calls Groq LLM to generate a 3–4 paragraph essay on the topic
- Essay includes inline citations referencing the verified sources
- Appends a numbered citation list at the end of the essay
- Applies `llm_retry` to the LLM call
- Stores result in `state.essay`

---

## Task 7 — Wire agents into a LangGraph state graph
**Status:** TODO

**What's included:**
- `src/online_research_agents/graph.py`
- Defines `build_graph()` returning a compiled `StateGraph`
- Nodes: `search` → `extract` → `verify` → `write_essay`
- Edges wired in sequence; `ResearchState` is the shared state type
- Each node function wraps the corresponding agent's `run(state)` method
- Graph is compiled with `graph.compile()` and returned for use in CLI and tests

---

## Task 8 — Add CLI entrypoint
**Status:** TODO

**What's included:**
- `src/online_research_agents/main.py`
- Uses `argparse` to accept:
  - `--topic TEXT` (required) — the research topic
  - `--num-claims INT` (optional, default 8) — number of claims to extract
- Loads `.env` via `python-dotenv`
- Instantiates `ResearchState(topic=..., num_claims=...)`
- Calls `build_graph().invoke(state)`
- Pretty-prints the final essay and verification summary to stdout
- Entry point registered in `pyproject.toml` as `research` script

---

## Task 9 — Write pytest unit tests
**Status:** TODO

**What's included:**
- `tests/test_retry.py` — tests `llm_retry` and `web_retry`: verifies retry count, correct exceptions are caught, non-retryable exceptions bubble immediately
- `tests/test_search_agent.py` — mocks `DDGS` and `trafilatura`; verifies the agent keeps looping until ≥15 sources are collected; verifies `web_retry` is applied
- `tests/test_extraction_agent.py` — mocks the Groq LLM call; verifies structured `Claim` list is returned with correct count; verifies prompt construction
- `tests/test_verification_agent.py` — mocks `DDGS`; verifies cross-domain logic (same domain = UNVERIFIED, different domain = VERIFIED)
- `tests/test_essay_agent.py` — mocks the Groq LLM call; verifies essay is non-empty and contains citation markers; verifies only VERIFIED claims are included
- All tests use `pytest-mock` (`mocker` fixture); no real API or network calls

---

## Task 10 — Write end-to-end integration test
**Status:** TODO

**What's included:**
- `tests/test_integration.py`
- Uses a **real** DuckDuckGo search and **real** Groq API call (marked `@pytest.mark.integration` so excluded from default `pytest` run)
- Topic: `"benefits of renewable energy"`
- Asserts: `len(state.raw_sources) >= 15`, `len(state.claims) == num_claims`, essay is non-empty string, at least one claim is VERIFIED
- Requires `GROQ_API_KEY` in env; skips automatically if key is missing

---

## Task 11 — Write README
**Status:** TODO

**What's included:**
- `README.md`
- Project overview and architecture diagram (ASCII)
- Prerequisites: Python 3.11+, `uv`
- Setup steps: `uv venv`, `uv pip install -e ".[dev]"`, copy `.env.example` to `.env`, add Groq API key
- How to run CLI: `uv run research --topic "your topic" --num-claims 8`
- How to run the Streamlit UI: `uv run streamlit run src/online_research_agents/ui.py`
- How to run tests: `uv run pytest` (unit) and `uv run pytest -m integration` (integration)
- Description of each agent and the LangGraph flow
- Notes on rate limits, retry behavior, and Groq free tier constraints

---

## Task 12 — Build Streamlit web UI
**Status:** TODO

**What's included:**
- Add `streamlit>=1.40.0` to `pyproject.toml` dependencies
- `src/online_research_agents/ui.py` — Streamlit single-page app
- **Input panel (sidebar):**
  - Text input for research topic (required)
  - Number input for `num_claims` (default 8, min 1, max 20)
  - "Run Research" button to start the pipeline
- **Live progress display (main area):**
  - A custom `logging.Handler` subclass (`StreamlitLogHandler`) that captures log records emitted by all agents and appends them in real-time to a `st.container()` as color-coded messages:
    - `INFO` → normal text
    - `WARNING` → orange/amber text
    - `ERROR` → red text
  - Four collapsible `st.expander` sections, one per agent, that expand automatically as each agent runs:
    - **Search Agent:** shows each source URL as it is collected, with domain and character count
    - **Extraction Agent:** shows each extracted claim with its source URL as a bullet list
    - **Verification Agent:** shows a live table row per claim with VERIFIED (green) / UNVERIFIED (red) badge and corroboration URL
    - **Essay Writer Agent:** shows a spinner while writing, then renders the final essay in a styled `st.markdown` block
- **Final output section:**
  - Full essay rendered as Markdown
  - Verification summary table: claim text | status badge | original source | corroboration source
  - Download button to save the essay as a `.txt` file
- **Implementation details:**
  - Pipeline runs in the same thread using `build_graph().invoke(state)` — Streamlit re-renders after each agent completes via `st.session_state`
  - Agent node functions emit structured log messages at key steps; `StreamlitLogHandler` intercepts these to drive the live UI updates
  - No page reload required — uses `st.rerun()` only after full pipeline completion
  - Launched via: `uv run streamlit run src/online_research_agents/ui.py`
