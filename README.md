# Online Research Agents

A multi-agent research system that takes a topic, automatically gathers web sources, extracts and cross-verifies factual claims, and writes a cited essay — all without human intervention.

Built with **LangGraph**, **Groq** (free tier), and **DuckDuckGo** (no API key needed).

---

## How It Works

```
                         ┌─────────────────┐
                         │  Streamlit UI   │  browser-based, verbose live output
                         └────────┬────────┘
                                  │ topic + num_claims
                                  ▼
              ┌───────────────────────────────────────────────────┐
              │           run_pipeline() — boost loop             │
              │   Repeats up to 3 rounds until ≥50% verified      │
              │                                                   │
              │   ┌─────────────────────────────────────────┐    │
              │   │           LangGraph Graph               │    │
              │   │                                         │    │
              │   │  START → search_news     ─┐             │    │
              │   │  START → search_academic  ├→ merge      │    │
              │   │  START → search_general  ─┘             │    │
              │   │              │                          │    │
              │   │              ▼                          │    │
              │   │      extract → verify                   │    │
              │   └─────────────────────────────────────────┘    │
              │                                                   │
              │   If verified < 50% → run another round          │
              └───────────────────────────────────────────────────┘
                                  │ all accumulated verified claims
                                  ▼
                          essay_agent.run()
                                  │
                                  ▼
                     Essay with inline citations
                     + verification summary table
```

### The Four Agents

| Agent | What it does |
|---|---|
| **Search Agent** (×3 parallel) | Generates angle-biased queries (news / academic / general) via Groq LLM, scrapes ≥5 sources each via DuckDuckGo + trafilatura. Skips Wikipedia and content aggregators. Three workers run simultaneously. |
| **Extraction Agent** | Combines all scraped text, sends to Groq with structured output to extract exactly N factual claims. Explicitly instructs the LLM to use the correct `SOURCE_URL` header and spread claims across all sources. |
| **Verification Agent** | For each claim, runs two DuckDuckGo passes (short keyword query + full claim). Checks 15 results per pass. Requires the corroborating result to contain a named entity from the claim (prevents e.g. a medical site corroborating a cricket claim via an acronym match). |
| **Essay Writer Agent** | Writes a 3–4 paragraph essay using only VERIFIED claims, with inline `[N]` citations and a References section (one entry per line). |

### Verification Boost Loop

After each search→extract→verify round, `run_pipeline()` checks whether at least 50% of accumulated claims are VERIFIED. If not, it runs another full round and merges the new verified claims into the pool. This repeats up to 3 rounds total before writing the essay.

### Model Fallback

The system uses `llama-3.3-70b-versatile` as the primary model. If it is throttled and all tenacity retries are exhausted, each agent automatically falls back to `llama-3.1-8b-instant` for that call.

---

## Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com) (no credit card required)
- Internet access (DuckDuckGo search + page scraping)

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/AkshayTharval/online-research-agents.git
cd online-research-agents
```

**2. Create a virtual environment and install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # installs the package + dev deps — must run before the app
```

**3. Configure your Groq API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=gsk_your_key_here
```

Optional overrides:
```
GROQ_MODEL=llama-3.3-70b-versatile       # primary model (default)
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant # fallback when throttled (default)
```

---

## Running the UI

```bash
source .venv/bin/activate
streamlit run src/online_research_agents/ui.py
```

This opens **http://localhost:8501** in your browser automatically.

### UI Screenshot

![Online Research Agents UI](docs/screenshots/ui.png)

The sidebar lets you enter a topic and choose how many claims to extract. Each agent section streams live progress as the pipeline runs — sources being collected, claims being extracted and verified, and the final essay with citations.

---

## Running Tests

**Unit tests** (no API calls, fast):
```bash
pytest
```

**Integration test** (real Groq API + real DuckDuckGo, ~2–4 min per round):
```bash
pytest -m integration -v
```

The integration test requires `GROQ_API_KEY` in `.env` and internet access. It is automatically skipped if the key is missing.

---

## Project Structure

```
online-research-agents/
├── .env                                  # Your Groq key (git-ignored)
├── pyproject.toml                        # Dependencies + build config
├── CLAUDE.md                             # Claude Code context document
├── LOW_LEVEL_DESIGN.md                   # Full architecture + debugging guide
├── TASK_LIST.md                          # Build task tracker
│
├── src/online_research_agents/
│   ├── models.py                         # Shared Pydantic models (ResearchState etc.)
│   ├── config.py                         # Settings loaded from .env (primary + fallback model)
│   ├── retry.py                          # @llm_retry (5×), @web_retry (3×) decorators
│   ├── graph.py                          # LangGraph pipeline + run_pipeline() boost loop
│   ├── ui.py                             # Streamlit web UI with st.status() live streaming
│   └── agents/
│       ├── search_agent.py               # Parallel web search + scraping; EXCLUDED_DOMAINS
│       ├── extraction_agent.py           # Structured claim extraction via Groq
│       ├── verification_agent.py         # Two-pass cross-domain corroboration with relevance check
│       └── essay_agent.py               # Cited essay generation + references formatting
│
└── tests/
    ├── test_retry.py                     # 11 unit tests
    ├── test_search_agent.py              # 10 unit tests
    ├── test_extraction_agent.py          # 8 unit tests
    ├── test_verification_agent.py        # 20 unit tests
    ├── test_essay_agent.py               # 16 unit tests
    └── test_integration.py              # 15 end-to-end assertions (real APIs)
```

---

For a full architecture and debugging guide, see [LOW_LEVEL_DESIGN.md](LOW_LEVEL_DESIGN.md).
