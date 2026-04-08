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
              ┌───────────────────────────────────────┐
              │          LangGraph Pipeline           │
              │                                       │
              │  START → search_news     ─┐           │
              │  START → search_academic  ├→ merge    │
              │  START → search_general  ─┘           │
              │              │                        │
              │              ▼                        │
              │          extract → verify → write     │
              └───────────────────────────────────────┘
                                  │
                                  ▼
                     Essay with inline citations
                     + verification summary table
```

### The Four Agents

| Agent | What it does |
|---|---|
| **Search Agent** (×3 parallel) | Generates angle-biased queries (news / academic / general) via Groq LLM, scrapes ≥5 sources each via DuckDuckGo + trafilatura. Three workers run simultaneously. |
| **Extraction Agent** | Combines all scraped text, sends to Groq with structured output to extract exactly N factual claims as typed Pydantic objects. |
| **Verification Agent** | For each claim, searches DuckDuckGo for a corroborating source on a *different domain*. Marks each claim VERIFIED or UNVERIFIED. |
| **Essay Writer Agent** | Writes a 3–4 paragraph essay using only VERIFIED claims, with inline `[N]` citations and a full References section. |

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

With `uv` (recommended):
```bash
uv venv
uv pip install -e ".[dev]"
```

Or with plain pip:
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

**3. Configure your Groq API key**
```bash
cp .env.example .env
# Edit .env and paste your key:
# GROQ_API_KEY=gsk_...
```

---

## Running the UI

```bash
# With uv:
uv run streamlit run src/online_research_agents/ui.py

# With plain venv:
.venv/bin/streamlit run src/online_research_agents/ui.py
```

This opens **http://localhost:8501** in your browser automatically.

### What you'll see

![Streamlit UI Screenshot](docs/screenshots/ui.png)

The UI shows live output from each agent as the pipeline runs — sources being collected, claims being extracted, verification results, and the final essay with citations.

---

## Running Tests

**Unit tests** (no API calls, fast):
```bash
pytest
```

**Integration test** (real Groq API + real DuckDuckGo, ~60–120s):
```bash
pytest -m integration -v
```

The integration test requires `GROQ_API_KEY` in `.env` and internet access. It is automatically skipped if the key is missing.

---

## Project Structure

```
online-research-agents/
├── .env.example                          # Copy to .env, add your Groq key
├── pyproject.toml                        # Dependencies + build config
├── LOW_LEVEL_DESIGN.md                   # Full architecture + debugging guide
├── TASK_LIST.md                          # Build task tracker
│
├── src/online_research_agents/
│   ├── models.py                         # Shared Pydantic models (ResearchState etc.)
│   ├── config.py                         # Settings loaded from .env
│   ├── retry.py                          # @llm_retry, @web_retry decorators
│   ├── graph.py                          # LangGraph pipeline (build_graph, run_pipeline)
│   ├── ui.py                             # Streamlit web UI
│   └── agents/
│       ├── search_agent.py               # Parallel web search + scraping
│       ├── extraction_agent.py           # Structured claim extraction
│       ├── verification_agent.py         # Cross-domain corroboration
│       └── essay_agent.py               # Cited essay generation
│
└── tests/
    ├── test_retry.py                     # 11 unit tests
    ├── test_search_agent.py              # 10 unit tests
    ├── test_extraction_agent.py          # 8 unit tests
    ├── test_verification_agent.py        # 14 unit tests
    ├── test_essay_agent.py               # 16 unit tests
    └── test_integration.py              # 15 end-to-end assertions (real APIs)
```

---

For a full architecture and debugging guide, see [LOW_LEVEL_DESIGN.md](LOW_LEVEL_DESIGN.md).
