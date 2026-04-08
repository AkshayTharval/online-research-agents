# Low-Level Design — Multi-Agent Research System

## 1. Overview

The system takes a research **topic** as input, automatically gathers information from the web, extracts and verifies factual claims, and produces a cited essay — all without any human intervention after the initial invocation.

It is structured as a **pipeline of four specialized agents** wired together using **LangGraph**, a framework for building stateful, graph-based AI workflows. Each agent does one focused job. A single shared data structure (`ResearchState`) flows through every agent, with each agent reading what it needs and writing its output back into it.

The system can be invoked in two ways:
- **CLI** — `uv run research --topic "..." --num-claims 8` prints the essay and verification summary to stdout
- **Streamlit UI** — `uv run streamlit run src/online_research_agents/ui.py` opens a browser-based interface where the user types a topic and watches every agent step execute in real time

---

## 2. High-Level Flow

```
        ┌──────────────────────┐       ┌──────────────────────────┐
        │  CLI (main.py)       │       │  Streamlit UI (ui.py)    │
        │  --topic "..."       │       │  Topic text input        │
        │  --num-claims 8      │       │  num_claims slider       │
        └──────────┬───────────┘       └────────────┬─────────────┘
                   │                                │
                   └──────────────┬─────────────────┘
                                  │ Creates ResearchState { topic, num_claims }
                                  v
                         build_graph().invoke(state)
                                  │
                         ┌────────┴────────┐
                         │   LangGraph     │
                         │  fan-out (×3)   │
                         └──┬────┬────┬───┘
                            │    │    │
               ┌────────────┘    │    └────────────┐
               ▼                 ▼                 ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │ Search Agent    │ │ Search Agent    │ │ Search Agent    │
    │ angle: "news"   │ │ angle:"academic"│ │ angle:"general" │
    │ (≥5 sources)    │ │ (≥5 sources)    │ │ (≥5 sources)    │
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
             │                   │                   │
             └──────────┬────────┘                   │
                        └──────────────┬─────────────┘
                                       ▼
                           ┌───────────────────────┐
                           │   merge_sources node  │  Deduplicates by URL
                           │   (≥15 sources total) │  across all 3 workers
                           └───────────┬───────────┘
                                       │  state.raw_sources filled
                                       v
                    ┌─────────────────────┐
                    │  Extraction Agent   │  Pulls N factual claims from sources
                    └──────────┬──────────┘
                               │  state.claims filled
                               v
                    ┌──────────────────────┐
                    │  Verification Agent  │  Cross-checks each claim on the web
                    └──────────┬───────────┘
                               │  state.verified_claims filled
                               v
                    ┌─────────────────────┐
                    │  Essay Writer Agent │  Writes essay from verified claims
                    └──────────┬──────────┘
                               │  state.essay filled
                               v
               ┌───────────────────────────────┐
               │  CLI: prints essay + table    │
               │  UI:  renders essay + table   │
               │       + download button       │
               └───────────────────────────────┘
```

---

## 3. Shared State: `ResearchState`

Every agent reads from and writes to one object that travels through the entire pipeline. Think of it as a baton passed in a relay race — each runner (agent) adds something to it before handing it off.

```
ResearchState
├── topic              str           "climate change"
├── num_claims         int           8 (user-configurable)
├── raw_sources        list[RawSource]      ← written by Search Agent
├── claims             list[Claim]          ← written by Extraction Agent
├── verified_claims    list[VerifiedClaim]  ← written by Verification Agent
└── essay              str                  ← written by Essay Writer Agent
```

### Sub-models

```
RawSource
├── url         str   "https://bbc.com/article/123"
├── domain      str   "bbc.com"
└── text        str   (full scraped article text, up to ~10k chars)

Claim
├── claim         str   "Global temps rose 1.1°C since pre-industrial era"
├── source_url    str   "https://bbc.com/article/123"
└── source_domain str   "bbc.com"

VerifiedClaim
├── claim             str
├── source_url        str
├── source_domain     str
├── status            "VERIFIED" | "UNVERIFIED"
└── corroboration_url str | None   "https://reuters.com/article/456"
```

---

## 4. Component-by-Component Design

### 4.1 Retry Utility (`retry.py`)

**Why it exists:** Both LLM calls (to Groq) and web calls (DuckDuckGo + trafilatura) can fail transiently — rate limits, timeouts, temporary server errors. Rather than crashing, the system waits and retries. This logic lives in one place so no agent duplicates it.

**Two decorators:**

```
@llm_retry
- Max attempts    : 5
- Backoff         : exponential (2s → 4s → 8s → 16s)
- Catches         : groq.RateLimitError, groq.APIStatusError
- On each retry   : logs "Retry attempt N, waiting Xs..."

@web_retry
- Max attempts    : 3
- Backoff         : exponential (1s → 2s → 4s)
- Catches         : httpx.HTTPError, httpx.TimeoutException,
                    requests.exceptions.RequestException
- On each retry   : logs "Retry attempt N, waiting Xs..."
```

Usage: any function that calls Groq gets `@llm_retry`; any function that calls DuckDuckGo or trafilatura gets `@web_retry`.

---

### 4.2 Search Agent (`agents/search_agent.py`)

**Responsibility:** One of three parallel workers that each gather raw web content from at least 5 distinct sources, focused on a specific angle of the topic. The three workers run simultaneously in LangGraph and their outputs are merged before Extraction.

**Three parallel instances and their angles:**

| Instance | `query_angle` | Query focus |
|---|---|---|
| `search_news` | `"news"` | Recent events, headlines, latest developments |
| `search_academic` | `"academic"` | Research papers, statistics, scientific findings |
| `search_general` | `"general"` | Broad overviews, causes, effects, solutions |

**Step-by-step logic (per worker):**

```
1. Call Groq LLM (with @llm_retry) to generate 5 queries biased toward
   the given query_angle.
   e.g. topic = "climate change", query_angle = "academic" →
        queries = [
          "climate change peer reviewed research 2024",
          "greenhouse gas emissions scientific data",
          "IPCC sixth assessment report findings",
          "climate sensitivity studies",
          "carbon cycle research papers"
        ]

2. For each query, call DuckDuckGo (with @web_retry) to get search results.

3. For each unique URL not yet visited by this worker:
   a. Call trafilatura.fetch_url(url)   ← downloads the page (with @web_retry)
   b. Call trafilatura.extract(html)    ← strips nav/ads, returns clean text
   c. If text is non-empty:
      - Parse domain from URL
      - Append RawSource(url, domain, text) to collected list

4. Keep looping until len(collected) >= 5 (per worker target).

5. Return a partial state dict with only raw_sources populated.
   The merge_sources node combines all three workers' results.
```

**merge_sources node:**
```
- Receives raw_sources from all 3 workers (up to 15+ total)
- Deduplicates by URL across all three lists
- Writes the combined deduplicated list into state.raw_sources
- Logs total unique sources collected
```

**Key decisions:**
- Each worker targets ≥5 sources (not 15) — three workers together reliably exceed 15.
- Deduplication within a worker is by URL set; cross-worker deduplication happens in the merge node.
- `query_angle` is injected via the LangGraph node wrapper, not the `ResearchState` — state stays clean.
- If trafilatura returns None (paywalled, JS-only pages), the URL is skipped silently.
- Domain is extracted from URL using `urllib.parse.urlparse(url).netloc` to strip `www.`.

---

### 4.3 Extraction Agent (`agents/extraction_agent.py`)

**Responsibility:** Read all raw source text and produce exactly `num_claims` structured factual claims.

**Step-by-step logic:**

```
1. Concatenate all raw source texts into one big string.
   Truncate to ~40,000 characters to fit within Groq's context window.

2. Build a prompt:
   "You are a research assistant. From the text below, extract exactly
    {num_claims} distinct, factual claims about '{topic}'. For each claim,
    identify which source URL it came from. Return valid JSON only."

3. Call Groq LLM (with @llm_retry) using structured output mode —
   LangChain forces the response to parse into list[Claim].

4. Validate: if fewer than num_claims are returned, log a warning but continue.

5. Write list[Claim] into state.claims and return state.
```

**Key decisions:**
- Uses LangChain's `.with_structured_output(list[Claim])` to get a guaranteed Pydantic object back from the LLM, not raw text that needs manual parsing.
- `source_url` and `source_domain` in each Claim are populated by the LLM based on context clues in the concatenated text (each source block is prefixed with its URL before concatenation).

---

### 4.4 Verification Agent (`agents/verification_agent.py`)

**Responsibility:** For each extracted claim, find an independent web source that supports it — from a **different domain** than where the claim was originally found.

**Step-by-step logic:**

```
For each Claim in state.claims:

  1. Build a targeted search query from the claim text.
     e.g. "Global temps rose 1.1°C since pre-industrial era" →
          query = "global temperature rise 1.1 degrees pre-industrial"

  2. Call DuckDuckGo (with @web_retry) and get top 5 results.

  3. For each result URL:
     a. Parse domain
     b. If domain != claim.source_domain:
        → Mark claim as VERIFIED
        → Set corroboration_url = this result URL
        → Break (first valid cross-domain source is enough)

  4. If no cross-domain result found after checking all results:
     → Mark claim as UNVERIFIED, corroboration_url = None

  5. Append VerifiedClaim(...) to collected list.

Write list[VerifiedClaim] into state.verified_claims and return state.
```

**Key decisions:**
- Cross-domain check is strict: `reuters.com` and `www.reuters.com` are treated as the same domain (both normalize to `reuters.com`).
- Does NOT re-scrape pages for verification — DuckDuckGo result snippets alone are sufficient to confirm the claim is referenced.
- Operates claim-by-claim (sequential, not batched) to avoid hitting DuckDuckGo rate limits.

---

### 4.5 Essay Writer Agent (`agents/essay_agent.py`)

**Responsibility:** Write a 3–4 paragraph essay using only VERIFIED claims, with citations.

**Step-by-step logic:**

```
1. Filter state.verified_claims → keep only those where status == VERIFIED.

2. Build a numbered citation list:
   [1] https://reuters.com/article/456
   [2] https://nature.com/article/789
   ...

3. Build a prompt:
   "Write a well-structured 3-4 paragraph essay on '{topic}' using only
    the following verified facts. Use inline citations like [1], [2].
    Facts:
      - Global temps rose 1.1°C since pre-industrial era [1]
      - Arctic ice has shrunk by 13% per decade [2]
      ..."

4. Call Groq LLM (with @llm_retry) to generate the essay.

5. Append the full citation list at the end of the essay text.

6. Write result into state.essay and return state.
```

**Key decisions:**
- If there are zero VERIFIED claims, the agent writes a short "insufficient verified information" message rather than hallucinating.
- Essay prompt explicitly forbids adding facts not in the provided list, minimizing hallucination.

---

## 5. LangGraph Wiring (`graph.py`)

LangGraph treats each agent as a **node** in a directed graph. Edges define execution order. Multiple edges from `START` to different nodes cause LangGraph to execute those nodes **in parallel**.

### 5.1 Full Graph Topology

```
START → search_news     ─┐
START → search_academic  ├─→ merge_sources → extract → verify → write → END
START → search_general  ─┘
```

### 5.2 Conceptual Code Structure

```python
# Conceptual structure (not exact code)

graph = StateGraph(dict)

# Three parallel search workers — each wraps search_agent.run()
# with a different query_angle injected at the node level
graph.add_node("search_news",     lambda s: search_agent.run(s, query_angle="news"))
graph.add_node("search_academic", lambda s: search_agent.run(s, query_angle="academic"))
graph.add_node("search_general",  lambda s: search_agent.run(s, query_angle="general"))

# Merge node — combines raw_sources from all three workers
graph.add_node("merge_sources", _merge_sources_node)

# Sequential pipeline after merge
graph.add_node("extract", _extraction_node)
graph.add_node("verify",  _verification_node)
graph.add_node("write",   _essay_node)

# Fan-out: START triggers all three search workers simultaneously
graph.add_edge(START, "search_news")
graph.add_edge(START, "search_academic")
graph.add_edge(START, "search_general")

# Fan-in: all three workers feed into merge
graph.add_edge("search_news",     "merge_sources")
graph.add_edge("search_academic", "merge_sources")
graph.add_edge("search_general",  "merge_sources")

# Sequential from merge onward
graph.add_edge("merge_sources", "extract")
graph.add_edge("extract",       "verify")
graph.add_edge("verify",        "write")
graph.add_edge("write",         END)

compiled_graph = graph.compile()
```

### 5.3 How Execution Works

1. `compiled_graph.invoke(initial_state_dict)` is called.
2. LangGraph detects three edges from `START` → launches `search_news`, `search_academic`, `search_general` **concurrently** (in separate threads).
3. Each worker collects ≥5 sources and returns its partial state dict.
4. Once **all three** workers finish, LangGraph calls `merge_sources` with their combined outputs.
5. `merge_sources` deduplicates by URL and writes the merged list to `state["raw_sources"]`.
6. Continues sequentially: `extract` → `verify` → `write`.
7. Returns the final state dict, which is cast back to `ResearchState`.

### 5.4 Why LangGraph Instead of Plain Function Calls?

- **Native parallelism** — edges from `START` to multiple nodes triggers concurrent execution with no manual threading code.
- **Built-in state merging** — LangGraph handles passing each parallel node's output into the merge node automatically.
- **Inspectable** — `.get_graph().draw_ascii()` prints the full topology for debugging.
- **Extensible** — conditional edges can be added later (e.g., re-run search if merge yields fewer than 10 sources).

---

## 6. CLI Entrypoint (`main.py`)

```
$ uv run research --topic "climate change" --num-claims 8

Execution sequence:
  1. Parse CLI args
  2. Load .env (GROQ_API_KEY)
  3. Build ResearchState(topic="climate change", num_claims=8)
  4. Call compiled_graph.invoke(state)
  5. Print essay to stdout
  6. Print verification summary table:
       Claim                          Status      Corroboration
       ─────────────────────────────────────────────────────────
       Global temps rose 1.1°C...    VERIFIED    reuters.com
       CO2 levels at 421 ppm...      VERIFIED    nasa.gov
       ...
```

---

## 7. Streamlit Web UI (`ui.py`)

### 7.1 Purpose

The UI gives non-technical users a browser-based way to run the full pipeline while watching every internal step happen in real time — no terminal required. It uses the **exact same LangGraph graph** as the CLI; the only difference is how input is collected and how output is rendered.

### 7.2 Layout

```
┌─────────────────────┬──────────────────────────────────────────────┐
│  SIDEBAR            │  MAIN AREA                                   │
│                     │                                              │
│  Topic:             │  ▼ Search Agent          [live log lines]    │
│  [____________]     │    ✓ bbc.com (4,231 chars)                   │
│                     │    ✓ reuters.com (3,890 chars)               │
│  Num Claims: [8]    │    ✓ nature.com (2,100 chars) ...            │
│                     │                                              │
│  [Run Research]     │  ▼ Extraction Agent       [live log lines]   │
│                     │    • Global temps rose 1.1°C [bbc.com]       │
│                     │    • CO2 at 421 ppm [reuters.com] ...        │
│                     │                                              │
│                     │  ▼ Verification Agent     [live log lines]   │
│                     │    VERIFIED   | claim... | reuters.com       │
│                     │    UNVERIFIED | claim... | —                 │
│                     │                                              │
│                     │  ▼ Essay Writer           [spinner → essay]  │
│                     │    Climate change represents...              │
│                     │                                              │
│                     │  ── Final Output ──                          │
│                     │    [Full essay as Markdown]                  │
│                     │    [Verification summary table]              │
│                     │    [Download Essay .txt]                     │
└─────────────────────┴──────────────────────────────────────────────┘
```

### 7.3 Live Log Capture — `StreamlitLogHandler`

The key mechanism that makes the UI verbose is a custom `logging.Handler` subclass:

```
StreamlitLogHandler
│
├── Attached to the root logger at app startup
├── Each agent already logs via logging.getLogger(__name__)
│     → these records are automatically captured
│
├── on each log record:
│     INFO    → st.write("ℹ " + message)          normal text
│     WARNING → st.warning("⚠ " + message)        amber box
│     ERROR   → st.error("✖ " + message)          red box
│
└── writes into the active st.container() for the current agent section
```

No changes are needed in the agent code — because they already use Python's standard `logging` module, the handler intercepts their output automatically.

### 7.4 Per-Agent Expander Sections

Each agent gets a `st.expander` that opens automatically when the agent starts and stays expanded when it finishes:

| Expander | What it shows live |
|---|---|
| Search Agent | Each `RawSource` URL + domain + char count as it is collected |
| Extraction Agent | Each `Claim` as a bullet: claim text + source domain |
| Verification Agent | A growing table: claim \| VERIFIED/UNVERIFIED badge \| corroboration URL |
| Essay Writer | A spinner while the LLM writes, then the full essay in `st.markdown` |

### 7.5 Execution Model

```
User clicks "Run Research"
        │
        ▼
st.session_state["running"] = True
        │
        ▼
StreamlitLogHandler attached to root logger
        │
        ▼
build_graph().invoke(ResearchState(...))
  [runs synchronously in the Streamlit main thread]
  [each agent's log.info() calls → StreamlitLogHandler → st.write()]
        │
        ▼
st.session_state["result"] = final_state
st.session_state["running"] = False
        │
        ▼
Final output section rendered (essay + table + download button)
```

**Why synchronous?** Streamlit's execution model reruns the script top-to-bottom on every interaction. Running the graph synchronously keeps the state simple — `st.session_state` holds the result and log buffer between reruns.

### 7.6 How to Launch

```bash
uv run streamlit run src/online_research_agents/ui.py
# Opens http://localhost:8501 in the browser automatically
```

---

## 8. File Structure

```
online-research-agents/
├── .env                          # Secrets (git-ignored)
├── .env.example                  # Template
├── pyproject.toml                # Dependencies + build config
├── TASK_LIST.md                  # Task tracker
├── LOW_LEVEL_DESIGN.md           # This document
│
├── src/
│   └── online_research_agents/
│       ├── __init__.py
│       ├── config.py             # Settings, get_settings()
│       ├── models.py             # RawSource, Claim, VerifiedClaim, ResearchState
│       ├── retry.py              # @llm_retry, @web_retry (Task 2)
│       ├── graph.py              # LangGraph StateGraph wiring (Task 7)
│       ├── main.py               # CLI entrypoint (Task 8)
│       ├── ui.py                 # Streamlit web UI (Task 12)
│       └── agents/
│           ├── __init__.py
│           ├── search_agent.py       # Task 3
│           ├── extraction_agent.py   # Task 4
│           ├── verification_agent.py # Task 5
│           └── essay_agent.py        # Task 6
│
└── tests/
    ├── __init__.py
    ├── test_retry.py              # Task 9
    ├── test_search_agent.py       # Task 9
    ├── test_extraction_agent.py   # Task 9
    ├── test_verification_agent.py # Task 9
    ├── test_essay_agent.py        # Task 9
    └── test_integration.py        # Task 10
```

---

## 9. Data Flow Diagram (end-to-end)

```
CLI/UI: topic="climate change", num_claims=8
          |
          | Creates ResearchState { topic, num_claims=8 }
          v
          LangGraph fan-out — all three start simultaneously
          |              |               |
          v              v               v
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ SEARCH AGENT │ │ SEARCH AGENT │ │ SEARCH AGENT │
│ angle: news  │ │angle:academic│ │angle: general│
│              │ │              │ │              │
│ LLM → 5     │ │ LLM → 5     │ │ LLM → 5     │
│ queries      │ │ queries      │ │ queries      │
│              │ │              │ │              │
│ DDG → URLs  │ │ DDG → URLs  │ │ DDG → URLs  │
│ trafilatura  │ │ trafilatura  │ │ trafilatura  │
│ → text       │ │ → text       │ │ → text       │
│              │ │              │ │              │
│ ≥5 sources   │ │ ≥5 sources   │ │ ≥5 sources   │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┴────────────────┘
                        |
                        v
          ┌─────────────────────────────┐
          │      merge_sources node     │
          │  Deduplicate by URL         │
          │  raw_sources = ≥15 unique   │
          └─────────────┬───────────────┘
                        |
                        v
┌──────────────────────────────────────────────────────────────┐
│ EXTRACTION AGENT                                             │
│                                                              │
│  Combine all raw text with URL labels                        │
│  → "SOURCE: https://bbc.com\n{text}\nSOURCE: ..."           │
│                    |                                         │
│                    v                                         │
│               Groq LLM (structured output)                   │
│                    |                                         │
│                    v                                         │
│  list[Claim] = [                                             │
│    { claim: "...", source_url: "...", source_domain: "..." } │
│    × 8 claims                                                │
│  ]                                                           │
│                                                              │
│  state.claims = [Claim × 8]                                  │
└──────────────────────────────────────────────────────────────┘
          |
          v
┌──────────────────────────────────────────────────────────────┐
│ VERIFICATION AGENT                                           │
│                                                              │
│  For each Claim:                                             │
│    DuckDuckGo search for claim text                          │
│         |                                                    │
│         | result domain == claim.source_domain?              │
│         |   YES → skip, try next result                      │
│         |   NO  → VERIFIED, record corroboration_url         │
│         |                                                    │
│         | no cross-domain result found?                      │
│              → UNVERIFIED                                    │
│                                                              │
│  state.verified_claims = [VerifiedClaim × 8]                 │
└──────────────────────────────────────────────────────────────┘
          |
          v
┌──────────────────────────────────────────────────────────────┐
│ ESSAY WRITER AGENT                                           │
│                                                              │
│  Filter: only VERIFIED claims (e.g. 6 out of 8)             │
│  Build citation list: [1] url1, [2] url2, ...                │
│  Prompt Groq LLM → 3-4 paragraph essay with [N] citations   │
│                                                              │
│  state.essay = "Climate change represents one of the most..." │
│                "...[1][2]...\n\nReferences:\n[1] ..."        │
└──────────────────────────────────────────────────────────────┘
          |
          v
     ┌────────────────────────────────────┐
     │  CLI stdout                        │
     │  - essay text                      │
     │  - verification summary table      │
     └────────────────────────────────────┘
          OR
     ┌────────────────────────────────────┐
     │  Streamlit UI (browser)            │
     │  - live log lines per agent        │
     │  - per-agent expander sections     │
     │  - essay rendered as Markdown      │
     │  - verification table with badges  │
     │  - download essay as .txt          │
     └────────────────────────────────────┘
```

---

## 10. External Dependencies and Why Each Was Chosen

| Dependency | Role | Why |
|---|---|---|
| `langgraph` | Orchestrates the agent pipeline | Stateful graph execution, designed for multi-agent LLM workflows |
| `langchain-groq` | LangChain wrapper for Groq API | Provides `.with_structured_output()` for guaranteed Pydantic responses |
| `groq` | Raw Groq API client | Used by langchain-groq under the hood; also needed to catch `RateLimitError` |
| `duckduckgo-search` | Free web search | No API key required; no paid tier |
| `trafilatura` | Web page text extraction | Best-in-class article extraction; strips ads/nav automatically |
| `tenacity` | Retry logic | Battle-tested, composable, clean decorator API |
| `pydantic` | Data models and validation | Type-safe state; prevents agents from passing malformed data |
| `pydantic-settings` | Config from environment | Reads `.env` automatically with type coercion |
| `python-dotenv` | Load `.env` file | Works alongside pydantic-settings for `.env` loading |
| `streamlit` | Web UI framework | Python-native, zero-JS, built-in support for live updates via `st.write()` and `st.status()`; no separate frontend needed |

---

## 11. Key Design Decisions

**Why not one big function?**
Separating into four agents means each is independently testable. You can mock just DuckDuckGo for the Search Agent tests without touching the LLM at all.

**Why is `ResearchState` a Pydantic model?**
LangGraph requires the state to be serializable. Pydantic ensures every field has the right type before it's passed to the next agent — if the Extraction Agent produces a malformed `Claim`, it fails loudly at the boundary rather than corrupting downstream agents silently.

**Why is retry logic in a shared module?**
If retry behavior needs to change (e.g., increase max attempts), it changes in one place. If it were copy-pasted into each agent, you'd have four places to update and four places to forget.

**Why cross-domain verification?**
A claim sourced from `bbc.com` that is also on `bbc.com` proves nothing — it's the same source. A claim that appears on both `bbc.com` and `reuters.com` is genuinely corroborated by an independent source.

**Why parallel search workers instead of one sequential search agent?**
The search phase is the slowest part of the pipeline — each of the 15+ sources requires a DuckDuckGo call plus a `trafilatura` page fetch. Running three concurrent workers (news / academic / general) cuts the wall-clock time for the search phase from ~3× to ~1× while also producing more diverse sources. Each worker focuses on a distinct query angle, so the resulting `raw_sources` pool has better coverage than if the same 15 queries all came from one undirected search. The merge node's URL-based deduplication ensures no source is processed twice.

**Why only parallelise search and not verification?**
Verification is slower per-claim but bounded — `num_claims` is typically small (8–20) and each DuckDuckGo call is fast. More importantly, splitting verification into batches requires order-preserving merging of `list[VerifiedClaim]` to match claim indices, adding fiddly merge logic. The search parallelism gives the largest absolute time saving for the least implementation complexity.

**Why Streamlit for the UI instead of Flask/FastAPI + React?**
Streamlit lets us write the entire UI in Python with zero JavaScript. Since all agents already use Python's `logging` module, a single custom `logging.Handler` intercepts every log line and routes it to the correct UI section — no additional instrumentation in the agent code is required. A Flask + React approach would need a websocket layer, a separate frontend build step, and frontend code, all for the same result.
