# Low-Level Design — Multi-Agent Research System

## 1. Overview

The system takes a research **topic** as input, automatically gathers information from the web, extracts and verifies factual claims, and produces a cited essay — all without any human intervention after the initial invocation.

It is structured as a **pipeline of four specialized agents** wired together using **LangGraph**, a framework for building stateful, graph-based AI workflows. Each agent does one focused job. A single shared data structure (`ResearchState`) flows through every agent, with each agent reading what it needs and writing its output back into it.

The primary interface is the **Streamlit UI** (`uv run streamlit run src/online_research_agents/ui.py`), which opens a browser-based interface where the user types a topic and watches every agent step execute in real time. The `run_pipeline()` function in `graph.py` also serves as the programmatic entry point for direct invocation.

---

## 2. High-Level Flow

```
        ┌──────────────────────────────────────┐
        │  Streamlit UI (ui.py)                │
        │  Topic text input / num_claims slider │
        └──────────────────┬───────────────────┘
                           │ Creates ResearchState { topic, num_claims }
                           │ Calls run_pipeline(topic, num_claims)
                           v
               ┌──────────────────────────────────┐
               │  run_pipeline() orchestrator     │
               │  (verification boost loop)       │
               │  MAX_PIPELINE_ROUNDS = 3          │
               │  VERIFICATION_THRESHOLD = 0.50   │
               └──────────────┬───────────────────┘
                              │
               ┌──────────────▼───────────────────┐
               │  _run_partial_pipeline()          │
               │  (one round: search→merge→        │
               │   extract→verify, NO essay)       │
               └──────────────┬───────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   LangGraph       │
                    │  fan-out (×3)     │
                    └──┬────┬────┬──────┘
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
                           │  state.verified_claims filled (partial round)
                           v
               ┌────────────────────────────────────┐
               │  run_pipeline() accumulates unique  │
               │  verified claims across rounds      │
               │  Check: verified_count / total ≥ 50%│
               │  YES → break; NO + rounds left →    │
               │  run another round                   │
               └──────────────┬─────────────────────┘
                              │ After loop ends:
                              v
                ┌─────────────────────┐
                │  Essay Writer Agent │  Writes essay from verified claims
                └──────────┬──────────┘
                           │  state.essay filled
                           v
           ┌───────────────────────────────┐
           │  UI: renders essay + table   │
           │      + download button        │
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

**Excluded domains:** The agent maintains an `EXCLUDED_DOMAINS` frozenset that blocks results from low-quality or community-edited sources before any scraping occurs:

```
EXCLUDED_DOMAINS = frozenset({
    "wikipedia.org", "wikimedia.org", "wikidata.org",
    "wikiwand.com", "dbpedia.org",
    "answers.com", "ask.com", "quora.com"
})
```

The domain check happens **before** calling trafilatura, avoiding unnecessary network calls to pages that will always be rejected.

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
   a. Parse the domain from the URL
   b. If domain is in EXCLUDED_DOMAINS → skip immediately (no fetch)
   c. Call trafilatura.fetch_url(url)   ← downloads the page (with @web_retry)
   d. Call trafilatura.extract(html)    ← strips nav/ads, returns clean text
   e. If text is non-empty:
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
   Each source block is prefixed with its URL:
     "SOURCE_URL: https://bbc.com/article/123\n{text}\n\n"
   Truncate to ~40,000 characters to fit within Groq's context window.

2. Build a prompt with two explicit instructions to the LLM:
   a. source_url MUST be the exact SOURCE_URL: header value, NOT any URL
      found inside the article body text.
   b. Spread claims across ALL provided sources rather than clustering
      claims on a single source.

3. Call Groq LLM (with @llm_retry) using structured output mode —
   LangChain forces the response to parse into list[Claim].

4. Log each extracted claim individually as it is processed.

5. Validate: if fewer than num_claims are returned, log a warning but continue.

6. Write list[Claim] into state.claims and return state.
```

**Key decisions:**
- Uses LangChain's `.with_structured_output(list[Claim])` to get a guaranteed Pydantic object back from the LLM, not raw text that needs manual parsing.
- The `SOURCE_URL:` prefix scheme prevents the LLM from confusing hyperlinks inside article body text with the true origin URL of the source.
- The prompt instruction to spread claims across all sources avoids all claims citing the same one or two sources, which would reduce essay citation diversity.

---

### 4.4 Verification Agent (`agents/verification_agent.py`)

**Responsibility:** For each extracted claim, find an independent web source that supports it — from a **different domain** than where the claim was originally found. Uses a two-pass search strategy with topical relevance gating to prevent false-positive corroboration.

**Constants:**
```
MAX_RESULTS_PER_CLAIM = 15
```

**Step-by-step logic:**

```
For each Claim in state.claims:

  Pass 1 — broad search:
    query = _short_query(claim.claim)   # first 10 words of claim text
    results = DuckDuckGo(query, max_results=MAX_RESULTS_PER_CLAIM)

  Pass 2 — fallback if Pass 1 yields no usable corroborator:
    query = claim.claim                 # full verbatim claim text
    results = DuckDuckGo(query, max_results=MAX_RESULTS_PER_CLAIM)

  For each result in combined results:
    a. Parse domain
    b. If domain == claim.source_domain → skip (same source)
    c. Run _is_topically_relevant(result, claim.claim):
         Gate 1 — named-entity gate:
           Extract capitalised proper nouns from claim text,
           excluding generic words in _GENERIC_CAPITALIZED
           (e.g. "Indian", "International", "World").
           If any named entities exist, at least one must appear
           in the result's title or snippet.
           Prevents e.g. mayoclinic.org corroborating "MS Dhoni"
           claims by matching "MS" as "Multiple Sclerosis".
         Gate 2 — keyword gate:
           At least 2 meaningful content words from the claim must
           appear in the result's title or snippet.
         Both gates must pass → result is accepted.
    d. First accepted result → VERIFIED, corroboration_url = result URL; break.

  If no accepted cross-domain result found:
    → UNVERIFIED, corroboration_url = None

Append VerifiedClaim(...) to collected list.
Write list[VerifiedClaim] into state.verified_claims and return state.
```

**Helper functions:**
- `_short_query(claim_text)` — returns the first 10 words of the claim for a broader initial DuckDuckGo query.
- `_is_topically_relevant(result, claim_text)` — applies the two-gate relevance check described above.
- `_find_cross_domain_result(results, source_domain, claim_text)` — iterates results, applies domain and relevance checks, returns the first valid corroborator.

**Key decisions:**
- Two search passes ensure broad initial coverage (short query) with a verbatim fallback, increasing the chance of finding a genuine corroborator.
- `MAX_RESULTS_PER_CLAIM = 15` (up from 5) gives more candidates per claim before giving up.
- The named-entity gate specifically targets the false-positive pattern where a medical, legal, or unrelated site happens to share an acronym or short word with the claim's subject.
- Does NOT re-scrape pages for verification — DuckDuckGo result title and snippet alone are used for relevance gating.
- Operates claim-by-claim (sequential, not batched) to avoid hitting DuckDuckGo rate limits.

---

### 4.5 Essay Writer Agent (`agents/essay_agent.py`)

**Responsibility:** Write a 3–4 paragraph essay using only VERIFIED claims, with citations.

**Step-by-step logic:**

```
1. Filter state.verified_claims → keep only those where status == VERIFIED.

2. Build a numbered citation list mapping claim index to its source_url
   and corroboration_url.

3. Build a prompt:
   "Write a well-structured 3-4 paragraph essay on '{topic}' using only
    the following verified facts. Use inline citations like [1], [2].
    Facts:
      - Global temps rose 1.1°C since pre-industrial era [1]
      - Arctic ice has shrunk by 13% per decade [2]
      ..."

4. Call Groq LLM (with @llm_retry) to generate the essay.

5. Append the references section at the end. Format uses blank lines
   between entries to guarantee one-per-line rendering in Markdown:

   **[1]** https://bbc.com/article/123
   *Corroborated by:* https://reuters.com/article/456

   **[2]** https://nature.com/article/789
   *Corroborated by:* https://nasa.gov/article/012

6. Write result into state.essay and return state.
```

**Key decisions:**
- If there are zero VERIFIED claims, the agent writes a short "insufficient verified information" message rather than hallucinating.
- Essay prompt explicitly forbids adding facts not in the provided list, minimising hallucination.
- Blank lines between reference entries ensure each entry renders on its own line in any Markdown renderer, regardless of whether the renderer treats single newlines as hard breaks.

---

## 5. LangGraph Wiring (`graph.py`)

LangGraph treats each agent as a **node** in a directed graph. Edges define execution order. Multiple edges from `START` to different nodes cause LangGraph to execute those nodes **in parallel**.

### 5.1 Partial Graph (used inside the boost loop)

`_build_partial_graph()` constructs a graph **without** the essay writer node:

```
START → search_news     ─┐
START → search_academic  ├─→ merge_sources → extract → verify → END
START → search_general  ─┘
```

This graph is invoked by `_run_partial_pipeline(topic, num_claims)` for each round of the verification boost loop.

### 5.2 Full Graph (backward compatibility)

`build_graph()` builds the complete graph including the essay writer node, identical to the original single-pass design. It remains available for direct invocation when the boost loop orchestration is not needed.

```
START → search_news     ─┐
START → search_academic  ├─→ merge_sources → extract → verify → write → END
START → search_general  ─┘
```

### 5.3 `run_pipeline()` — Verification Boost Loop

`run_pipeline(topic, num_claims)` is the primary programmatic entry point. It orchestrates multiple partial pipeline rounds to achieve a minimum verification rate before writing the essay.

```
Constants:
  VERIFICATION_THRESHOLD = 0.50   # 50% of accumulated claims must be VERIFIED
  MAX_PIPELINE_ROUNDS    = 3      # never run more than 3 rounds

Algorithm:
  accumulated_claims = []    # unique verified claims collected across rounds
  all_claims         = []    # all claims (verified + unverified) across rounds

  for round in 1..MAX_PIPELINE_ROUNDS:

    1. Run _run_partial_pipeline(topic, num_claims)
       → produces state with .claims and .verified_claims for this round

    2. Deduplicate: add only claims whose text is not already in
       accumulated_claims (dedup by claim text)

    3. Check rate = len(verified_in_accumulated) / len(all_in_accumulated)

    4. If rate >= VERIFICATION_THRESHOLD → break (enough quality)

    5. If rate < VERIFICATION_THRESHOLD AND rounds remaining → loop again

  After loop:
    6. Merge accumulated verified_claims into a final merged_state
    7. Call essay_agent.run(merged_state) once to produce the essay
    8. Return final state
```

This loop ensures the essay is written from a pool of claims that meets a minimum quality bar. If the first round yields poor verification (e.g. 30%), another round's fresh sources and claims are accumulated before essay writing.

### 5.4 Conceptual Code Structure

```python
# Partial graph (no essay node)
def _build_partial_graph():
    graph = StateGraph(dict)
    graph.add_node("search_news",     lambda s: search_agent.run(s, query_angle="news"))
    graph.add_node("search_academic", lambda s: search_agent.run(s, query_angle="academic"))
    graph.add_node("search_general",  lambda s: search_agent.run(s, query_angle="general"))
    graph.add_node("merge_sources",   _merge_sources_node)
    graph.add_node("extract",         _extraction_node)
    graph.add_node("verify",          _verification_node)
    graph.add_edge(START, "search_news")
    graph.add_edge(START, "search_academic")
    graph.add_edge(START, "search_general")
    graph.add_edge("search_news",     "merge_sources")
    graph.add_edge("search_academic", "merge_sources")
    graph.add_edge("search_general",  "merge_sources")
    graph.add_edge("merge_sources",   "extract")
    graph.add_edge("extract",         "verify")
    graph.add_edge("verify",          END)
    return graph.compile()

# Full graph (includes essay node, for backward compat)
def build_graph():
    graph = StateGraph(dict)
    # ... same nodes + "write" node ...
    graph.add_edge("verify", "write")
    graph.add_edge("write",  END)
    return graph.compile()
```

### 5.5 How Execution Works

1. `run_pipeline(topic, num_claims)` is called (e.g. by the Streamlit UI).
2. For each round, `_run_partial_pipeline` invokes the partial compiled graph.
3. LangGraph detects three edges from `START` → launches `search_news`, `search_academic`, `search_general` **concurrently** (in separate threads).
4. Each worker collects ≥5 sources and returns its partial state dict.
5. Once **all three** workers finish, LangGraph calls `merge_sources` with their combined outputs.
6. `merge_sources` deduplicates by URL and writes the merged list to `state["raw_sources"]`.
7. Continues sequentially: `extract` → `verify`.
8. `run_pipeline` accumulates results across rounds, checks the threshold, and calls `essay_agent.run()` once after the loop.

### 5.6 Why LangGraph Instead of Plain Function Calls?

- **Native parallelism** — edges from `START` to multiple nodes triggers concurrent execution with no manual threading code.
- **Built-in state merging** — LangGraph handles passing each parallel node's output into the merge node automatically.
- **Inspectable** — `.get_graph().draw_ascii()` prints the full topology for debugging.
- **Extensible** — conditional edges can be added later (e.g. re-run search if merge yields fewer than 10 sources).

---

## 6. Model Fallback (`config.py` + each agent)

### 6.1 Purpose

Groq LLM calls can exhaust all retry attempts under sustained rate limiting or API errors. Rather than failing the entire pipeline, each agent falls back to a smaller, lighter model that is less likely to be rate-limited.

### 6.2 Configuration

```
Settings (config.py)
├── groq_model          str   Primary model (env: GROQ_MODEL)
└── groq_fallback_model str   "llama-3.1-8b-instant" (env: GROQ_FALLBACK_MODEL)
```

### 6.3 Pattern in Each Agent

Every agent implements `_build_llm()` (primary) and `_build_fallback_llm()` (fallback). The `run()` function wraps LLM calls with a try/except:

```python
try:
    result = _llm_call(..., llm=_build_llm())       # primary, 5-retry budget
except Exception:
    result = _llm_call(..., llm=_build_fallback_llm())  # fallback, 5-retry budget
```

Both the primary and fallback calls are decorated with `@llm_retry`, so the fallback also gets its own 5-attempt budget with exponential backoff before the agent gives up entirely.

### 6.4 Why a Separate Fallback Model?

- The primary model (e.g. `llama-3.3-70b`) is more capable but has tighter rate limits on Groq's free tier.
- The fallback model (`llama-3.1-8b-instant`) is smaller, faster, and less frequently rate-limited, making it a reliable safety net.
- Keeping fallback logic inside each agent (rather than in the retry decorator) keeps the retry decorator simple and reusable.

---

## 7. Streamlit Web UI (`ui.py`)

### 7.1 Purpose

The UI gives users a browser-based way to run the full pipeline while watching every internal step happen in real time. It calls `run_pipeline()` directly — the same orchestrator used programmatically — with no separate CLI layer.

### 7.2 Layout

```
┌─────────────────────┬──────────────────────────────────────────────┐
│  SIDEBAR            │  MAIN AREA                                   │
│                     │                                              │
│  Topic:             │  ◉ Search Agent (running...)                 │
│  [____________]     │  ┌─scrollable 300px──────────────────────┐  │
│                     │  │ ✓ bbc.com (4,231 chars)               │  │
│  Num Claims: [8]    │  │ ✓ reuters.com (3,890 chars)           │  │
│                     │  └───────────────────────────────────────┘  │
│  [Run Research]     │                                              │
│                     │  ◉ Extraction Agent (running...)             │
│                     │  ┌─scrollable 250px──────────────────────┐  │
│                     │  │ • Global temps rose 1.1°C [bbc.com]   │  │
│                     │  └───────────────────────────────────────┘  │
│                     │                                              │
│                     │  ◉ Verification Agent (running...)           │
│                     │  ┌─scrollable 300px──────────────────────┐  │
│                     │  │ VERIFIED   | claim... | reuters.com   │  │
│                     │  └───────────────────────────────────────┘  │
│                     │                                              │
│                     │  ◉ Essay Writer (running...)                 │
│                     │  ┌─scrollable 200px──────────────────────┐  │
│                     │  │ Climate change represents...           │  │
│                     │  └───────────────────────────────────────┘  │
│                     │                                              │
│                     │  ── Final Output ──                          │
│                     │    [Full essay as Markdown]                  │
│                     │    [Verification summary table]              │
│                     │    [Download Essay .txt]                     │
└─────────────────────┴──────────────────────────────────────────────┘
```

### 7.3 `st.status()` Instead of `st.expander()`

Each agent section uses `st.status()` (not `st.expander()`). `st.status()` is designed for real-time streaming writes during blocking calls and shows a live spinner while the agent runs. After the pipeline completes, each status section is marked `state="complete"` and automatically collapsed.

Inside each status block, a `st.container(height=N)` creates a fixed-height scrollable area:

| Agent section | Container height |
|---|---|
| Search Agent | 300 px |
| Extraction Agent | 250 px |
| Verification Agent | 300 px |
| Essay Writer | 200 px |

There is no outer `st.spinner()` wrapper — each `st.status()` block shows its own spinner while the corresponding agent is running.

### 7.4 Live Log Capture — `StreamlitLogHandler`

The key mechanism that makes the UI verbose is a custom `logging.Handler` subclass:

```
StreamlitLogHandler
│
├── Attached to the root logger at app startup
├── Each agent already logs via logging.getLogger(__name__)
│     → these records are automatically captured
│
├── _resolve(record) → maps logger name to the correct st.container()
│     Returns None for unknown loggers (prevents HTTP/library logs
│     from rendering in the main page area)
│
├── on each log record:
│     INFO    → st.write("ℹ " + message)          normal text
│     WARNING → st.warning("⚠ " + message)        amber box
│     ERROR   → st.error("✖ " + message)          red box
│
└── writes into the active st.container() for the current agent section
```

No changes are needed in the agent code — because they already use Python's standard `logging` module, the handler intercepts their output automatically.

### 7.5 Noisy Logger Suppression

Before the pipeline runs, the UI suppresses third-party loggers to `WARNING` level to prevent HTTP-level noise from cluttering the agent log sections:

```python
SUPPRESSED_LOGGERS = [
    "httpx", "httpcore", "trafilatura", "urllib3", "urllib",
    "requests", "groq", "_client", "charset_normalizer",
    "LangChain", "langchain", "langsmith", "openai",
]
for name in SUPPRESSED_LOGGERS:
    logging.getLogger(name).setLevel(logging.WARNING)
```

This runs immediately before each `run_pipeline()` call.

### 7.6 Execution Model

```
User clicks "Run Research"
        │
        ▼
st.session_state["running"] = True
        │
        ▼
Noisy loggers suppressed to WARNING
StreamlitLogHandler attached to root logger
        │
        ▼
run_pipeline(topic, num_claims)
  [runs synchronously in the Streamlit main thread]
  [each agent's log.info() calls → StreamlitLogHandler → st.write()]
        │
        ▼
Each st.status() block marked state="complete" and collapsed
        │
        ▼
st.session_state["result"] = final_state
st.session_state["running"] = False
        │
        ▼
Final output section rendered (essay + table + download button)
```

**Why synchronous?** Streamlit's execution model reruns the script top-to-bottom on every interaction. Running the graph synchronously keeps the state simple — `st.session_state` holds the result and log buffer between reruns.

### 7.7 How to Launch

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
│       ├── config.py             # Settings, get_settings(), groq_fallback_model
│       ├── models.py             # RawSource, Claim, VerifiedClaim, ResearchState
│       ├── retry.py              # @llm_retry, @web_retry
│       ├── graph.py              # LangGraph wiring, run_pipeline(), build_graph()
│       ├── ui.py                 # Streamlit web UI
│       └── agents/
│           ├── __init__.py
│           ├── search_agent.py       # EXCLUDED_DOMAINS, domain-first filtering
│           ├── extraction_agent.py   # SOURCE_URL header, claim spreading
│           ├── verification_agent.py # Two-pass search, relevance gating
│           └── essay_agent.py        # Blank-line separated references
│
└── tests/
    ├── __init__.py
    ├── test_retry.py
    ├── test_search_agent.py
    ├── test_extraction_agent.py
    ├── test_verification_agent.py
    ├── test_essay_agent.py
    └── test_integration.py
```

---

## 9. Data Flow Diagram (end-to-end)

```
UI: topic="climate change", num_claims=8
      |
      | run_pipeline("climate change", 8)
      v
      Boost loop — up to 3 rounds of _run_partial_pipeline()
      |
      | Round 1: _build_partial_graph().invoke(initial_state)
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
│ domain check │ │ domain check │ │ domain check │
│ (skip excl.) │ │ (skip excl.) │ │ (skip excl.) │
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
│  Combine all raw text with SOURCE_URL: headers               │
│  → "SOURCE_URL: https://bbc.com\n{text}\n\n..."              │
│                    |                                         │
│                    v                                         │
│               Groq LLM (structured output)                   │
│               Primary model → fallback if exhausted          │
│                    |                                         │
│                    v                                         │
│  list[Claim] = [                                             │
│    { claim: "...", source_url: "...", source_domain: "..." } │
│    × 8 claims (spread across sources, correct URLs)          │
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
│    Pass 1: DuckDuckGo(first 10 words of claim), 15 results   │
│    Pass 2: DuckDuckGo(full claim text), 15 results           │
│                                                              │
│    For each result:                                          │
│      domain == claim.source_domain? → skip                   │
│      _is_topically_relevant()? → named-entity gate +         │
│                                   keyword gate               │
│      Both pass → VERIFIED, record corroboration_url          │
│                                                              │
│    No valid result → UNVERIFIED                              │
│                                                              │
│  state.verified_claims = [VerifiedClaim × 8]                 │
└──────────────────────────────────────────────────────────────┘
          |
          v
      run_pipeline() accumulates across rounds
      Check verified_count / total >= 50%
      If YES or MAX_ROUNDS reached → proceed
          |
          v
┌──────────────────────────────────────────────────────────────┐
│ ESSAY WRITER AGENT                                           │
│                                                              │
│  Filter: only VERIFIED claims (e.g. 6 out of 8)             │
│  Build citation list: [1] url1, [2] url2, ...                │
│  Prompt Groq LLM → 3-4 paragraph essay with [N] citations   │
│  Primary model → fallback if exhausted                       │
│                                                              │
│  References section (blank lines between entries):           │
│  **[1]** https://bbc.com/article/123                         │
│  *Corroborated by:* https://reuters.com/article/456          │
│                                                              │
│  **[2]** https://nature.com/article/789                      │
│  *Corroborated by:* https://nasa.gov/article/012             │
│                                                              │
│  state.essay = "Climate change represents one of the most..." │
└──────────────────────────────────────────────────────────────┘
          |
          v
     ┌────────────────────────────────────┐
     │  Streamlit UI (browser)            │
     │  - live log lines per agent        │
     │  - per-agent st.status() sections  │
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
If retry behavior needs to change (e.g. increase max attempts), it changes in one place. If it were copy-pasted into each agent, you'd have four places to update and four places to forget.

**Why cross-domain verification?**
A claim sourced from `bbc.com` that is also on `bbc.com` proves nothing — it's the same source. A claim that appears on both `bbc.com` and `reuters.com` is genuinely corroborated by an independent source.

**Why the two-gate relevance check in Verification?**
A naive domain check allows false positives — e.g. mayoclinic.org corroborating a claim about "MS Dhoni" because "MS" matches "Multiple Sclerosis" in their content. The named-entity gate prevents this by requiring at least one capitalised proper noun from the claim to appear in the result. The keyword gate adds a second layer, requiring shared content vocabulary. Together they eliminate irrelevant corroborators without over-filtering genuine matches.

**Why a verification boost loop instead of a single pass?**
A single pipeline round may produce a poor verification rate due to sparse or low-quality sources on a given topic. The boost loop runs up to three partial rounds, accumulating unique verified claims until 50% of all claims pass verification. This gives the essay writer a higher-quality pool without requiring the user to manually retry.

**Why exclude Wikipedia and similar domains from Search?**
Wikipedia and community Q&A sites (Quora, Ask.com, Answers.com) are secondary sources that aggregate information from primary sources. Fetching them adds noise — the same information appears in better form from the original primary sources that DuckDuckGo also returns. Checking the domain before fetching (rather than after) avoids unnecessary network calls.

**Why parallel search workers instead of one sequential search agent?**
The search phase is the slowest part of the pipeline — each of the 15+ sources requires a DuckDuckGo call plus a `trafilatura` page fetch. Running three concurrent workers (news / academic / general) cuts the wall-clock time for the search phase from ~3× to ~1× while also producing more diverse sources. Each worker focuses on a distinct query angle, so the resulting `raw_sources` pool has better coverage than if the same 15 queries all came from one undirected search.

**Why only parallelise search and not verification?**
Verification is slower per-claim but bounded — `num_claims` is typically small (8–20) and each DuckDuckGo call is fast. More importantly, splitting verification into batches requires order-preserving merging of `list[VerifiedClaim]` to match claim indices, adding fiddly merge logic. The search parallelism gives the largest absolute time saving for the least implementation complexity.

**Why `st.status()` instead of `st.expander()` in the UI?**
`st.status()` is purpose-built for showing progress during blocking operations — it displays a spinner while running, streams content in real time, and collapses cleanly when done. `st.expander()` is a static disclosure widget that does not support live streaming updates reliably during a blocking call.

**Why is there no CLI?**
The CLI was deprioritised. `run_pipeline()` in `graph.py` serves as the programmatic entry point and is called directly by the Streamlit UI. A CLI wrapper could be added later by calling `run_pipeline()` from a `main.py` argparse script, but no such file exists in the current codebase.

---

## 12. Debugging Guide

### 12.1 Reading Log Output

Every agent uses Python's standard `logging` module with a consistent format. Enable verbose logging by setting the log level before running:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or in the Streamlit UI, all `INFO` and `WARNING` lines are automatically surfaced per agent section via `StreamlitLogHandler` — no extra setup needed.

**Log line format:**

```
[LEVEL] module_name: message
```

**Key log patterns to watch for:**

| Log message | What it means |
|---|---|
| `=== Search Agent [NEWS] starting ===` | Parallel search worker started |
| `[NEWS] Generated 5 queries for topic '...'` | LLM successfully generated queries |
| `[NEWS] Skipping excluded domain: wikipedia.org` | Domain blocked before fetch |
| `[NEWS] Collected source 3: bbc.com (4231 chars)` | Successfully scraped a source |
| `[NEWS] Only collected 3/5 sources — proceeding anyway` | Worker fell short; merger may still hit 15 total |
| `=== merge_sources: combining parallel search results ===` | Fan-in node fired after all 3 workers finished |
| `=== merge_sources complete: 14 unique sources from 16 total ===` | 2 duplicate URLs removed across workers |
| `Extracting 8 claims from 14 sources (38420 chars total)` | Extraction Agent about to call LLM |
| `Extracted claim 1: "Global temps rose..."` | Individual claim logged after extraction |
| `Expected 8 claims, got 6 — proceeding with what was returned` | LLM returned fewer claims than requested |
| `Pass 1 search for claim: "Global temps..."` | Verification first pass (short query) |
| `Pass 2 fallback search for claim: "Global temps..."` | Verification second pass (full claim) |
| `Relevance gate failed (entity): ...` | Named-entity gate blocked a result |
| `VERIFIED: 'Global temps rose...' — corroborated by reuters.com` | Cross-domain source found and passed both gates |
| `UNVERIFIED: 'CO2 levels...' — no cross-domain source found` | No independent source found for this claim |
| `Verification complete: 5/8 claims verified` | Summary before essay writing |
| `Round 1 rate: 0.375 — below threshold, running round 2` | Boost loop triggered a second round |
| `Verification threshold met after round 2` | Boost loop exited early |
| `Falling back to fallback LLM` | Primary LLM exhausted all retries |
| `No verified claims available — writing insufficient-info message` | All claims failed verification |

---

### 12.2 Debugging the Retry Behaviour

Each retry emits a `WARNING` log with the attempt number and wait duration:

```
WARNING retry: Retry attempt 1 | waiting 2.0s | reason: RateLimitError: ...
WARNING retry: Retry attempt 2 | waiting 4.0s | reason: RateLimitError: ...
```

**Retry configuration at a glance:**

| Decorator | Max attempts | Backoff range | Exceptions caught |
|---|---|---|---|
| `@llm_retry` | 5 | 2s → 16s (exponential) | `groq.RateLimitError`, `groq.APIStatusError` |
| `@web_retry` | 3 | 1s → 4s (exponential) | `httpx.HTTPError`, `httpx.TimeoutException`, `requests.RequestException` |

**Model fallback sequence:**
1. Primary model call → `@llm_retry` (up to 5 attempts)
2. If all 5 fail → `except Exception` triggers in agent `run()`
3. Fallback model call → `@llm_retry` (up to 5 more attempts)
4. If all 5 fail → exception propagates to the caller

**If you see persistent `RateLimitError` retries:**
- Groq free tier has per-minute token limits; 5 retries × 16s = ~80s of waiting before switching to fallback
- The fallback model (`llama-3.1-8b-instant`) will then be tried
- If both models are rate-limited, reduce `num_claims` (fewer LLM calls needed) or wait 60s for the window to reset

**If you see persistent `web_retry` retries:**
- DuckDuckGo occasionally blocks rapid requests — the `SEARCH_DELAY` in the Verification Agent mitigates this
- Try a different network or add a VPN if DuckDuckGo is consistently rejecting requests
- Check `trafilatura` failures: some sites (paywalls, JS-heavy pages) will always return `None` — this is expected and logged at `DEBUG` level

---

### 12.3 Debugging Individual Agents in Isolation

Each agent's `run()` function accepts a `ResearchState` directly, so you can test any stage independently in a Python REPL or script without running the full graph:

```python
from online_research_agents.models import ResearchState, RawSource
from online_research_agents.agents import search_agent, extraction_agent

# Test Search Agent alone
state = ResearchState(topic="quantum computing", num_claims=5)
result = search_agent.run(state, query_angle="academic")
print(f"Sources collected: {len(result.raw_sources)}")
for s in result.raw_sources:
    print(f"  {s.domain}: {len(s.text)} chars")

# Test Extraction Agent alone with hand-crafted sources
state_with_sources = ResearchState(
    topic="quantum computing",
    num_claims=3,
    raw_sources=[
        RawSource(
            url="https://example.com/article",
            domain="example.com",
            text="Quantum computers use qubits instead of classical bits...",
        )
    ],
)
result = extraction_agent.run(state_with_sources)
for claim in result.claims:
    print(claim.claim)
```

---

### 12.4 Inspecting the LangGraph Topology

Print the compiled graph's ASCII topology at runtime to verify the parallel wiring:

```python
from online_research_agents.graph import build_graph
graph = build_graph()
print(graph.get_graph().draw_ascii())
```

Expected output:

```
                +-----------+
                | __start__ |
                +-----------+
          *****/      |      \*****
         *            |            *
        *             |             *
+-------------+ +----------------+ +--------------+
| search_news | | search_academic| | search_general|
+-------------+ +----------------+ +--------------+
          *****\      |      /*****
               \      |      /
                \      |      /
           +----------------+
           | merge_sources  |
           +----------------+
                    |
                +-------+
                | extract|
                +-------+
                    |
                +-------+
                | verify |
                +-------+
                    |
                +-------+
                |  write |
                +-------+
                    |
               +---------+
               | __end__ |
               +---------+
```

---

### 12.5 Common Failure Modes and Fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ValidationError: GROQ_API_KEY missing` | `.env` file not found or key not set | Run `cp .env.example .env` and add your key |
| `groq.RateLimitError` after 5 retries on primary AND fallback | Both Groq models rate-limited simultaneously | Wait 60s, reduce `num_claims`, or upgrade Groq plan |
| `len(raw_sources) < 15` after merge | DuckDuckGo rate limiting or too many paywalled/excluded pages | Try a broader topic; retry after 30s |
| All claims UNVERIFIED | DuckDuckGo returning same-domain results or failing relevance gates | Try a more specific topic; increase `num_claims` to give more chances |
| Verification rate stays below 50% after 3 rounds | Topic is niche with sparse corroboration online | Essay is still written from accumulated claims; consider a broader topic |
| Essay = INSUFFICIENT_INFO_MSG | Zero verified claims across all rounds | See "all claims UNVERIFIED" above |
| `TypeError: unhashable type` in graph | Wrong state schema passed to `StateGraph` | Ensure `_GraphState(TypedDict)` is used, not a plain `dict` |
| `pydantic_core.ValidationError: num_claims >= 1` | `num_claims=0` passed | Minimum is 1; UI slider should prevent this |
| HTTP log lines appearing in main page area | `StreamlitLogHandler._resolve()` returning wrong container | Check `_resolve()` returns `None` for unknown logger names |
