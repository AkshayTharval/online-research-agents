# Low-Level Design — Multi-Agent Research System

## 1. Overview

The system takes a research **topic** as input, automatically gathers information from the web, extracts and verifies factual claims, and produces a cited essay — all without any human intervention after the initial invocation.

It is structured as a **pipeline of four specialized agents** wired together using **LangGraph**, a framework for building stateful, graph-based AI workflows. Each agent does one focused job. A single shared data structure (`ResearchState`) flows through every agent, with each agent reading what it needs and writing its output back into it.

---

## 2. High-Level Flow

```
User CLI Input
  --topic "climate change"
  --num-claims 8
        |
        v
  ResearchState (initial)
  { topic, num_claims }
        |
        v
 ┌─────────────────┐
 │  Search Agent   │  Gathers raw web content (≥15 sources)
 └────────┬────────┘
          |  state.raw_sources filled
          v
 ┌─────────────────────┐
 │  Extraction Agent   │  Pulls N factual claims from sources
 └──────────┬──────────┘
            |  state.claims filled
            v
 ┌──────────────────────┐
 │  Verification Agent  │  Cross-checks each claim on the web
 └──────────┬───────────┘
            |  state.verified_claims filled
            v
 ┌─────────────────────┐
 │  Essay Writer Agent │  Writes essay using only verified claims
 └──────────┬──────────┘
            |  state.essay filled
            v
     Final Output (stdout)
     - Essay with inline citations
     - Verification summary
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

**Responsibility:** Gather raw text content from at least 15 distinct web sources for the given topic.

**Step-by-step logic:**

```
1. Call Groq LLM (with @llm_retry) to generate 5 diverse search queries
   for the topic.
   e.g. topic = "climate change" →
        queries = [
          "climate change scientific evidence",
          "global warming effects 2024",
          "greenhouse gas emissions data",
          "climate change policy solutions",
          "IPCC report findings"
        ]

2. For each query, call DuckDuckGo (with @web_retry) to get search results
   (URLs + snippets). Paginate if needed.

3. For each unique URL not yet visited:
   a. Call trafilatura.fetch_url(url)   ← downloads the page (with @web_retry)
   b. Call trafilatura.extract(html)    ← strips nav/ads, returns clean text
   c. If text is non-empty:
      - Parse domain from URL
      - Append RawSource(url, domain, text) to collected list

4. Keep looping across queries/pages until len(collected) >= 15.

5. Write collected list into state.raw_sources and return state.
```

**Key decisions:**
- Deduplication is by URL (a `set` of seen URLs).
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

LangGraph treats each agent as a **node** in a directed graph. Edges define execution order. The state flows from node to node automatically.

```python
# Conceptual structure (not exact code)

graph = StateGraph(ResearchState)

graph.add_node("search",    search_agent.run)
graph.add_node("extract",   extraction_agent.run)
graph.add_node("verify",    verification_agent.run)
graph.add_node("write",     essay_agent.run)

graph.set_entry_point("search")
graph.add_edge("search",  "extract")
graph.add_edge("extract", "verify")
graph.add_edge("verify",  "write")
graph.set_finish_point("write")

compiled_graph = graph.compile()
```

**How execution works:**
1. `compiled_graph.invoke(initial_state)` is called.
2. LangGraph calls `search_agent.run(state)` → gets back updated state.
3. Automatically calls `extraction_agent.run(state)` with the updated state.
4. Continues through `verify` → `write`.
5. Returns the final `ResearchState` with all fields populated.

**Why LangGraph instead of plain function calls?**
- Built-in state management and serialization.
- Easy to add conditional edges later (e.g., retry search if fewer than 15 sources found).
- Graph is inspectable and debuggable (`.get_graph().draw_ascii()`).

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

## 7. File Structure

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
│       └── agents/
│           ├── __init__.py
│           ├── search_agent.py      # Task 3
│           ├── extraction_agent.py  # Task 4
│           ├── verification_agent.py # Task 5
│           └── essay_agent.py       # Task 6
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

## 8. Data Flow Diagram (end-to-end)

```
CLI: --topic "climate change" --num-claims 8
          |
          | Creates ResearchState { topic, num_claims=8 }
          v
┌──────────────────────────────────────────────────────────────┐
│ SEARCH AGENT                                                 │
│                                                              │
│  Groq LLM ──generates──> 5 search queries                   │
│                                   |                          │
│              ┌────────────────────┘                          │
│              | for each query                                │
│              v                                               │
│         DuckDuckGo ──returns──> list of URLs                 │
│              |                                               │
│              | for each URL                                  │
│              v                                               │
│         trafilatura ──scrapes──> clean article text          │
│              |                                               │
│              └──> RawSource { url, domain, text }            │
│                         (repeat until 15 sources)            │
│                                                              │
│  state.raw_sources = [RawSource × 15+]                       │
└──────────────────────────────────────────────────────────────┘
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
     stdout: essay + verification table
```

---

## 9. External Dependencies and Why Each Was Chosen

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

---

## 10. Key Design Decisions

**Why not one big function?**
Separating into four agents means each is independently testable. You can mock just DuckDuckGo for the Search Agent tests without touching the LLM at all.

**Why is `ResearchState` a Pydantic model?**
LangGraph requires the state to be serializable. Pydantic ensures every field has the right type before it's passed to the next agent — if the Extraction Agent produces a malformed `Claim`, it fails loudly at the boundary rather than corrupting downstream agents silently.

**Why is retry logic in a shared module?**
If retry behavior needs to change (e.g., increase max attempts), it changes in one place. If it were copy-pasted into each agent, you'd have four places to update and four places to forget.

**Why cross-domain verification?**
A claim sourced from `bbc.com` that is also on `bbc.com` proves nothing — it's the same source. A claim that appears on both `bbc.com` and `reuters.com` is genuinely corroborated by an independent source.
