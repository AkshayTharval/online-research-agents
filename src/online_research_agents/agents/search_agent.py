"""Search Agent — one of three parallel workers that gather raw web content."""

import logging
from typing import Literal
from urllib.parse import urlparse

import trafilatura
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from online_research_agents.config import get_settings
from online_research_agents.models import RawSource, ResearchState
from online_research_agents.retry import llm_retry, web_retry

logger = logging.getLogger(__name__)

QueryAngle = Literal["news", "academic", "general"]

# Each worker targets this many sources; three workers → ≥15 total after merge
MIN_SOURCES_PER_WORKER = 5
MAX_RESULTS_PER_QUERY = 10

# Domains excluded from scraping — aggregators and mirrors that dominate
# DuckDuckGo results but are secondary sources themselves.
# Wikipedia cites primary sources; we want those primary sources directly.
EXCLUDED_DOMAINS: frozenset[str] = frozenset({
    "en.wikipedia.org",
    "wikipedia.org",
    "simple.wikipedia.org",
    "en.m.wikipedia.org",
    "wikidata.org",
    "wikimedia.org",
    "wikiwand.com",          # Wikipedia mirror
    "dbpedia.org",           # Wikipedia-derived
    "answers.com",           # content farm
    "ask.com",               # content farm
    "quora.com",             # crowdsourced, low reliability
})

_ANGLE_INSTRUCTIONS: dict[str, str] = {
    "news": (
        "Focus on RECENT NEWS and current events: headlines, latest developments, "
        "breaking stories, policy changes, and real-world impacts reported in the last 1-2 years."
    ),
    "academic": (
        "Focus on RESEARCH AND DATA: scientific studies, peer-reviewed findings, "
        "statistics, expert analyses, reports from institutions, and empirical evidence."
    ),
    "general": (
        "Focus on BROAD COVERAGE: overviews, causes and effects, historical context, "
        "common explanations, solutions, and widely cited general knowledge."
    ),
}


def _build_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model)


@llm_retry
def _generate_queries(topic: str, query_angle: QueryAngle, llm: ChatGroq) -> list[str]:
    """Ask the LLM to produce 5 search queries biased toward the given angle."""
    angle_instruction = _ANGLE_INSTRUCTIONS[query_angle]
    messages = [
        SystemMessage(
            content=(
                "You are a research assistant. Given a topic and a search angle, "
                "return exactly 5 search queries that explore the topic from that angle. "
                f"Search angle: {angle_instruction}\n"
                "Return only the queries, one per line, no numbering, no extra text."
            )
        ),
        HumanMessage(content=f"Topic: {topic}"),
    ]
    response = llm.invoke(messages)
    queries = [q.strip() for q in response.content.strip().splitlines() if q.strip()]
    logger.info(
        "[%s] Generated %d queries for topic '%s'",
        query_angle.upper(),
        len(queries),
        topic,
    )
    return queries[:5]


@web_retry
def _ddg_search(query: str, max_results: int) -> list[dict]:
    """Run a DuckDuckGo text search and return result dicts."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results


@web_retry
def _scrape_url(url: str) -> str | None:
    """Fetch and extract clean article text from a URL via trafilatura."""
    html = trafilatura.fetch_url(url)
    if not html:
        return None
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    return text or None


def _extract_domain(url: str) -> str:
    """Return bare domain (without www.) from a URL."""
    netloc = urlparse(url).netloc
    return netloc.removeprefix("www.")


def run(state: ResearchState, query_angle: QueryAngle = "general") -> ResearchState:
    """
    Search Agent entry point — runs as one of three parallel workers.

    Generates angle-biased queries via Groq LLM, iterates DuckDuckGo results,
    scrapes each page with trafilatura, and returns a partial ResearchState
    with raw_sources populated (≥MIN_SOURCES_PER_WORKER).

    The merge_sources node in graph.py combines outputs from all three workers.

    Args:
        state:       Shared ResearchState (reads topic only).
        query_angle: One of "news", "academic", "general" — controls LLM
                     query generation bias.
    """
    llm = _build_llm()
    queries = _generate_queries(state.topic, query_angle, llm)

    seen_urls: set[str] = set()
    sources: list[RawSource] = []

    for query in queries:
        if len(sources) >= MIN_SOURCES_PER_WORKER:
            break

        logger.info(
            "[%s] Searching: '%s' | sources so far: %d",
            query_angle.upper(),
            query,
            len(sources),
        )

        try:
            results = _ddg_search(query, max_results=MAX_RESULTS_PER_QUERY)
        except Exception as exc:
            logger.warning(
                "[%s] DuckDuckGo search failed for '%s': %s",
                query_angle.upper(),
                query,
                exc,
            )
            continue

        for result in results:
            if len(sources) >= MIN_SOURCES_PER_WORKER:
                break

            url: str = result.get("href", "")
            if not url or url in seen_urls:
                continue

            seen_urls.add(url)

            domain = _extract_domain(url)
            if domain in EXCLUDED_DOMAINS:
                logger.debug(
                    "[%s] Skipping excluded domain: %s",
                    query_angle.upper(),
                    domain,
                )
                continue

            try:
                text = _scrape_url(url)
            except Exception as exc:
                logger.warning(
                    "[%s] Scraping failed for %s: %s",
                    query_angle.upper(),
                    url,
                    exc,
                )
                continue

            if not text:
                logger.debug(
                    "[%s] No text extracted from %s — skipping",
                    query_angle.upper(),
                    url,
                )
                continue

            sources.append(RawSource(url=url, domain=domain, text=text))
            logger.info(
                "[%s] Collected source %d: %s (%d chars)",
                query_angle.upper(),
                len(sources),
                domain,
                len(text),
            )

    if len(sources) < MIN_SOURCES_PER_WORKER:
        logger.warning(
            "[%s] Only collected %d/%d sources — proceeding anyway",
            query_angle.upper(),
            len(sources),
            MIN_SOURCES_PER_WORKER,
        )
    else:
        logger.info(
            "[%s] Worker complete: %d sources collected",
            query_angle.upper(),
            len(sources),
        )

    # Return partial state — only raw_sources updated; merge node combines all workers
    return state.model_copy(update={"raw_sources": sources})
