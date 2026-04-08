"""Search Agent — gathers raw web content from at least 15 distinct sources."""

import logging
from urllib.parse import urlparse

import trafilatura
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from online_research_agents.config import get_settings
from online_research_agents.models import RawSource, ResearchState
from online_research_agents.retry import llm_retry, web_retry

logger = logging.getLogger(__name__)

MIN_SOURCES = 15
MAX_RESULTS_PER_QUERY = 10


def _build_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model)


@llm_retry
def _generate_queries(topic: str, llm: ChatGroq) -> list[str]:
    """Ask the LLM to produce 5 diverse search queries for the topic."""
    messages = [
        SystemMessage(
            content=(
                "You are a research assistant. Given a topic, return exactly 5 "
                "diverse search queries that together cover different angles of the topic. "
                "Return only the queries, one per line, no numbering, no extra text."
            )
        ),
        HumanMessage(content=f"Topic: {topic}"),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()
    queries = [q.strip() for q in raw.splitlines() if q.strip()]
    logger.info("Generated %d search queries for topic '%s'", len(queries), topic)
    return queries[:5]  # guard against over-generation


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


def run(state: ResearchState) -> ResearchState:
    """
    Search Agent entry point.

    Generates diverse queries, iterates DuckDuckGo results, scrapes each page
    with trafilatura, and populates state.raw_sources with at least MIN_SOURCES
    distinct sources.
    """
    llm = _build_llm()
    queries = _generate_queries(state.topic, llm)

    seen_urls: set[str] = set()
    sources: list[RawSource] = []

    for query in queries:
        if len(sources) >= MIN_SOURCES:
            break

        logger.info("Searching: '%s' | sources so far: %d", query, len(sources))

        try:
            results = _ddg_search(query, max_results=MAX_RESULTS_PER_QUERY)
        except Exception as exc:
            logger.warning("DuckDuckGo search failed for query '%s': %s", query, exc)
            continue

        for result in results:
            if len(sources) >= MIN_SOURCES:
                break

            url: str = result.get("href", "")
            if not url or url in seen_urls:
                continue

            seen_urls.add(url)

            try:
                text = _scrape_url(url)
            except Exception as exc:
                logger.warning("Scraping failed for %s: %s", url, exc)
                continue

            if not text:
                logger.debug("No text extracted from %s — skipping", url)
                continue

            domain = _extract_domain(url)
            sources.append(RawSource(url=url, domain=domain, text=text))
            logger.info(
                "Collected source %d: %s (%d chars)", len(sources), domain, len(text)
            )

    if len(sources) < MIN_SOURCES:
        logger.warning(
            "Only collected %d sources (target: %d) — proceeding anyway",
            len(sources),
            MIN_SOURCES,
        )
    else:
        logger.info("Search complete. Total sources collected: %d", len(sources))

    return state.model_copy(update={"raw_sources": sources})
