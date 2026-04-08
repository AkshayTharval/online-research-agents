"""LangGraph state graph — parallel search workers + sequential extract/verify/write."""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from online_research_agents.agents import (
    essay_agent,
    extraction_agent,
    search_agent,
    verification_agent,
)
from online_research_agents.models import RawSource, ResearchState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parallel search worker nodes
# Each wraps search_agent.run() with a fixed query_angle.
# LangGraph runs all three simultaneously when edges from START fan out.
# ---------------------------------------------------------------------------

def _search_news_node(state: dict[str, Any]) -> dict[str, Any]:
    logger.info("=== Search Agent [NEWS] starting ===")
    result = search_agent.run(ResearchState(**state), query_angle="news")
    logger.info("=== Search Agent [NEWS] complete: %d sources ===", len(result.raw_sources))
    return {"raw_sources": [s.model_dump() for s in result.raw_sources]}


def _search_academic_node(state: dict[str, Any]) -> dict[str, Any]:
    logger.info("=== Search Agent [ACADEMIC] starting ===")
    result = search_agent.run(ResearchState(**state), query_angle="academic")
    logger.info("=== Search Agent [ACADEMIC] complete: %d sources ===", len(result.raw_sources))
    return {"raw_sources": [s.model_dump() for s in result.raw_sources]}


def _search_general_node(state: dict[str, Any]) -> dict[str, Any]:
    logger.info("=== Search Agent [GENERAL] starting ===")
    result = search_agent.run(ResearchState(**state), query_angle="general")
    logger.info("=== Search Agent [GENERAL] complete: %d sources ===", len(result.raw_sources))
    return {"raw_sources": [s.model_dump() for s in result.raw_sources]}


# ---------------------------------------------------------------------------
# Merge node — fan-in after the three parallel search workers
# LangGraph accumulates raw_sources lists from all three workers before
# calling this node. We deduplicate by URL and write the combined list.
# ---------------------------------------------------------------------------

def _merge_sources_node(state: dict[str, Any]) -> dict[str, Any]:
    logger.info("=== merge_sources: combining parallel search results ===")

    raw: list[dict] = state.get("raw_sources", [])
    seen_urls: set[str] = set()
    merged: list[dict] = []

    for source in raw:
        url = source.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            merged.append(source)

    logger.info(
        "=== merge_sources complete: %d unique sources from %d total ===",
        len(merged),
        len(raw),
    )
    return {"raw_sources": merged}


# ---------------------------------------------------------------------------
# Sequential pipeline nodes
# ---------------------------------------------------------------------------

def _extraction_node(state: dict[str, Any]) -> dict[str, Any]:
    logger.info("=== Extraction Agent starting ===")
    result = extraction_agent.run(ResearchState(**state))
    logger.info("=== Extraction Agent complete: %d claims ===", len(result.claims))
    return result.model_dump()


def _verification_node(state: dict[str, Any]) -> dict[str, Any]:
    logger.info("=== Verification Agent starting ===")
    result = verification_agent.run(ResearchState(**state))
    verified = sum(1 for c in result.verified_claims if c.status.value == "VERIFIED")
    logger.info(
        "=== Verification Agent complete: %d/%d verified ===",
        verified,
        len(result.verified_claims),
    )
    return result.model_dump()


def _essay_node(state: dict[str, Any]) -> dict[str, Any]:
    logger.info("=== Essay Writer Agent starting ===")
    result = essay_agent.run(ResearchState(**state))
    logger.info("=== Essay Writer Agent complete: %d chars ===", len(result.essay))
    return result.model_dump()


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph for the research pipeline.

    Graph topology:
        START → search_news     ─┐
        START → search_academic  ├─→ merge_sources → extract → verify → write → END
        START → search_general  ─┘

    The three search nodes run in parallel; merge_sources fans them back in.

    Returns:
        A compiled LangGraph graph ready for .invoke(state_dict).
    """
    # Use a dict-based state with a reducer on raw_sources so LangGraph
    # accumulates lists from parallel nodes before calling merge_sources.
    from typing import Annotated
    import operator

    def _merge_lists(a: list, b: list) -> list:
        """Reducer: concatenate raw_sources lists from parallel nodes."""
        return (a or []) + (b or [])

    # Annotated state schema — raw_sources uses a custom list-merge reducer
    # so parallel node outputs are concatenated rather than overwritten.
    class _State(dict):
        pass

    from langgraph.graph.state import StateGraph as _SG

    graph: Any = StateGraph(
        # Inline TypedDict-style annotation for the reducer
        # LangGraph accepts a plain dict schema with Annotated reducers
        {
            "topic": str,
            "num_claims": int,
            "raw_sources": Annotated[list, _merge_lists],
            "claims": list,
            "verified_claims": list,
            "essay": str,
        }
    )

    # Parallel search worker nodes
    graph.add_node("search_news",     _search_news_node)
    graph.add_node("search_academic", _search_academic_node)
    graph.add_node("search_general",  _search_general_node)

    # Merge + sequential pipeline nodes
    graph.add_node("merge_sources", _merge_sources_node)
    graph.add_node("extract",       _extraction_node)
    graph.add_node("verify",        _verification_node)
    graph.add_node("write",         _essay_node)

    # Fan-out: START → three parallel workers
    graph.add_edge(START, "search_news")
    graph.add_edge(START, "search_academic")
    graph.add_edge(START, "search_general")

    # Fan-in: all three workers → merge
    graph.add_edge("search_news",     "merge_sources")
    graph.add_edge("search_academic", "merge_sources")
    graph.add_edge("search_general",  "merge_sources")

    # Sequential pipeline after merge
    graph.add_edge("merge_sources", "extract")
    graph.add_edge("extract",       "verify")
    graph.add_edge("verify",        "write")
    graph.add_edge("write",         END)

    compiled = graph.compile()
    logger.info("Research pipeline graph compiled (parallel search workers enabled)")
    return compiled


def run_pipeline(topic: str, num_claims: int = 8) -> ResearchState:
    """
    Convenience wrapper: build the graph, invoke it, return a ResearchState.

    Args:
        topic:      The research topic string.
        num_claims: Number of claims to extract (default 8).

    Returns:
        Fully populated ResearchState after all agents have run.
    """
    initial = ResearchState(topic=topic, num_claims=num_claims)
    graph = build_graph()
    result_dict = graph.invoke(initial.model_dump())
    return ResearchState(**result_dict)
