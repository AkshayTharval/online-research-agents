"""LangGraph state graph — parallel search workers + sequential extract/verify/write."""

import logging
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from online_research_agents.agents import (
    essay_agent,
    extraction_agent,
    search_agent,
    verification_agent,
)
from online_research_agents.models import ResearchState, VerificationStatus, VerifiedClaim


# ---------------------------------------------------------------------------
# TypedDict state schema for LangGraph
# Annotated[list, operator.add] on raw_sources tells LangGraph to concatenate
# lists returned by the three parallel search nodes before merge_sources fires.
# ---------------------------------------------------------------------------

class _GraphState(TypedDict):
    topic: str
    num_claims: int
    raw_sources: Annotated[list, operator.add]
    claims: list
    verified_claims: list
    essay: str

logger = logging.getLogger(__name__)

# Minimum fraction of claims that must be VERIFIED before writing the essay.
# If a round falls below this, another search-extract-verify round is triggered.
VERIFICATION_THRESHOLD: float = 0.50
# Hard cap on total rounds (initial + boost rounds) to avoid indefinite looping.
MAX_PIPELINE_ROUNDS: int = 3


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
    graph: Any = StateGraph(_GraphState)

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


def _build_partial_graph() -> Any:
    """
    Compile a graph that runs search → merge → extract → verify but stops
    before the essay writer.  Used by the retry loop in run_pipeline so that
    the essay is only written once, after enough claims are verified.
    """
    graph: Any = StateGraph(_GraphState)

    graph.add_node("search_news",     _search_news_node)
    graph.add_node("search_academic", _search_academic_node)
    graph.add_node("search_general",  _search_general_node)
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


def _run_partial_pipeline(topic: str, num_claims: int) -> ResearchState:
    """Run one search-extract-verify round and return the resulting state."""
    partial = _build_partial_graph()
    initial = ResearchState(topic=topic, num_claims=num_claims)
    result_dict = partial.invoke(initial.model_dump())
    return ResearchState(**result_dict)


def run_pipeline(topic: str, num_claims: int = 8) -> ResearchState:
    """
    Orchestrate the full research pipeline with a verification boost loop.

    Runs search → extract → verify up to MAX_PIPELINE_ROUNDS times, merging
    newly discovered verified claims each round.  Once at least
    VERIFICATION_THRESHOLD of all accumulated claims are VERIFIED (or the
    round cap is hit), runs the essay writer once on the final merged pool.

    Args:
        topic:      The research topic string.
        num_claims: Claims to extract per round (default 8).

    Returns:
        Fully populated ResearchState including the final essay.
    """
    all_verified: list[VerifiedClaim] = []
    seen_claim_texts: set[str] = set()
    last_round_state: ResearchState | None = None

    for round_num in range(1, MAX_PIPELINE_ROUNDS + 1):
        logger.info("=== Pipeline round %d/%d starting ===", round_num, MAX_PIPELINE_ROUNDS)

        round_state = _run_partial_pipeline(topic, num_claims)
        last_round_state = round_state

        # Merge unique verified claims from this round into the running pool
        for vc in round_state.verified_claims:
            if vc.claim not in seen_claim_texts:
                seen_claim_texts.add(vc.claim)
                all_verified.append(vc)

        total = len(all_verified)
        verified_count = sum(
            1 for vc in all_verified if vc.status == VerificationStatus.VERIFIED
        )
        rate = verified_count / total if total > 0 else 0.0

        logger.info(
            "=== Round %d complete: %d/%d verified overall (%.0f%%) ===",
            round_num, verified_count, total, 100 * rate,
        )

        if rate >= VERIFICATION_THRESHOLD:
            logger.info("=== Verification threshold met — proceeding to essay writer ===")
            break

        if round_num < MAX_PIPELINE_ROUNDS:
            logger.info(
                "=== Only %.0f%% verified (threshold %.0f%%) — running boost round %d ===",
                100 * rate, 100 * VERIFICATION_THRESHOLD, round_num + 1,
            )
        else:
            logger.warning(
                "=== Max rounds reached with %.0f%% verified — proceeding anyway ===",
                100 * rate,
            )

    # Build a merged state using the last round's sources/claims as context,
    # but with ALL accumulated verified claims (across all rounds).
    assert last_round_state is not None
    merged_state = last_round_state.model_copy(update={"verified_claims": all_verified})

    logger.info("=== Essay Writer starting ===")
    return essay_agent.run(merged_state)
