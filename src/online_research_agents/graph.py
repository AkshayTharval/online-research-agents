"""LangGraph state graph — wires all four agents into a sequential pipeline."""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from online_research_agents.agents import (
    essay_agent,
    extraction_agent,
    search_agent,
    verification_agent,
)
from online_research_agents.models import ResearchState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node functions
# Each node receives the current state dict from LangGraph, converts it to
# a ResearchState, delegates to the agent, and returns the updated dict.
# ---------------------------------------------------------------------------

def _search_node(state: dict[str, Any]) -> dict[str, Any]:
    logger.info("=== Search Agent starting ===")
    result = search_agent.run(ResearchState(**state))
    logger.info("=== Search Agent complete: %d sources ===", len(result.raw_sources))
    return result.model_dump()


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
        START → search → extract → verify → write → END

    Returns a compiled graph ready for .invoke(state_dict).
    """
    graph = StateGraph(dict)

    graph.add_node("search", _search_node)
    graph.add_node("extract", _extraction_node)
    graph.add_node("verify", _verification_node)
    graph.add_node("write", _essay_node)

    graph.add_edge(START, "search")
    graph.add_edge("search", "extract")
    graph.add_edge("extract", "verify")
    graph.add_edge("verify", "write")
    graph.add_edge("write", END)

    compiled = graph.compile()
    logger.info("Research pipeline graph compiled successfully")
    return compiled


def run_pipeline(topic: str, num_claims: int = 8) -> ResearchState:
    """
    Convenience wrapper: build the graph, invoke it, return a ResearchState.

    Args:
        topic:      The research topic string.
        num_claims: Number of claims to extract (default 8).

    Returns:
        Fully populated ResearchState after all four agents have run.
    """
    initial = ResearchState(topic=topic, num_claims=num_claims)
    graph = build_graph()
    result_dict = graph.invoke(initial.model_dump())
    return ResearchState(**result_dict)
