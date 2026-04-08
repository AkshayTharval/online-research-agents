"""Extraction Agent — reads raw sources and extracts N structured factual claims."""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from online_research_agents.config import get_settings
from online_research_agents.models import Claim, ResearchState
from online_research_agents.retry import llm_retry

logger = logging.getLogger(__name__)

# Stay comfortably within Groq's context window
MAX_COMBINED_CHARS = 40_000
# Characters allocated per source before truncation
MAX_CHARS_PER_SOURCE = 3_000


class _ClaimList(BaseModel):
    """Wrapper so structured output returns a list of claims."""

    claims: list[Claim] = Field(description="List of extracted factual claims")


def _build_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model)


def _build_source_block(state: ResearchState) -> str:
    """
    Concatenate raw source texts, each prefixed with its URL.
    Truncates per-source text and caps the total to MAX_COMBINED_CHARS.
    """
    blocks: list[str] = []
    total = 0

    for source in state.raw_sources:
        header = f"SOURCE_URL: {source.url}\n"
        body = source.text[:MAX_CHARS_PER_SOURCE]
        block = header + body + "\n---\n"

        if total + len(block) > MAX_COMBINED_CHARS:
            remaining = MAX_COMBINED_CHARS - total
            if remaining > len(header) + 50:
                blocks.append(block[:remaining])
            break

        blocks.append(block)
        total += len(block)

    return "\n".join(blocks)


@llm_retry
def _extract_claims(source_block: str, topic: str, num_claims: int, llm: Any) -> list[Claim]:
    """Call the LLM with structured output to extract exactly num_claims claims."""
    structured_llm = llm.with_structured_output(_ClaimList)

    messages = [
        SystemMessage(
            content=(
                "You are a meticulous fact-extraction assistant. "
                "Given source texts (each prefixed with SOURCE_URL), extract exactly "
                f"{num_claims} distinct, specific, factual claims about the topic. "
                "For each claim:\n"
                "  - 'claim': a single concrete factual sentence\n"
                "  - 'source_url': MUST be copied EXACTLY from the SOURCE_URL: header "
                "    that appears immediately before the source text block — do NOT use "
                "    any URL that appears inside the body of the article text\n"
                "  - 'source_domain': the bare domain of that SOURCE_URL (no www., no path)\n"
                "IMPORTANT: Use a different source for each claim where possible — "
                "spread claims across all provided sources rather than pulling all "
                "claims from a single source.\n"
                "Return only factual statements — no opinions or generalities. "
                f"You MUST return exactly {num_claims} claims."
            )
        ),
        HumanMessage(
            content=f"Topic: {topic}\n\nSources:\n{source_block}"
        ),
    ]

    result: _ClaimList = structured_llm.invoke(messages)
    return result.claims


def run(state: ResearchState) -> ResearchState:
    """
    Extraction Agent entry point.

    Combines all raw source texts, sends to Groq with structured output,
    and populates state.claims with exactly num_claims Claim objects.
    """
    if not state.raw_sources:
        logger.warning("No raw sources available — skipping extraction")
        return state

    llm = _build_llm()
    source_block = _build_source_block(state)

    logger.info(
        "Extracting %d claims from %d sources (%d chars total)",
        state.num_claims,
        len(state.raw_sources),
        len(source_block),
    )

    claims = _extract_claims(source_block, state.topic, state.num_claims, llm)

    if len(claims) != state.num_claims:
        logger.warning(
            "Expected %d claims, got %d — proceeding with what was returned",
            state.num_claims,
            len(claims),
        )
    else:
        logger.info("Extracted %d claims successfully", len(claims))

    for i, claim in enumerate(claims, start=1):
        logger.info(
            "  [%d/%d] \"%s\" — %s",
            i,
            len(claims),
            claim.claim[:90],
            claim.source_domain,
        )

    return state.model_copy(update={"claims": claims})
