"""Essay Writer Agent — composes a cited essay from verified claims."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from online_research_agents.config import get_settings
from online_research_agents.models import ResearchState, VerificationStatus, VerifiedClaim
from online_research_agents.retry import llm_retry

logger = logging.getLogger(__name__)

INSUFFICIENT_INFO_MSG = (
    "Insufficient verified information was found to write a reliable essay on this topic. "
    "All extracted claims failed cross-domain verification. "
    "Please try again with a different topic or increase the number of claims."
)


def _build_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model)


def _build_citation_list(verified_claims: list[VerifiedClaim]) -> dict[int, VerifiedClaim]:
    """Map citation number → VerifiedClaim for inline reference building."""
    return {i: claim for i, claim in enumerate(verified_claims, start=1)}


def _build_facts_block(citations: dict[int, VerifiedClaim]) -> str:
    """Format verified claims as a numbered fact list for the prompt."""
    lines = []
    for n, claim in citations.items():
        lines.append(f"[{n}] {claim.claim} (source: {claim.source_domain})")
    return "\n".join(lines)


def _build_references_section(citations: dict[int, VerifiedClaim]) -> str:
    """Build the References block appended to the end of the essay."""
    lines = ["", "---", "**References**", ""]
    for n, claim in citations.items():
        lines.append(f"[{n}] {claim.source_url}")
        if claim.corroboration_url:
            lines.append(f"    Corroborated by: {claim.corroboration_url}")
    return "\n".join(lines)


@llm_retry
def _write_essay(topic: str, facts_block: str, num_claims: int, llm: ChatGroq) -> str:
    """Call the LLM to write the essay."""
    messages = [
        SystemMessage(
            content=(
                "You are an expert research writer. Write a well-structured, "
                "3–4 paragraph essay on the given topic using ONLY the verified "
                "facts provided. Rules:\n"
                "  - Use inline citations like [1], [2] immediately after each fact\n"
                "  - Do NOT introduce any facts not listed below\n"
                "  - Do NOT use phrases like 'According to [1]' — embed citations naturally\n"
                "  - Write in clear, formal prose suitable for a general audience\n"
                "  - Every paragraph must use at least one citation\n"
                "  - End with a concise concluding paragraph that synthesises the findings"
            )
        ),
        HumanMessage(
            content=(
                f"Topic: {topic}\n\n"
                f"Verified facts ({num_claims} total):\n{facts_block}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def run(state: ResearchState) -> ResearchState:
    """
    Essay Writer Agent entry point.

    Filters to only VERIFIED claims, builds a numbered citation map,
    prompts the Groq LLM for a 3–4 paragraph essay with inline citations,
    and appends a full References section. Populates state.essay.
    """
    verified_only = [
        c for c in state.verified_claims
        if c.status == VerificationStatus.VERIFIED
    ]

    if not verified_only:
        logger.warning(
            "No verified claims available — writing insufficient-info message"
        )
        return state.model_copy(update={"essay": INSUFFICIENT_INFO_MSG})

    logger.info(
        "Writing essay on '%s' using %d verified claim(s)",
        state.topic,
        len(verified_only),
    )

    citations = _build_citation_list(verified_only)
    facts_block = _build_facts_block(citations)

    essay_body = _write_essay(state.topic, facts_block, len(verified_only), _build_llm())

    references = _build_references_section(citations)
    full_essay = essay_body + "\n" + references

    logger.info("Essay written successfully (%d characters)", len(full_essay))

    return state.model_copy(update={"essay": full_essay})
