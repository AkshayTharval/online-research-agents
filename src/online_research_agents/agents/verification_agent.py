"""Verification Agent — cross-domain corroboration for each extracted claim."""

import logging
import time
from urllib.parse import urlparse

from duckduckgo_search import DDGS

from online_research_agents.models import (
    Claim,
    ResearchState,
    VerificationStatus,
    VerifiedClaim,
)
from online_research_agents.retry import web_retry

logger = logging.getLogger(__name__)

# Number of DuckDuckGo results to check per claim
MAX_RESULTS_PER_CLAIM = 5
# Polite delay between searches to avoid rate limiting (seconds)
SEARCH_DELAY = 1.5


def _extract_domain(url: str) -> str:
    """Return bare domain (without www.) from a URL."""
    netloc = urlparse(url).netloc
    return netloc.removeprefix("www.")


@web_retry
def _search_corroboration(query: str) -> list[dict]:
    """Search DuckDuckGo for corroborating sources."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=MAX_RESULTS_PER_CLAIM))
    return results


def _verify_claim(claim: Claim) -> VerifiedClaim:
    """
    Attempt to find a corroborating source for a single claim whose domain
    differs from the claim's original source domain.

    Returns a VerifiedClaim marked VERIFIED (with corroboration_url) or
    UNVERIFIED (corroboration_url=None).
    """
    # Use the claim text directly as the search query — it's already concise
    query = claim.claim

    try:
        results = _search_corroboration(query)
    except Exception as exc:
        logger.warning(
            "Corroboration search failed for claim '%s...': %s",
            claim.claim[:60],
            exc,
        )
        return VerifiedClaim(
            claim=claim.claim,
            source_url=claim.source_url,
            source_domain=claim.source_domain,
            status=VerificationStatus.UNVERIFIED,
            corroboration_url=None,
        )

    for result in results:
        url: str = result.get("href", "")
        if not url:
            continue

        result_domain = _extract_domain(url)

        if result_domain and result_domain != claim.source_domain:
            logger.info(
                "VERIFIED: '%s...' — corroborated by %s",
                claim.claim[:60],
                result_domain,
            )
            return VerifiedClaim(
                claim=claim.claim,
                source_url=claim.source_url,
                source_domain=claim.source_domain,
                status=VerificationStatus.VERIFIED,
                corroboration_url=url,
            )

    logger.info(
        "UNVERIFIED: '%s...' — no cross-domain source found",
        claim.claim[:60],
    )
    return VerifiedClaim(
        claim=claim.claim,
        source_url=claim.source_url,
        source_domain=claim.source_domain,
        status=VerificationStatus.UNVERIFIED,
        corroboration_url=None,
    )


def run(state: ResearchState) -> ResearchState:
    """
    Verification Agent entry point.

    For each extracted claim, searches DuckDuckGo for a corroborating source
    whose domain differs from the claim's original source domain. Marks each
    claim as VERIFIED or UNVERIFIED and populates state.verified_claims.
    """
    if not state.claims:
        logger.warning("No claims to verify — skipping verification")
        return state

    logger.info("Verifying %d claims...", len(state.claims))

    verified: list[VerifiedClaim] = []

    for i, claim in enumerate(state.claims, start=1):
        logger.info(
            "Verifying claim %d/%d: '%s...'",
            i,
            len(state.claims),
            claim.claim[:60],
        )
        result = _verify_claim(claim)
        verified.append(result)

        # Polite delay between searches — avoid hammering DuckDuckGo
        if i < len(state.claims):
            time.sleep(SEARCH_DELAY)

    verified_count = sum(1 for v in verified if v.status == VerificationStatus.VERIFIED)
    logger.info(
        "Verification complete: %d/%d claims verified",
        verified_count,
        len(verified),
    )

    return state.model_copy(update={"verified_claims": verified})
