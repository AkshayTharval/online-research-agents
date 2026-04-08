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

# Results to check per query pass — more results = higher chance of cross-domain hit
MAX_RESULTS_PER_CLAIM = 15
# Polite delay between claim searches to avoid rate limiting (seconds)
SEARCH_DELAY = 1.5
# How many words from the claim to use as the primary short query
SHORT_QUERY_WORDS = 10


def _extract_domain(url: str) -> str:
    """Return bare domain (without www.) from a URL."""
    netloc = urlparse(url).netloc
    return netloc.removeprefix("www.")


def _short_query(claim_text: str) -> str:
    """
    Build a concise keyword query from a claim.

    Taking the first N words avoids the overly-specific phrasing that verbatim
    claim text produces, which causes DuckDuckGo to return only sites that
    copied the exact sentence rather than independent sources.
    """
    words = claim_text.split()
    return " ".join(words[:SHORT_QUERY_WORDS])


@web_retry
def _search_corroboration(query: str, max_results: int = MAX_RESULTS_PER_CLAIM) -> list[dict]:
    """Search DuckDuckGo for corroborating sources."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results


_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "was", "that", "with", "from", "has", "have",
    "been", "this", "they", "their", "also", "its", "but", "not", "who",
    "which", "when", "will", "his", "her", "made", "over", "more", "some",
    "such", "into", "than", "then", "only", "said", "were", "about", "after",
})

# Capitalised words that look like proper nouns but are too generic to be
# discriminating — e.g. "Indian", "International", "World" appear on any site.
_GENERIC_CAPITALIZED: frozenset[str] = frozenset({
    "indian", "international", "national", "world", "united", "states",
    "first", "second", "third", "new", "great", "general", "former",
    "team", "cup", "year", "award", "player", "series",
})


def _is_topically_relevant(result: dict, claim_text: str) -> bool:
    """
    Return True if the result title/snippet is genuinely about the same topic
    as the claim.

    Two-gate check:
      Gate 1 (named-entity gate) — if the claim contains specific proper nouns
        (e.g. "Dhoni", "Chennai", "ICC"), at least ONE must appear in the
        result.  This blocks false positives like mayoclinic.org corroborating
        an MS Dhoni cricket claim just because "MS" = Multiple Sclerosis.
      Gate 2 (keyword gate) — at least 2 meaningful content words from the
        claim must appear in the result (same as before).

    Both gates must pass when named entities are present; only Gate 2 applies
    when the claim contains no named entities.
    """
    words = claim_text.split()
    clean = [w.strip(".,;:!?\"'()[]") for w in words]

    # Named entities: capitalised words (anywhere except position 0 which is
    # always capitalised), longer than 2 chars, not generically capitalised.
    named_entities = [
        w.lower() for i, w in enumerate(clean)
        if w and w[0].isupper() and len(w) > 2
        and w.lower() not in _GENERIC_CAPITALIZED
        and w.lower() not in _STOPWORDS
        and (i > 0 or len(words) == 1)  # skip sentence-start capital
    ]

    # Meaningful content words (for Gate 2)
    meaningful = [
        w.lower() for w in clean
        if len(w) > 3 and w.lower() not in _STOPWORDS
    ]

    result_text = (
        (result.get("title") or "") + " " + (result.get("body") or "")
    ).lower()

    if not result_text.strip():
        return True  # can't check — allow

    # Gate 1: named-entity presence
    if named_entities:
        if not any(entity in result_text for entity in named_entities):
            return False

    # Gate 2: keyword overlap
    if meaningful:
        matches = sum(1 for w in meaningful if w in result_text)
        if matches < min(2, len(meaningful)):
            return False

    return True


def _find_cross_domain_result(
    results: list[dict], source_domain: str, claim_text: str
) -> str | None:
    """
    Scan DuckDuckGo results for the first URL whose domain differs from
    source_domain AND whose title/snippet is topically relevant to the claim.
    Returns the URL string or None.
    """
    for result in results:
        url: str = result.get("href", "")
        if not url:
            continue
        result_domain = _extract_domain(url)
        if result_domain and result_domain != source_domain:
            if _is_topically_relevant(result, claim_text):
                return url
    return None


def _verify_claim(claim: Claim) -> VerifiedClaim:
    """
    Attempt to find a corroborating source for a single claim whose domain
    differs from the claim's original source domain.

    Strategy (two passes):
      Pass 1 — short keyword query (first N words): broad results, more
               diverse domains, higher hit rate.
      Pass 2 — full claim text as query: catches claims where the short
               query is too generic and returns unrelated results.

    Returns a VerifiedClaim marked VERIFIED (with corroboration_url) or
    UNVERIFIED (corroboration_url=None).
    """
    queries = [
        _short_query(claim.claim),   # pass 1: concise keyword search
        claim.claim,                  # pass 2: verbatim claim text
    ]

    for pass_num, query in enumerate(queries, start=1):
        try:
            results = _search_corroboration(query)
        except Exception as exc:
            logger.warning(
                "Corroboration search pass %d failed for '%s...': %s",
                pass_num,
                claim.claim[:60],
                exc,
            )
            continue

        corroboration_url = _find_cross_domain_result(results, claim.source_domain, claim.claim)

        if corroboration_url:
            corroboration_domain = _extract_domain(corroboration_url)
            logger.info(
                "VERIFIED (pass %d): '%s...' — corroborated by %s",
                pass_num,
                claim.claim[:60],
                corroboration_domain,
            )
            return VerifiedClaim(
                claim=claim.claim,
                source_url=claim.source_url,
                source_domain=claim.source_domain,
                status=VerificationStatus.VERIFIED,
                corroboration_url=corroboration_url,
            )

        logger.debug(
            "Pass %d found no cross-domain result for '%s...' — trying next query",
            pass_num,
            claim.claim[:60],
        )

    logger.info(
        "UNVERIFIED: '%s...' — no cross-domain source found after %d passes",
        claim.claim[:60],
        len(queries),
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

    For each extracted claim, runs up to two DuckDuckGo search passes
    (short keyword query, then full claim text) checking up to
    MAX_RESULTS_PER_CLAIM results each time for a cross-domain corroborator.
    Marks each claim VERIFIED or UNVERIFIED and populates state.verified_claims.
    """
    if not state.claims:
        logger.warning("No claims to verify — skipping verification")
        return state

    logger.info(
        "Verifying %d claims (%d results per pass, 2 passes max)...",
        len(state.claims),
        MAX_RESULTS_PER_CLAIM,
    )

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

        # Polite delay between claims — avoid hammering DuckDuckGo
        if i < len(state.claims):
            time.sleep(SEARCH_DELAY)

    verified_count = sum(1 for v in verified if v.status == VerificationStatus.VERIFIED)
    logger.info(
        "Verification complete: %d/%d claims verified",
        verified_count,
        len(verified),
    )

    return state.model_copy(update={"verified_claims": verified})
