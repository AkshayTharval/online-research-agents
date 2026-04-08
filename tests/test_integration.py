"""
End-to-end integration test — runs the full pipeline against real APIs.

Requires:
  - GROQ_API_KEY set in .env or environment
  - Internet access for DuckDuckGo searches and page scraping

Run with:
    pytest -m integration -v

Excluded from the default pytest run (no -m flag) because it makes real
network calls and can take 60–120 seconds to complete.
"""

import os

import pytest
from dotenv import load_dotenv

from online_research_agents.graph import run_pipeline
from online_research_agents.models import ResearchState, VerificationStatus

load_dotenv()

# ---------------------------------------------------------------------------
# Skip entire module if GROQ_API_KEY is not present
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration

_GROQ_KEY = os.getenv("GROQ_API_KEY", "")

if not _GROQ_KEY:
    pytest.skip(
        "GROQ_API_KEY not set — skipping integration tests",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Test topic — broad, well-documented, reliably returns many sources
# ---------------------------------------------------------------------------

TOPIC = "benefits of renewable energy"
NUM_CLAIMS = 5  # Kept small to reduce API usage and test duration


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFullPipeline:
    @pytest.fixture(scope="class")
    def pipeline_result(self) -> ResearchState:
        """
        Run the full pipeline once and share the result across all tests in
        this class. scope="class" ensures the expensive run happens only once.
        """
        return run_pipeline(topic=TOPIC, num_claims=NUM_CLAIMS)

    # --- Search Agent ---

    def test_raw_sources_collected(self, pipeline_result: ResearchState):
        """Pipeline must collect at least 15 raw sources (3 workers × ≥5 each)."""
        assert len(pipeline_result.raw_sources) >= 15, (
            f"Expected ≥15 sources, got {len(pipeline_result.raw_sources)}"
        )

    def test_raw_sources_have_required_fields(self, pipeline_result: ResearchState):
        """Every RawSource must have a non-empty url, domain, and text."""
        for source in pipeline_result.raw_sources:
            assert source.url, f"Empty URL in source: {source}"
            assert source.domain, f"Empty domain in source: {source}"
            assert source.text, f"Empty text in source: {source}"

    def test_raw_sources_are_deduplicated(self, pipeline_result: ResearchState):
        """No duplicate URLs should appear in the merged source list."""
        urls = [s.url for s in pipeline_result.raw_sources]
        assert len(urls) == len(set(urls)), "Duplicate URLs found in raw_sources"

    # --- Extraction Agent ---

    def test_claims_extracted(self, pipeline_result: ResearchState):
        """Pipeline must return exactly NUM_CLAIMS claims (or warn and return fewer)."""
        assert len(pipeline_result.claims) > 0, "No claims were extracted"

    def test_claims_have_required_fields(self, pipeline_result: ResearchState):
        """Every Claim must have non-empty claim text, source_url, and source_domain."""
        for claim in pipeline_result.claims:
            assert claim.claim, f"Empty claim text: {claim}"
            assert claim.source_url, f"Empty source_url in claim: {claim}"
            assert claim.source_domain, f"Empty source_domain in claim: {claim}"

    def test_claims_are_strings(self, pipeline_result: ResearchState):
        """Claim text must be a non-trivial string (not a placeholder)."""
        for claim in pipeline_result.claims:
            assert len(claim.claim) > 20, (
                f"Claim text suspiciously short: '{claim.claim}'"
            )

    # --- Verification Agent ---

    def test_verified_claims_count_matches_claims(self, pipeline_result: ResearchState):
        """verified_claims must be produced for every extracted claim."""
        assert len(pipeline_result.verified_claims) == len(pipeline_result.claims), (
            "Mismatch between claims and verified_claims count"
        )

    def test_at_least_one_claim_verified(self, pipeline_result: ResearchState):
        """At least one claim must achieve VERIFIED status in a real run."""
        verified = [
            c for c in pipeline_result.verified_claims
            if c.status == VerificationStatus.VERIFIED
        ]
        assert len(verified) >= 1, (
            "No claims were verified — check DuckDuckGo search or cross-domain logic"
        )

    def test_verified_claims_have_corroboration_url(self, pipeline_result: ResearchState):
        """Every VERIFIED claim must have a non-empty corroboration_url."""
        for claim in pipeline_result.verified_claims:
            if claim.status == VerificationStatus.VERIFIED:
                assert claim.corroboration_url, (
                    f"VERIFIED claim missing corroboration_url: {claim.claim}"
                )

    def test_unverified_claims_have_no_corroboration_url(self, pipeline_result: ResearchState):
        """UNVERIFIED claims must have corroboration_url=None."""
        for claim in pipeline_result.verified_claims:
            if claim.status == VerificationStatus.UNVERIFIED:
                assert claim.corroboration_url is None, (
                    f"UNVERIFIED claim unexpectedly has corroboration_url: {claim}"
                )

    def test_corroboration_domain_differs_from_source(self, pipeline_result: ResearchState):
        """For VERIFIED claims, corroboration domain must differ from source domain."""
        from urllib.parse import urlparse

        for claim in pipeline_result.verified_claims:
            if claim.status == VerificationStatus.VERIFIED and claim.corroboration_url:
                corroboration_domain = (
                    urlparse(claim.corroboration_url).netloc.removeprefix("www.")
                )
                assert corroboration_domain != claim.source_domain, (
                    f"Corroboration domain matches source domain: {claim.source_domain}"
                )

    # --- Essay Writer Agent ---

    def test_essay_is_non_empty(self, pipeline_result: ResearchState):
        """Final essay must be a non-empty string."""
        assert isinstance(pipeline_result.essay, str)
        assert len(pipeline_result.essay) > 100, (
            f"Essay suspiciously short ({len(pipeline_result.essay)} chars)"
        )

    def test_essay_contains_references_section(self, pipeline_result: ResearchState):
        """Essay must contain the References block appended by the agent."""
        assert "**References**" in pipeline_result.essay, (
            "References section missing from essay"
        )

    def test_essay_contains_inline_citations(self, pipeline_result: ResearchState):
        """Essay body must include at least one inline citation marker like [1]."""
        assert "[1]" in pipeline_result.essay, (
            "No inline citation [1] found in essay — LLM may have ignored instructions"
        )

    def test_essay_topic_relevant(self, pipeline_result: ResearchState):
        """Essay should mention the topic keyword (basic relevance check)."""
        keyword = "renewable"
        assert keyword.lower() in pipeline_result.essay.lower(), (
            f"Essay doesn't mention '{keyword}' — may be off-topic"
        )
