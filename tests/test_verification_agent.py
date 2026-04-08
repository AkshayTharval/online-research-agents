"""Unit tests for the Verification Agent."""

import pytest
from unittest.mock import MagicMock, patch

from online_research_agents.agents import verification_agent
from online_research_agents.agents.verification_agent import (
    _extract_domain,
    _is_topically_relevant,
    _verify_claim,
)
from online_research_agents.models import (
    Claim,
    ResearchState,
    VerificationStatus,
    VerifiedClaim,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_claim(
    claim: str = "Global temps rose 1.1°C since pre-industrial times.",
    source_url: str = "https://bbc.com/article/123",
    source_domain: str = "bbc.com",
) -> Claim:
    return Claim(claim=claim, source_url=source_url, source_domain=source_domain)


def _make_state(claims: list[Claim] | None = None) -> ResearchState:
    return ResearchState(
        topic="climate change",
        num_claims=max(len(claims), 1) if claims is not None else 1,
        claims=claims or [],
    )


def _ddg_result(url: str, title: str = "Global temps rose pre-industrial climate", body: str = "Global temperatures have risen since pre-industrial times due to climate change") -> dict:
    """Return a DDG result dict. Default title/body contains keywords matching _make_claim()."""
    return {"href": url, "title": title, "body": body}


# ---------------------------------------------------------------------------
# _is_topically_relevant
# ---------------------------------------------------------------------------

class TestIsTopicallyRelevant:
    def test_relevant_when_named_entity_matches(self):
        result = {"title": "MS Dhoni cricket debut India", "body": "Dhoni made his cricket debut in 2004"}
        assert _is_topically_relevant(result, "MS Dhoni made his international cricket debut in 2004") is True

    def test_irrelevant_mayoclinic_for_dhoni_claim(self):
        """mayoclinic (MS=Multiple Sclerosis) must not corroborate an MS Dhoni cricket claim."""
        result = {"title": "Multiple sclerosis symptoms causes", "body": "MS is a disease affecting the nervous system"}
        assert _is_topically_relevant(result, "MS Dhoni made his international cricket debut in 2004") is False

    def test_irrelevant_when_generic_words_only_match(self):
        """'team' and 'international' on a medical page must not pass for a cricket claim."""
        result = {
            "title": "International patient care team",
            "body": "Our international team provides medical care",
        }
        assert _is_topically_relevant(result, "MS Dhoni captained the Indian national team") is False

    def test_relevant_when_icc_appears_in_result(self):
        result = {"title": "ICC ODI Player of the Year 2008 Dhoni", "body": "Dhoni won ICC award"}
        assert _is_topically_relevant(result, "MS Dhoni won the ICC ODI Player of the Year award") is True

    def test_allows_when_no_result_text(self):
        """If result has no title/body we can't disqualify it — allow."""
        assert _is_topically_relevant({"title": "", "body": ""}, "some claim about cricket") is True

    def test_allows_when_claim_has_no_meaningful_words(self):
        """Very short/stopword-only claim — cannot check, allow."""
        assert _is_topically_relevant({"title": "anything", "body": "text"}, "and the") is True


# ---------------------------------------------------------------------------
# _extract_domain
# ---------------------------------------------------------------------------

class TestExtractDomain:
    def test_strips_www(self):
        assert _extract_domain("https://www.reuters.com/story") == "reuters.com"

    def test_no_www(self):
        assert _extract_domain("https://nature.com/article") == "nature.com"

    def test_empty_string(self):
        assert _extract_domain("") == ""


# ---------------------------------------------------------------------------
# _verify_claim (unit — mocks _search_corroboration)
# ---------------------------------------------------------------------------

class TestVerifyClaim:
    def test_verified_when_cross_domain_result_found(self, mocker):
        """Claim is VERIFIED when a result domain differs from source_domain."""
        claim = _make_claim(source_domain="bbc.com")
        mocker.patch(
            "online_research_agents.agents.verification_agent._search_corroboration",
            return_value=[_ddg_result("https://reuters.com/corroboration")],
        )

        result = _verify_claim(claim)

        assert result.status == VerificationStatus.VERIFIED
        assert result.corroboration_url == "https://reuters.com/corroboration"

    def test_unverified_when_only_same_domain_results(self, mocker):
        """Claim is UNVERIFIED when all results share the original source domain."""
        claim = _make_claim(source_domain="bbc.com")
        mocker.patch(
            "online_research_agents.agents.verification_agent._search_corroboration",
            return_value=[
                _ddg_result("https://bbc.com/other-article"),
                _ddg_result("https://www.bbc.com/another"),
            ],
        )

        result = _verify_claim(claim)

        assert result.status == VerificationStatus.UNVERIFIED
        assert result.corroboration_url is None

    def test_unverified_when_no_results(self, mocker):
        """Claim is UNVERIFIED when DuckDuckGo returns empty list."""
        claim = _make_claim()
        mocker.patch(
            "online_research_agents.agents.verification_agent._search_corroboration",
            return_value=[],
        )

        result = _verify_claim(claim)

        assert result.status == VerificationStatus.UNVERIFIED

    def test_unverified_when_search_raises(self, mocker):
        """Claim is marked UNVERIFIED (not raised) when search throws."""
        claim = _make_claim()
        mocker.patch(
            "online_research_agents.agents.verification_agent._search_corroboration",
            side_effect=Exception("DDG error"),
        )

        result = _verify_claim(claim)

        assert result.status == VerificationStatus.UNVERIFIED
        assert result.corroboration_url is None

    def test_skips_results_with_empty_href(self, mocker):
        """Results with empty href should be skipped; continues to next result."""
        claim = _make_claim(source_domain="bbc.com")
        mocker.patch(
            "online_research_agents.agents.verification_agent._search_corroboration",
            return_value=[
                {"href": "", "title": "Empty"},
                _ddg_result("https://nature.com/article"),
            ],
        )

        result = _verify_claim(claim)

        assert result.status == VerificationStatus.VERIFIED
        assert "nature.com" in result.corroboration_url

    def test_preserves_original_claim_fields(self, mocker):
        """VerifiedClaim should carry forward all fields from the original Claim."""
        claim = _make_claim()
        mocker.patch(
            "online_research_agents.agents.verification_agent._search_corroboration",
            return_value=[_ddg_result("https://reuters.com/x")],
        )

        result = _verify_claim(claim)

        assert result.claim == claim.claim
        assert result.source_url == claim.source_url
        assert result.source_domain == claim.source_domain


# ---------------------------------------------------------------------------
# verification_agent.run
# ---------------------------------------------------------------------------

class TestVerificationAgentRun:
    def test_returns_verified_claim_for_each_input_claim(self, mocker):
        """Output list length must equal input claims length."""
        claims = [_make_claim() for _ in range(3)]
        state = _make_state(claims=claims)

        mocker.patch(
            "online_research_agents.agents.verification_agent._search_corroboration",
            return_value=[_ddg_result("https://reuters.com/x")],
        )
        mocker.patch("online_research_agents.agents.verification_agent.time.sleep")

        result = verification_agent.run(state)

        assert len(result.verified_claims) == 3

    def test_skips_verification_when_no_claims(self, mocker):
        """Should return unchanged state when claims list is empty."""
        state = _make_state(claims=[])
        search_mock = mocker.patch(
            "online_research_agents.agents.verification_agent._search_corroboration"
        )

        result = verification_agent.run(state)

        assert result.verified_claims == []
        search_mock.assert_not_called()

    def test_all_verified_when_cross_domain_always_found(self, mocker):
        """All claims should be VERIFIED when cross-domain results always exist."""
        claims = [_make_claim(source_domain="bbc.com") for _ in range(3)]
        state = _make_state(claims=claims)

        mocker.patch(
            "online_research_agents.agents.verification_agent._search_corroboration",
            return_value=[_ddg_result("https://reuters.com/x")],
        )
        mocker.patch("online_research_agents.agents.verification_agent.time.sleep")

        result = verification_agent.run(state)

        assert all(v.status == VerificationStatus.VERIFIED for v in result.verified_claims)

    def test_all_unverified_when_only_same_domain(self, mocker):
        """All claims should be UNVERIFIED when only same-domain results are returned."""
        claims = [_make_claim(source_domain="bbc.com") for _ in range(2)]
        state = _make_state(claims=claims)

        mocker.patch(
            "online_research_agents.agents.verification_agent._search_corroboration",
            return_value=[_ddg_result("https://bbc.com/same")],
        )
        mocker.patch("online_research_agents.agents.verification_agent.time.sleep")

        result = verification_agent.run(state)

        assert all(v.status == VerificationStatus.UNVERIFIED for v in result.verified_claims)

    def test_sleep_called_between_claims(self, mocker):
        """time.sleep should be called between claims to respect rate limits."""
        claims = [_make_claim() for _ in range(3)]
        state = _make_state(claims=claims)

        mocker.patch(
            "online_research_agents.agents.verification_agent._search_corroboration",
            return_value=[],
        )
        sleep_mock = mocker.patch(
            "online_research_agents.agents.verification_agent.time.sleep"
        )

        verification_agent.run(state)

        # Sleep called between claims: N-1 times for N claims
        assert sleep_mock.call_count == len(claims) - 1
