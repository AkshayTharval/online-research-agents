"""Unit tests for the Essay Writer Agent."""

import pytest
from unittest.mock import MagicMock

from online_research_agents.agents import essay_agent
from online_research_agents.agents.essay_agent import (
    INSUFFICIENT_INFO_MSG,
    _build_citation_list,
    _build_facts_block,
    _build_references_section,
)
from online_research_agents.models import ResearchState, VerificationStatus, VerifiedClaim


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_verified(
    claim: str = "Temps rose 1.1°C.",
    source_url: str = "https://bbc.com/a",
    source_domain: str = "bbc.com",
    status: VerificationStatus = VerificationStatus.VERIFIED,
    corroboration_url: str | None = "https://reuters.com/b",
) -> VerifiedClaim:
    return VerifiedClaim(
        claim=claim,
        source_url=source_url,
        source_domain=source_domain,
        status=status,
        corroboration_url=corroboration_url,
    )


def _make_state(
    verified_claims: list[VerifiedClaim] | None = None,
    topic: str = "climate change",
) -> ResearchState:
    return ResearchState(
        topic=topic,
        verified_claims=verified_claims or [],
    )


def _mock_llm(essay_text: str = "This is the essay body.") -> MagicMock:
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=essay_text)
    return mock


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------

class TestBuildCitationList:
    def test_keys_are_1_indexed(self):
        claims = [_make_verified(claim=f"Claim {i}") for i in range(3)]
        citations = _build_citation_list(claims)
        assert list(citations.keys()) == [1, 2, 3]

    def test_values_are_verified_claims(self):
        claims = [_make_verified()]
        citations = _build_citation_list(claims)
        assert citations[1] == claims[0]


class TestBuildFactsBlock:
    def test_contains_claim_text(self):
        claim = _make_verified(claim="Arctic ice shrunk by 13% per decade.")
        citations = {1: claim}
        block = _build_facts_block(citations)
        assert "Arctic ice shrunk by 13% per decade." in block

    def test_contains_citation_numbers(self):
        claims = [_make_verified(claim=f"Fact {i}.") for i in range(3)]
        citations = _build_citation_list(claims)
        block = _build_facts_block(citations)
        assert "[1]" in block
        assert "[2]" in block
        assert "[3]" in block

    def test_contains_source_domain(self):
        claim = _make_verified(source_domain="nature.com")
        citations = {1: claim}
        block = _build_facts_block(citations)
        assert "nature.com" in block


class TestBuildReferencesSection:
    def test_contains_source_url(self):
        claim = _make_verified(source_url="https://bbc.com/article")
        citations = {1: claim}
        refs = _build_references_section(citations)
        assert "https://bbc.com/article" in refs

    def test_contains_corroboration_url(self):
        claim = _make_verified(corroboration_url="https://reuters.com/corroboration")
        citations = {1: claim}
        refs = _build_references_section(citations)
        assert "https://reuters.com/corroboration" in refs

    def test_omits_corroboration_line_when_none(self):
        claim = _make_verified(corroboration_url=None)
        citations = {1: claim}
        refs = _build_references_section(citations)
        assert "Corroborated by" not in refs

    def test_contains_references_header(self):
        citations = {1: _make_verified()}
        refs = _build_references_section(citations)
        assert "**References**" in refs


# ---------------------------------------------------------------------------
# essay_agent.run
# ---------------------------------------------------------------------------

class TestEssayAgentRun:
    def test_essay_written_from_verified_claims_only(self, mocker):
        """Only VERIFIED claims should reach the LLM prompt."""
        verified = _make_verified(claim="Verified fact.", status=VerificationStatus.VERIFIED)
        unverified = _make_verified(
            claim="Unverified fact.",
            status=VerificationStatus.UNVERIFIED,
            corroboration_url=None,
        )
        state = _make_state(verified_claims=[verified, unverified])
        captured_msgs = []

        def capture(msgs):
            captured_msgs.extend(msgs)
            return MagicMock(content="Essay body.")

        mock_llm = MagicMock(invoke=capture)
        mocker.patch(
            "online_research_agents.agents.essay_agent._build_llm",
            return_value=mock_llm,
        )

        essay_agent.run(state)

        human_content = captured_msgs[-1].content
        assert "Verified fact." in human_content
        assert "Unverified fact." not in human_content

    def test_essay_is_non_empty(self, mocker):
        """State.essay should be a non-empty string after run()."""
        state = _make_state(verified_claims=[_make_verified()])
        mocker.patch(
            "online_research_agents.agents.essay_agent._build_llm",
            return_value=_mock_llm("A well-structured essay paragraph."),
        )

        result = essay_agent.run(state)

        assert isinstance(result.essay, str)
        assert len(result.essay) > 0

    def test_essay_contains_references_section(self, mocker):
        """Final essay must include the References block."""
        state = _make_state(verified_claims=[_make_verified()])
        mocker.patch(
            "online_research_agents.agents.essay_agent._build_llm",
            return_value=_mock_llm("Essay body text here."),
        )

        result = essay_agent.run(state)

        assert "**References**" in result.essay

    def test_insufficient_info_message_when_no_verified_claims(self, mocker):
        """Should return INSUFFICIENT_INFO_MSG when no VERIFIED claims exist."""
        unverified = _make_verified(
            status=VerificationStatus.UNVERIFIED,
            corroboration_url=None,
        )
        state = _make_state(verified_claims=[unverified])
        llm_mock = mocker.patch(
            "online_research_agents.agents.essay_agent._build_llm"
        )

        result = essay_agent.run(state)

        assert result.essay == INSUFFICIENT_INFO_MSG
        llm_mock.assert_not_called()

    def test_insufficient_info_when_verified_claims_empty(self, mocker):
        """Should return INSUFFICIENT_INFO_MSG when verified_claims list is empty."""
        state = _make_state(verified_claims=[])
        llm_mock = mocker.patch(
            "online_research_agents.agents.essay_agent._build_llm"
        )

        result = essay_agent.run(state)

        assert result.essay == INSUFFICIENT_INFO_MSG
        llm_mock.assert_not_called()

    def test_citation_numbers_present_in_facts_block(self, mocker):
        """The prompt sent to the LLM should include [1], [2] citation markers."""
        claims = [
            _make_verified(claim=f"Fact {i}.", status=VerificationStatus.VERIFIED)
            for i in range(3)
        ]
        state = _make_state(verified_claims=claims)
        captured_msgs = []

        def capture(msgs):
            captured_msgs.extend(msgs)
            return MagicMock(content="Essay.")

        mocker.patch(
            "online_research_agents.agents.essay_agent._build_llm",
            return_value=MagicMock(invoke=capture),
        )

        essay_agent.run(state)

        human_content = captured_msgs[-1].content
        assert "[1]" in human_content
        assert "[2]" in human_content
        assert "[3]" in human_content

    def test_preserves_other_state_fields(self, mocker):
        """run() should only update essay, leaving other state fields intact."""
        state = _make_state(verified_claims=[_make_verified()])
        state = state.model_copy(update={"topic": "renewable energy", "num_claims": 5})

        mocker.patch(
            "online_research_agents.agents.essay_agent._build_llm",
            return_value=_mock_llm("Essay text."),
        )

        result = essay_agent.run(state)

        assert result.topic == "renewable energy"
        assert result.num_claims == 5
