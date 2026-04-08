"""Unit tests for the Extraction Agent."""

import pytest
from unittest.mock import MagicMock

from online_research_agents.agents import extraction_agent
from online_research_agents.agents.extraction_agent import (
    MAX_CHARS_PER_SOURCE,
    MAX_COMBINED_CHARS,
    _build_source_block,
    _ClaimList,
)
from online_research_agents.models import Claim, RawSource, ResearchState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_source(url: str, domain: str, text: str) -> RawSource:
    return RawSource(url=url, domain=domain, text=text)


def _make_claim(i: int) -> Claim:
    return Claim(
        claim=f"Factual claim number {i}.",
        source_url=f"https://site{i}.com/article",
        source_domain=f"site{i}.com",
    )


def _make_state(num_sources: int = 3, num_claims: int = 3) -> ResearchState:
    sources = [
        _make_source(
            url=f"https://site{i}.com/article",
            domain=f"site{i}.com",
            text=f"Article {i} content. " * 50,
        )
        for i in range(num_sources)
    ]
    return ResearchState(topic="climate change", num_claims=num_claims, raw_sources=sources)


def _make_mock_llm(claims: list[Claim]) -> MagicMock:
    """Return a mock LLM that returns a _ClaimList when structured output is invoked."""
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = _ClaimList(claims=claims)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = structured_llm
    return mock_llm


# ---------------------------------------------------------------------------
# _build_source_block
# ---------------------------------------------------------------------------

class TestBuildSourceBlock:
    def test_includes_source_url_header(self):
        state = _make_state(num_sources=2)
        block = _build_source_block(state)
        assert "SOURCE_URL: https://site0.com/article" in block
        assert "SOURCE_URL: https://site1.com/article" in block

    def test_truncates_long_source_text(self):
        long_text = "x" * (MAX_CHARS_PER_SOURCE + 500)
        state = ResearchState(
            topic="test",
            raw_sources=[_make_source("https://a.com", "a.com", long_text)],
        )
        block = _build_source_block(state)
        # The body should be capped at MAX_CHARS_PER_SOURCE
        assert len(block) <= MAX_CHARS_PER_SOURCE + 100  # header + separator overhead

    def test_total_block_capped_at_max_combined_chars(self):
        # Create many large sources
        sources = [
            _make_source(f"https://site{i}.com", f"site{i}.com", "y" * MAX_CHARS_PER_SOURCE)
            for i in range(50)
        ]
        state = ResearchState(topic="test", raw_sources=sources)
        block = _build_source_block(state)
        assert len(block) <= MAX_COMBINED_CHARS + 200  # small buffer for separators


# ---------------------------------------------------------------------------
# extraction_agent.run
# ---------------------------------------------------------------------------

class TestExtractionAgentRun:
    def test_returns_correct_number_of_claims(self, mocker):
        """Should return exactly num_claims claims from the LLM."""
        num_claims = 4
        state = _make_state(num_claims=num_claims)
        mock_claims = [_make_claim(i) for i in range(num_claims)]

        mocker.patch(
            "online_research_agents.agents.extraction_agent._build_llm",
            return_value=_make_mock_llm(mock_claims),
        )

        result = extraction_agent.run(state)

        assert len(result.claims) == num_claims

    def test_claims_are_claim_instances(self, mocker):
        """All returned items should be Claim Pydantic objects."""
        state = _make_state(num_claims=2)
        mock_claims = [_make_claim(i) for i in range(2)]

        mocker.patch(
            "online_research_agents.agents.extraction_agent._build_llm",
            return_value=_make_mock_llm(mock_claims),
        )

        result = extraction_agent.run(state)

        for claim in result.claims:
            assert isinstance(claim, Claim)

    def test_skips_extraction_when_no_sources(self, mocker):
        """Agent should return unchanged state when raw_sources is empty."""
        state = ResearchState(topic="test", num_claims=3)
        mock_llm = mocker.patch(
            "online_research_agents.agents.extraction_agent._build_llm"
        )

        result = extraction_agent.run(state)

        assert result.claims == []
        mock_llm.assert_not_called()

    def test_llm_called_with_topic_in_prompt(self, mocker):
        """The LLM prompt must mention the topic."""
        state = _make_state(num_claims=2)
        mock_claims = [_make_claim(i) for i in range(2)]
        captured = []

        structured_llm = MagicMock()
        structured_llm.invoke.side_effect = lambda msgs: (
            captured.extend(msgs) or _ClaimList(claims=mock_claims)
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = structured_llm

        mocker.patch(
            "online_research_agents.agents.extraction_agent._build_llm",
            return_value=mock_llm,
        )

        extraction_agent.run(state)

        human_msg = captured[-1].content
        assert "climate change" in human_msg

    def test_preserves_other_state_fields(self, mocker):
        """run() should not overwrite unrelated state fields."""
        state = _make_state(num_claims=2)
        mock_claims = [_make_claim(i) for i in range(2)]

        mocker.patch(
            "online_research_agents.agents.extraction_agent._build_llm",
            return_value=_make_mock_llm(mock_claims),
        )

        result = extraction_agent.run(state)

        assert result.topic == state.topic
        assert result.num_claims == state.num_claims
        assert result.raw_sources == state.raw_sources
