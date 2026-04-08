"""Unit tests for the Search Agent."""

import pytest
from unittest.mock import MagicMock, patch

from online_research_agents.agents import search_agent
from online_research_agents.agents.search_agent import (
    MIN_SOURCES_PER_WORKER,
    _extract_domain,
)
from online_research_agents.models import ResearchState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_state() -> ResearchState:
    return ResearchState(topic="climate change", num_claims=8)


def _make_ddg_results(urls: list[str]) -> list[dict]:
    return [{"href": url, "title": "Title", "body": "Snippet"} for url in urls]


def _make_llm_response(queries: list[str]) -> MagicMock:
    mock = MagicMock()
    mock.content = "\n".join(queries)
    return mock


# ---------------------------------------------------------------------------
# _extract_domain
# ---------------------------------------------------------------------------

class TestExtractDomain:
    def test_strips_www(self):
        assert _extract_domain("https://www.bbc.com/article") == "bbc.com"

    def test_no_www(self):
        assert _extract_domain("https://reuters.com/story") == "reuters.com"

    def test_subdomain_preserved(self):
        assert _extract_domain("https://news.ycombinator.com/item") == "news.ycombinator.com"

    def test_empty_url(self):
        assert _extract_domain("") == ""


# ---------------------------------------------------------------------------
# search_agent.run
# ---------------------------------------------------------------------------

class TestSearchAgentRun:
    def test_collects_min_sources(self, base_state, mocker):
        """Agent should collect at least MIN_SOURCES_PER_WORKER sources."""
        queries = [f"query {i}" for i in range(5)]
        urls = [f"https://site{i}.com/article" for i in range(MIN_SOURCES_PER_WORKER + 2)]
        ddg_results = _make_ddg_results(urls)

        mocker.patch(
            "online_research_agents.agents.search_agent._build_llm",
            return_value=MagicMock(invoke=MagicMock(return_value=_make_llm_response(queries))),
        )
        mocker.patch(
            "online_research_agents.agents.search_agent._ddg_search",
            return_value=ddg_results,
        )
        mocker.patch(
            "online_research_agents.agents.search_agent._scrape_url",
            return_value="Article text with enough content.",
        )

        result = search_agent.run(base_state, query_angle="general")

        assert len(result.raw_sources) >= MIN_SOURCES_PER_WORKER

    def test_deduplicates_urls(self, base_state, mocker):
        """Same URL appearing in multiple query results should only be scraped once."""
        queries = ["query 1", "query 2"]
        # Both queries return the same URL
        duplicate_url = "https://bbc.com/article"
        ddg_results = _make_ddg_results([duplicate_url])

        mocker.patch(
            "online_research_agents.agents.search_agent._build_llm",
            return_value=MagicMock(invoke=MagicMock(return_value=_make_llm_response(queries))),
        )
        mocker.patch(
            "online_research_agents.agents.search_agent._ddg_search",
            return_value=ddg_results,
        )
        scrape_mock = mocker.patch(
            "online_research_agents.agents.search_agent._scrape_url",
            return_value="Some article text.",
        )

        search_agent.run(base_state, query_angle="general")

        # Should have been called exactly once despite appearing in multiple query results
        assert scrape_mock.call_count == 1

    def test_skips_empty_trafilatura_results(self, base_state, mocker):
        """URLs where trafilatura returns None should be skipped and not counted."""
        queries = ["query 1"]
        urls = [f"https://site{i}.com/article" for i in range(10)]

        mocker.patch(
            "online_research_agents.agents.search_agent._build_llm",
            return_value=MagicMock(invoke=MagicMock(return_value=_make_llm_response(queries))),
        )
        mocker.patch(
            "online_research_agents.agents.search_agent._ddg_search",
            return_value=_make_ddg_results(urls),
        )
        mocker.patch(
            "online_research_agents.agents.search_agent._scrape_url",
            return_value=None,  # trafilatura returns nothing
        )

        result = search_agent.run(base_state, query_angle="general")

        assert len(result.raw_sources) == 0

    def test_query_angle_affects_llm_prompt(self, base_state, mocker):
        """The query_angle should appear in the system message sent to the LLM."""
        captured_messages = []

        def capture_invoke(messages):
            captured_messages.extend(messages)
            return _make_llm_response(["query 1"])

        mock_llm = MagicMock(invoke=capture_invoke)
        mocker.patch(
            "online_research_agents.agents.search_agent._build_llm",
            return_value=mock_llm,
        )
        mocker.patch(
            "online_research_agents.agents.search_agent._ddg_search",
            return_value=[],
        )

        search_agent.run(base_state, query_angle="academic")

        system_msg = captured_messages[0].content
        assert "RESEARCH AND DATA" in system_msg  # academic angle instruction

    def test_handles_ddg_failure_gracefully(self, base_state, mocker):
        """If DuckDuckGo raises, the agent should log and continue, not crash."""
        queries = ["query 1", "query 2"]

        mocker.patch(
            "online_research_agents.agents.search_agent._build_llm",
            return_value=MagicMock(invoke=MagicMock(return_value=_make_llm_response(queries))),
        )
        mocker.patch(
            "online_research_agents.agents.search_agent._ddg_search",
            side_effect=Exception("DDG is down"),
        )

        # Should not raise — returns state with 0 sources and logs a warning
        result = search_agent.run(base_state, query_angle="general")
        assert isinstance(result, ResearchState)
        assert result.raw_sources == []

    def test_raw_sources_have_correct_domain(self, base_state, mocker):
        """Each RawSource should have a correctly parsed bare domain."""
        queries = ["query 1"]
        urls = ["https://www.nature.com/article/123"]

        mocker.patch(
            "online_research_agents.agents.search_agent._build_llm",
            return_value=MagicMock(invoke=MagicMock(return_value=_make_llm_response(queries))),
        )
        mocker.patch(
            "online_research_agents.agents.search_agent._ddg_search",
            return_value=_make_ddg_results(urls),
        )
        mocker.patch(
            "online_research_agents.agents.search_agent._scrape_url",
            return_value="Nature article text.",
        )

        result = search_agent.run(base_state, query_angle="academic")

        assert len(result.raw_sources) == 1
        assert result.raw_sources[0].domain == "nature.com"
        assert result.raw_sources[0].url == urls[0]
