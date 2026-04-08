"""Unit tests for the shared retry decorators."""

import groq
import httpx
import pytest
import requests
from tenacity import RetryError

from online_research_agents.retry import llm_retry, web_retry


# ---------------------------------------------------------------------------
# Helpers — minimal fake exceptions
# ---------------------------------------------------------------------------

def _raise(exc: Exception):
    """Return a callable that always raises exc."""
    def _inner(*args, **kwargs):
        raise exc
    return _inner


# ---------------------------------------------------------------------------
# llm_retry
# ---------------------------------------------------------------------------

class TestLlmRetry:
    def test_passes_through_on_success(self):
        @llm_retry
        def succeed():
            return "ok"

        assert succeed() == "ok"

    def test_retries_on_rate_limit_error(self, mocker):
        """Should retry up to 5 times on RateLimitError then reraise."""
        mock_sleep = mocker.patch("tenacity.nap.time.sleep")

        call_count = 0

        @llm_retry
        def always_rate_limit():
            nonlocal call_count
            call_count += 1
            raise groq.RateLimitError(
                message="rate limit",
                response=mocker.MagicMock(status_code=429),
                body={},
            )

        with pytest.raises(groq.RateLimitError):
            always_rate_limit()

        assert call_count == 5
        assert mock_sleep.call_count == 4  # sleep between attempts 1-2, 2-3, 3-4, 4-5

    def test_retries_on_api_status_error(self, mocker):
        """Should retry on APIStatusError (e.g. 503)."""
        mocker.patch("tenacity.nap.time.sleep")

        call_count = 0

        @llm_retry
        def always_api_error():
            nonlocal call_count
            call_count += 1
            raise groq.APIStatusError(
                message="service unavailable",
                response=mocker.MagicMock(status_code=503),
                body={},
            )

        with pytest.raises(groq.APIStatusError):
            always_api_error()

        assert call_count == 5

    def test_does_not_retry_on_unrelated_exception(self):
        """Non-Groq exceptions should bubble immediately without retrying."""
        call_count = 0

        @llm_retry
        def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not a groq error")

        with pytest.raises(ValueError):
            raise_value_error()

        assert call_count == 1  # no retries

    def test_succeeds_after_transient_failure(self, mocker):
        """Should succeed on 2nd attempt if 1st raises RateLimitError."""
        mocker.patch("tenacity.nap.time.sleep")

        call_count = 0

        @llm_retry
        def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise groq.RateLimitError(
                    message="rate limit",
                    response=mocker.MagicMock(status_code=429),
                    body={},
                )
            return "recovered"

        result = fail_once()
        assert result == "recovered"
        assert call_count == 2


# ---------------------------------------------------------------------------
# web_retry
# ---------------------------------------------------------------------------

class TestWebRetry:
    def test_passes_through_on_success(self):
        @web_retry
        def succeed():
            return [{"href": "https://example.com"}]

        assert succeed() == [{"href": "https://example.com"}]

    def test_retries_on_httpx_error(self, mocker):
        """Should retry up to 3 times on httpx.HTTPError."""
        mocker.patch("tenacity.nap.time.sleep")

        call_count = 0

        @web_retry
        def always_httpx_error():
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPError("connection error")

        with pytest.raises(httpx.HTTPError):
            always_httpx_error()

        assert call_count == 3

    def test_retries_on_httpx_timeout(self, mocker):
        """Should retry on httpx.TimeoutException."""
        mocker.patch("tenacity.nap.time.sleep")

        call_count = 0

        @web_retry
        def always_timeout():
            nonlocal call_count
            call_count += 1
            raise httpx.TimeoutException("timed out")

        with pytest.raises(httpx.TimeoutException):
            always_timeout()

        assert call_count == 3

    def test_retries_on_requests_exception(self, mocker):
        """Should retry on requests.exceptions.RequestException."""
        mocker.patch("tenacity.nap.time.sleep")

        call_count = 0

        @web_retry
        def always_requests_error():
            nonlocal call_count
            call_count += 1
            raise requests.exceptions.RequestException("network error")

        with pytest.raises(requests.exceptions.RequestException):
            always_requests_error()

        assert call_count == 3

    def test_does_not_retry_on_unrelated_exception(self):
        """Non-web exceptions should bubble immediately."""
        call_count = 0

        @web_retry
        def raise_runtime():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("not a web error")

        with pytest.raises(RuntimeError):
            raise_runtime()

        assert call_count == 1

    def test_succeeds_after_transient_failure(self, mocker):
        """Should succeed on 2nd attempt if 1st raises httpx.HTTPError."""
        mocker.patch("tenacity.nap.time.sleep")

        call_count = 0

        @web_retry
        def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPError("transient")
            return "ok"

        assert fail_once() == "ok"
        assert call_count == 2
