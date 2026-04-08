"""Shared retry decorators for LLM and web calls using tenacity."""

import logging

import groq
import httpx
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


def _log_retry(retry_state) -> None:
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    logger.warning(
        "Retry attempt %d | waiting %.1fs | reason: %s",
        retry_state.attempt_number,
        wait,
        retry_state.outcome.exception(),
    )


llm_retry = retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=16),
    retry=retry_if_exception_type((groq.RateLimitError, groq.APIStatusError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

web_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(
        (httpx.HTTPError, httpx.TimeoutException, requests.exceptions.RequestException)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
