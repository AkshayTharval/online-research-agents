"""Shared Pydantic data models used across all agents."""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class RawSource(BaseModel):
    """A scraped web source with its URL and extracted text."""

    url: str
    domain: str
    text: str


class Claim(BaseModel):
    """A factual claim extracted from raw sources."""

    claim: str
    source_url: str
    source_domain: str


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"


class VerifiedClaim(BaseModel):
    """A claim annotated with verification status and corroboration URL."""

    claim: str
    source_url: str
    source_domain: str
    status: VerificationStatus
    corroboration_url: str | None = None


class ResearchState(BaseModel):
    """Shared state passed between all agents in the LangGraph graph."""

    topic: str
    num_claims: int = Field(default=8, ge=1)
    raw_sources: list[RawSource] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    verified_claims: list[VerifiedClaim] = Field(default_factory=list)
    essay: str = ""
