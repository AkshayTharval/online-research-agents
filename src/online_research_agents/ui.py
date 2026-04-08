"""Streamlit web UI — verbose live output for the multi-agent research pipeline."""

import logging
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from online_research_agents.graph import run_pipeline  # noqa: E402 (after load_dotenv)
from online_research_agents.models import ResearchState, VerificationStatus  # noqa: E402

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Online Research Agents",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# StreamlitLogHandler
# Routes log records from each agent module to the correct UI container.
# ---------------------------------------------------------------------------

class StreamlitLogHandler(logging.Handler):
    """
    Intercepts log records emitted by agent modules and writes them into
    the matching per-agent Streamlit container in real time.

    Routing is based on the logger name (module path):
      online_research_agents.agents.search_agent      → "search"
      online_research_agents.agents.extraction_agent  → "extract"
      online_research_agents.agents.verification_agent→ "verify"
      online_research_agents.agents.essay_agent        → "write"
      online_research_agents.graph                     → "graph"
    """

    _ROUTES: dict[str, str] = {
        "search_agent":       "search",
        "extraction_agent":   "extract",
        "verification_agent": "verify",
        "essay_agent":        "write",
        "graph":              "graph",
    }

    def __init__(self, containers: dict[str, Any]) -> None:
        super().__init__(level=logging.INFO)
        self.setFormatter(logging.Formatter("%(message)s"))
        self._containers = containers

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            key = self._resolve(record.name)
            container = self._containers.get(key)
            if container is None:
                return
            if record.levelno >= logging.ERROR:
                container.error(msg)
            elif record.levelno >= logging.WARNING:
                container.warning(msg)
            else:
                container.caption(msg)
        except Exception:
            pass  # Never let a logging error crash the UI

    def _resolve(self, logger_name: str) -> str | None:
        for fragment, key in self._ROUTES.items():
            if fragment in logger_name:
                return key
        return None  # unknown logger — discard silently


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _verification_badge(status: VerificationStatus) -> str:
    return "✅ VERIFIED" if status == VerificationStatus.VERIFIED else "❌ UNVERIFIED"


def _render_sidebar() -> tuple[str, int]:
    """Render sidebar inputs and return (topic, num_claims)."""
    with st.sidebar:
        st.title("🔬 Research Settings")
        st.caption("Configure your research run below.")
        st.divider()

        topic = st.text_input(
            "Topic",
            placeholder="e.g. benefits of renewable energy",
            help="Enter any subject you want researched end-to-end.",
        )
        num_claims = st.slider(
            "Number of claims to extract",
            min_value=1,
            max_value=20,
            value=8,
            help="More claims = more comprehensive essay, but slower pipeline.",
        )
        st.divider()
        st.caption(
            "**How it works:**\n\n"
            "1. Three search workers run in parallel (news / academic / general)\n"
            "2. Claims are extracted from scraped sources\n"
            "3. Each claim is cross-verified on a different domain\n"
            "4. A cited essay is written from verified claims only"
        )

    return topic.strip(), num_claims


def _run_with_live_output(topic: str, num_claims: int) -> ResearchState:
    """
    Execute the pipeline while streaming log output into per-agent
    st.status() sections in real time via StreamlitLogHandler.

    st.status() is used instead of st.expander() because it supports
    real-time streaming writes during a blocking call; expander.container()
    does not reliably flush intermediate writes.
    """
    st.subheader("Pipeline Progress")

    # st.status() is designed for live streaming output — each section
    # shows a spinner while running and a tick when marked complete.
    search_status  = st.status("🔍 Search Agent — collecting sources",       expanded=True)
    extract_status = st.status("📋 Extraction Agent — extracting claims",     expanded=True)
    verify_status  = st.status("✅ Verification Agent — cross-checking claims", expanded=True)
    essay_status   = st.status("✍️ Essay Writer — composing essay",           expanded=True)

    # Scrollable inner containers so each section stays a fixed height
    search_log  = search_status.container(height=300)
    extract_log = extract_status.container(height=250)
    verify_log  = verify_status.container(height=300)
    essay_log   = essay_status.container(height=200)

    # graph-level banners (=== round N starting ===) appear in a plain container
    graph_log = st.container()

    containers = {
        "search":  search_log,
        "extract": extract_log,
        "verify":  verify_log,
        "write":   essay_log,
        "graph":   graph_log,
    }

    # Suppress noisy third-party library loggers before attaching our handler.
    # _resolve() returns None for any logger not in _ROUTES, so this is a belt-
    # and-suspenders defence: level suppression stops them at the source, and
    # the None guard in emit() drops anything that still slips through.
    _NOISY_LOGGERS = [
        "httpx", "httpcore", "trafilatura", "urllib3", "urllib",
        "requests", "groq", "_client", "charset_normalizer",
        "LangChain", "langchain", "langsmith", "openai",
    ]
    original_levels: dict[str, int] = {}
    for name in _NOISY_LOGGERS:
        lg = logging.getLogger(name)
        original_levels[name] = lg.level
        lg.setLevel(logging.WARNING)

    handler = StreamlitLogHandler(containers)
    root = logging.getLogger()
    original_root_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    try:
        result = run_pipeline(topic=topic, num_claims=num_claims)
    finally:
        root.removeHandler(handler)
        root.setLevel(original_root_level)
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)

    # Mark all status sections complete
    search_status.update(label="🔍 Search Agent — done",        state="complete", expanded=False)
    extract_status.update(label="📋 Extraction Agent — done",   state="complete", expanded=False)
    verify_status.update(label="✅ Verification Agent — done",  state="complete", expanded=False)
    essay_status.update(label="✍️ Essay Writer — done",         state="complete", expanded=False)

    return result


def _render_results(state: ResearchState) -> None:
    """Render the final essay and verification summary table."""
    st.divider()

    # ── Summary metrics ──────────────────────────────────────────────────
    verified_count = sum(
        1 for c in state.verified_claims if c.status == VerificationStatus.VERIFIED
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Sources collected", len(state.raw_sources))
    c2.metric("Claims extracted", len(state.claims))
    c3.metric("Claims verified", f"{verified_count}/{len(state.verified_claims)}")

    st.divider()

    # ── Essay ─────────────────────────────────────────────────────────────
    st.subheader("📄 Essay")
    st.markdown(state.essay)

    st.download_button(
        label="⬇️ Download Essay (.txt)",
        data=state.essay,
        file_name=f"essay_{state.topic[:40].replace(' ', '_')}.txt",
        mime="text/plain",
    )

    st.divider()

    # ── Verification summary ──────────────────────────────────────────────
    st.subheader("🔍 Verification Summary")

    if not state.verified_claims:
        st.info("No claims were processed.")
        return

    rows = [
        {
            "Status":        _verification_badge(vc.status),
            "Claim":         vc.claim,
            "Source":        vc.source_domain,
            "Corroboration": vc.corroboration_url or "—",
        }
        for vc in state.verified_claims
    ]

    st.dataframe(
        rows,
        use_container_width=True,
        column_config={
            "Status":        st.column_config.TextColumn("Status",        width="small"),
            "Claim":         st.column_config.TextColumn("Claim",         width="large"),
            "Source":        st.column_config.TextColumn("Source domain", width="medium"),
            "Corroboration": st.column_config.LinkColumn("Corroboration", width="medium"),
        },
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🔬 Online Research Agents")
    st.caption(
        "Powered by **LangGraph** · **Groq** (llama-3.3-70b) · **DuckDuckGo** · **trafilatura**"
    )

    topic, num_claims = _render_sidebar()

    # Run button — disabled if no topic entered
    run = st.button(
        "🚀 Run Research",
        type="primary",
        disabled=not topic,
        use_container_width=True,
    )

    # Persist result across reruns via session state
    if "result" not in st.session_state:
        st.session_state["result"] = None

    if run and topic:
        st.session_state["result"] = None  # clear previous run
        result = _run_with_live_output(topic, num_claims)
        st.session_state["result"] = result
        st.success("✅ Research complete!")

    if st.session_state["result"] is not None:
        _render_results(st.session_state["result"])


if __name__ == "__main__":
    main()
