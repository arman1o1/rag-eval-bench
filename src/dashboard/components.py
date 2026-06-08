"""Reusable Streamlit UI components for the RAG Eval Bench dashboard.

Provides styled metric cards, chunk viewers, citation highlighters,
and comparison tables used by the main app.
"""

from __future__ import annotations

import re
from typing import Any

import streamlit as st


def metric_card(
    label: str,
    value: float | str,
    delta: float | str | None = None,
    help_text: str | None = None,
) -> None:
    """Render a styled metric card using Streamlit's metric widget.

    Args:
        label: Metric name displayed as the card title.
        value: Primary value (formatted as percentage if float).
        delta: Optional delta value for comparison.
        help_text: Optional tooltip text.
    """
    formatted_value = f"{value:.2%}" if isinstance(value, float) else str(value)
    formatted_delta = None
    if delta is not None:
        formatted_delta = f"{delta:.2%}" if isinstance(delta, float) else str(delta)

    st.metric(
        label=label,
        value=formatted_value,
        delta=formatted_delta,
        help=help_text,
    )


def chunk_viewer(
    chunks: list[dict[str, Any] | str],
    expanded: bool = False,
    max_display: int = 10,
) -> None:
    """Render expandable source chunks in the sidebar or main area.

    Args:
        chunks: List of chunk dicts (with 'text' and optional 'metadata' keys)
            or plain strings.
        expanded: Whether expanders start open.
        max_display: Maximum number of chunks to show.
    """
    if not chunks:
        st.info("No source chunks to display.")
        return

    st.markdown(f"**Source Chunks** ({len(chunks)} total)")
    for i, chunk in enumerate(chunks[:max_display]):
        if isinstance(chunk, str):
            text = chunk
            meta = {}
        else:
            text = chunk.get("text", str(chunk))
            meta = chunk.get("metadata", {})

        with st.expander(f"Chunk {i + 1}", expanded=expanded):
            if meta:
                cols = st.columns(3)
                if "source" in meta:
                    cols[0].caption(f"📄 {meta['source']}")
                if "page" in meta:
                    cols[1].caption(f"Page {meta['page']}")
                if "section" in meta:
                    cols[2].caption(f"§ {meta['section']}")
            st.text(text)

    if len(chunks) > max_display:
        st.caption(f"...and {len(chunks) - max_display} more chunks")


def citation_highlighter(text: str, citations: dict[int, str] | None = None) -> None:
    """Render text with highlighted [N] citation markers.

    Citations are colorized inline. If a citations dict is provided,
    hovering shows the source chunk text.

    Args:
        text: Answer text containing [N] markers.
        citations: Optional mapping of citation index → source text snippet.
    """
    citations = citations or {}
    citation_pattern = re.compile(r"\[(\d+)\]")

    # Build HTML with colored citation badges
    def _replace_citation(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        source = citations.get(idx, "")
        tooltip = f' title="{source}"' if source else ""
        return (
            f'<span style="background-color: #1f77b4; color: white; '
            f'padding: 1px 6px; border-radius: 10px; font-size: 0.85em; '
            f'cursor: help;"{tooltip}>[{idx}]</span>'
        )

    highlighted = citation_pattern.sub(_replace_citation, text)
    st.markdown(highlighted, unsafe_allow_html=True)


def comparison_table(results: list[dict[str, Any]]) -> None:
    """Render a formatted comparison table of benchmark results.

    Args:
        results: List of result dicts, each with 'config' and 'metrics' keys.
    """
    if not results:
        st.info("No benchmark results to display.")
        return

    rows = []
    for r in results:
        cfg = r.get("config", {})
        metrics = r.get("metrics", {})
        if metrics is None:
            continue
        rows.append(
            {
                "Chunking": cfg.get("chunking_strategy", "—"),
                "Retrieval": cfg.get("retrieval_method", "—"),
                "Reranker": "✓" if cfg.get("use_reranker") else "✗",
                "Faithfulness": f"{metrics.get('faithfulness', 0):.3f}",
                "Answer Relevancy": f"{metrics.get('answer_relevancy', 0):.3f}",
                "Context Precision": f"{metrics.get('context_precision', 0):.3f}",
                "Context Recall": f"{metrics.get('context_recall', 0):.3f}",
                "Hallucination": f"{metrics.get('hallucination_rate', 0):.3f}",
                "Citation Acc.": f"{metrics.get('citation_accuracy', 0):.3f}",
                "Latency (s)": f"{metrics.get('avg_total_latency', 0):.2f}",
            }
        )

    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.warning("All results had errors — no data to display.")
