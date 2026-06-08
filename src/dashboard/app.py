"""RAG Eval Bench — Streamlit dashboard.

Two modes:
  - **Chat**: Interactive RAG Q&A with citation display and hallucination highlighting.
  - **Dashboard**: Benchmark result visualization with charts and drill-downs.

Run with: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.components import (
    chunk_viewer,
    citation_highlighter,
    comparison_table,
    metric_card,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Eval Bench",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 RAG Eval Bench")
    mode = st.radio("Mode", ["Chat", "Evaluation Dashboard"], index=0)
    st.divider()


# ---------------------------------------------------------------------------
# Chat mode
# ---------------------------------------------------------------------------
def render_chat_mode() -> None:
    """Render the interactive RAG chat interface."""
    st.header("💬 RAG Chat")

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                citation_highlighter(msg["content"], msg.get("citations"))
                if msg.get("chunks"):
                    with st.expander("📄 Source Chunks"):
                        chunk_viewer(msg["chunks"])
            else:
                st.markdown(msg["content"])

    # Input
    query = st.chat_input("Ask a question about your documents...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                response = _run_rag_query(query)
            citation_highlighter(response["answer"], response.get("citations"))
            if response.get("chunks"):
                with st.expander("📄 Source Chunks"):
                    chunk_viewer(response["chunks"])

            # Hallucination indicator
            if response.get("hallucination_score", 0) > 0.3:
                st.warning(
                    f"⚠️ Potential hallucination detected "
                    f"(score: {response['hallucination_score']:.2f})"
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["answer"],
                "citations": response.get("citations"),
                "chunks": response.get("chunks"),
            }
        )


def _run_rag_query(query: str) -> dict[str, Any]:
    """Execute a RAG query through the pipeline.

    Returns dict with 'answer', 'citations', 'chunks', 'hallucination_score'.
    Falls back to a placeholder if pipeline modules aren't available.
    """
    try:
        from src.config import get_settings
        from src.generation.generator import RAGGenerator
        from src.retrieval.retriever import RetrieverFactory

        settings = get_settings()
        retriever = RetrieverFactory.create(
            settings.retrieval, settings.qdrant, settings.embedding
        )
        generator = RAGGenerator(settings.generation)

        retrieved = retriever.retrieve(query)
        ctx_texts = [r.text for r in retrieved]
        citations = {i + 1: text[:100] for i, text in enumerate(ctx_texts)}

        answer = generator.generate(query, ctx_texts)

        hallucination_score = 0.0
        try:
            from src.generation.hallucination import HallucinationDetector

            detector = HallucinationDetector()
            result = detector.detect(answer=answer, context="\n\n".join(ctx_texts))
            hallucination_score = 1.0 - result.grounded_ratio
        except (ImportError, AttributeError):
            pass

        return {
            "answer": answer,
            "citations": citations,
            "chunks": ctx_texts,
            "hallucination_score": hallucination_score,
        }

    except ImportError:
        return {
            "answer": (
                "⚙️ Pipeline modules not yet available. "
                "Please build the ingestion, retrieval, and generation modules first."
            ),
            "citations": {},
            "chunks": [],
            "hallucination_score": 0.0,
        }


# ---------------------------------------------------------------------------
# Dashboard mode
# ---------------------------------------------------------------------------
def render_dashboard_mode() -> None:
    """Render the evaluation dashboard with charts and drill-downs."""
    st.header("📊 Evaluation Dashboard")

    # Sidebar: load results
    with st.sidebar:
        st.subheader("Load Results")
        results_path = st.text_input(
            "Results JSON path",
            value="results/benchmark_summary.json",
        )
        if st.button("Load Results"):
            _load_results(results_path)

        uploaded = st.file_uploader("Or upload results JSON", type=["json"])
        if uploaded is not None:
            try:
                data = json.loads(uploaded.read())
                st.session_state.benchmark_results = data if isinstance(data, list) else [data]
                st.success(f"Loaded {len(st.session_state.benchmark_results)} results")
            except json.JSONDecodeError:
                st.error("Invalid JSON file")

    results = st.session_state.benchmark_results
    if not results:
        st.info("Load benchmark results to get started. Use the sidebar to select a results file.")
        return

    # Filter out errored results
    valid = [r for r in results if r.get("metrics") is not None]
    if not valid:
        st.error("All loaded results contain errors.")
        return

    # Tab layout
    tabs = st.tabs([
        "Overview",
        "Chunking Comparison",
        "Strategy × Retrieval Heatmap",
        "Per-Query Drill-Down",
        "Ablation Studies",
        "Latency",
    ])

    with tabs[0]:
        _render_overview(valid)
    with tabs[1]:
        _render_chunking_comparison(valid)
    with tabs[2]:
        _render_heatmap(valid)
    with tabs[3]:
        _render_per_query(valid)
    with tabs[4]:
        _render_ablation_plots()
    with tabs[5]:
        _render_latency(valid)


def _load_results(path_str: str) -> None:
    """Load results from a JSON file path."""
    path = Path(path_str)
    if not path.exists():
        st.sidebar.error(f"File not found: {path}")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        st.session_state.benchmark_results = data if isinstance(data, list) else [data]
        st.sidebar.success(f"Loaded {len(st.session_state.benchmark_results)} results")
    except json.JSONDecodeError:
        st.sidebar.error("Invalid JSON")


def _render_overview(results: list[dict[str, Any]]) -> None:
    """Render overview metrics — best config and aggregate stats."""
    st.subheader("Best Configuration")

    # Find best by faithfulness
    best = max(results, key=lambda r: r["metrics"].get("faithfulness", 0))
    best_cfg = best["config"]
    best_metrics = best["metrics"]

    cols = st.columns(4)
    with cols[0]:
        metric_card("Faithfulness", best_metrics.get("faithfulness", 0))
    with cols[1]:
        metric_card("Answer Relevancy", best_metrics.get("answer_relevancy", 0))
    with cols[2]:
        metric_card("Context Precision", best_metrics.get("context_precision", 0))
    with cols[3]:
        metric_card("Hallucination Rate", best_metrics.get("hallucination_rate", 0))

    st.caption(
        f"Best config: **{best_cfg.get('chunking_strategy')}** chunking, "
        f"**{best_cfg.get('retrieval_method')}** retrieval, "
        f"reranker {'on' if best_cfg.get('use_reranker') else 'off'}"
    )

    st.divider()
    st.subheader("All Results")
    comparison_table(results)


def _render_chunking_comparison(results: list[dict[str, Any]]) -> None:
    """Bar charts comparing chunking strategies across RAGAS metrics."""
    st.subheader("Chunking Strategy Comparison")

    metrics_to_plot = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    rows = []
    for r in results:
        cfg = r["config"]
        m = r["metrics"]
        for metric_name in metrics_to_plot:
            rows.append(
                {
                    "Chunking Strategy": cfg.get("chunking_strategy", "unknown"),
                    "Metric": metric_name.replace("_", " ").title(),
                    "Score": m.get(metric_name, 0),
                }
            )

    if rows:
        fig = px.bar(
            rows,
            x="Chunking Strategy",
            y="Score",
            color="Metric",
            barmode="group",
            title="RAGAS Metrics by Chunking Strategy",
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)


def _render_heatmap(results: list[dict[str, Any]]) -> None:
    """Heatmap of chunking strategy × retrieval method → metric."""
    st.subheader("Strategy × Retrieval Heatmap")

    metric_choice = st.selectbox(
        "Metric",
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    strategies = sorted({r["config"].get("chunking_strategy", "") for r in results})
    methods = sorted({r["config"].get("retrieval_method", "") for r in results})

    # Build matrix — average across reranker settings
    matrix: list[list[float]] = []
    for strategy in strategies:
        row: list[float] = []
        for method in methods:
            matching = [
                r["metrics"].get(metric_choice, 0)
                for r in results
                if r["config"].get("chunking_strategy") == strategy
                and r["config"].get("retrieval_method") == method
            ]
            row.append(sum(matching) / len(matching) if matching else 0.0)
        matrix.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=methods,
            y=strategies,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=[[f"{v:.3f}" for v in row] for row in matrix],
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title=f"{metric_choice.replace('_', ' ').title()} by Strategy × Retrieval",
        xaxis_title="Retrieval Method",
        yaxis_title="Chunking Strategy",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_per_query(results: list[dict[str, Any]]) -> None:
    """Per-query drill-down table."""
    st.subheader("Per-Query Drill-Down")

    if not results:
        return

    config_labels = [
        f"{r['config'].get('chunking_strategy', '?')}_"
        f"{r['config'].get('retrieval_method', '?')}_"
        f"{'rr' if r['config'].get('use_reranker') else 'norr'}"
        for r in results
    ]
    selected = st.selectbox("Select Configuration", config_labels)
    idx = config_labels.index(selected)
    per_query = results[idx]["metrics"].get("per_query", [])

    if per_query:
        st.dataframe(per_query, use_container_width=True)
    else:
        st.info("No per-query data available for this configuration.")


def _render_ablation_plots() -> None:
    """Line plots for ablation study results."""
    st.subheader("Ablation Studies")

    ablation_path = st.text_input(
        "Ablation results path",
        value="results/ablations/ablation_summary.json",
        key="ablation_path",
    )

    path = Path(ablation_path)
    if not path.exists():
        st.info(f"No ablation results found at {path}")
        return

    try:
        ablation_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        st.error("Invalid ablation JSON")
        return

    for study_name, study_results in ablation_data.items():
        if isinstance(study_results, dict) and study_results.get("error"):
            st.warning(f"Study '{study_name}' had errors")
            continue

        if not isinstance(study_results, list) or not study_results:
            continue

        st.markdown(f"#### {study_name.replace('_', ' ').title()}")

        rows = []
        for r in study_results:
            if r.get("metrics") is None:
                continue
            cfg = r["config"]
            m = r["metrics"]

            # Determine the swept parameter value for x-axis
            param_val = cfg.get(study_name, str(cfg))
            for metric_name in ["faithfulness", "answer_relevancy", "context_precision"]:
                rows.append(
                    {
                        "Parameter": str(param_val),
                        "Metric": metric_name.replace("_", " ").title(),
                        "Score": m.get(metric_name, 0),
                    }
                )

        if rows:
            fig = px.line(
                rows,
                x="Parameter",
                y="Score",
                color="Metric",
                markers=True,
                title=f"{study_name.replace('_', ' ').title()} Ablation",
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)


def _render_latency(results: list[dict[str, Any]]) -> None:
    """Latency breakdown charts."""
    st.subheader("Latency Analysis")

    rows = []
    for r in results:
        cfg = r["config"]
        m = r["metrics"]
        label = (
            f"{cfg.get('chunking_strategy', '?')}_"
            f"{cfg.get('retrieval_method', '?')}_"
            f"{'rr' if cfg.get('use_reranker') else 'norr'}"
        )
        rows.append(
            {
                "Config": label,
                "Retrieval": m.get("avg_retrieval_latency", 0),
                "Generation": m.get("avg_generation_latency", 0),
            }
        )

    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        fig = px.bar(
            df,
            x="Config",
            y=["Retrieval", "Generation"],
            barmode="stack",
            title="Average Latency Breakdown by Configuration",
            labels={"value": "Seconds", "variable": "Stage"},
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
if mode == "Chat":
    render_chat_mode()
else:
    render_dashboard_mode()
