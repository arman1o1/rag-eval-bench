"""RAG benchmark runner — systematic evaluation across config combinations.

Generates a config matrix (4 chunking strategies × 3 retrieval methods × 2 reranking)
and runs the full pipeline (ingest → retrieve → generate → evaluate) for each.
Supports result resumption and CLI invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Any

from src.config import (
    ChunkingConfig,
    ChunkingStrategy,
    RetrievalConfig,
    RetrievalMethod,
    Settings,
    get_settings,
)
from src.evaluation.dataset import EvalDataset
from src.evaluation.metrics import (
    MetricsReport,
    compute_citation_accuracy,
    compute_hallucination_rate,
    compute_latency,
    compute_ragas_metrics,
)

logger = logging.getLogger(__name__)

# Full config matrix dimensions
ALL_CHUNKING_STRATEGIES = list(ChunkingStrategy)
ALL_RETRIEVAL_METHODS = list(RetrievalMethod)
ALL_RERANKING = [True, False]


def generate_config_matrix() -> list[dict[str, Any]]:
    """Generate all 24 benchmark configurations.

    Returns:
        List of config dicts, each with 'chunking_strategy', 'retrieval_method',
        and 'use_reranker' keys.
    """
    configs: list[dict[str, Any]] = []
    for strategy, method, rerank in product(
        ALL_CHUNKING_STRATEGIES, ALL_RETRIEVAL_METHODS, ALL_RERANKING
    ):
        configs.append(
            {
                "chunking_strategy": strategy.value,
                "retrieval_method": method.value,
                "use_reranker": rerank,
            }
        )
    logger.info("Generated %d benchmark configurations", len(configs))
    return configs


def config_hash(config: dict[str, Any]) -> str:
    """Deterministic hash for a config dict, used for result file naming."""
    canonical = json.dumps(config, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()[:10]


class RAGBenchmark:
    """Run full RAG pipeline evaluation across multiple configurations.

    Args:
        settings: Base settings (API keys, paths). Individual configs are overridden
            per-run.
        eval_dataset: Ground-truth Q&A dataset to evaluate against.
    """

    def __init__(self, settings: Settings | None = None, eval_dataset: EvalDataset | None = None):
        self.settings = settings or get_settings()
        self.eval_dataset = eval_dataset

    def run(
        self,
        configs: list[dict[str, Any]] | str = "all",
        output_dir: Path | str = Path("results"),
    ) -> list[dict[str, Any]]:
        """Execute benchmark runs across configurations.

        Args:
            configs: List of config dicts, or 'all' to run the full 24-config matrix.
            output_dir: Directory to store per-run result JSON files.

        Returns:
            List of result dicts with 'config' and 'metrics' keys.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if configs == "all":
            configs = generate_config_matrix()

        if self.eval_dataset is None:
            gt_path = self.settings.ground_truth_dir
            if gt_path.exists():
                self.eval_dataset = EvalDataset.from_path(gt_path)
            else:
                logger.error("No ground-truth data at %s", gt_path)
                return []

        results: list[dict[str, Any]] = []
        total = len(configs)

        for idx, cfg in enumerate(configs, 1):
            cfg_id = config_hash(cfg)
            result_file = output_dir / f"benchmark_{cfg_id}.json"

            # Resumption: skip if result already exists
            if result_file.exists():
                logger.info(
                    "[%d/%d] Skipping (cached): %s", idx, total, cfg
                )
                cached = json.loads(result_file.read_text(encoding="utf-8"))
                results.append(cached)
                continue

            logger.info("[%d/%d] Running config: %s", idx, total, cfg)
            try:
                report = self._run_single(cfg)
                result = {"config": cfg, "metrics": report.to_dict()}
                result_file.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                results.append(result)
                logger.info("[%d/%d] Completed: %s", idx, total, cfg_id)
            except Exception:
                logger.exception("[%d/%d] Failed config %s", idx, total, cfg)
                error_result = {"config": cfg, "metrics": None, "error": True}
                results.append(error_result)

        # Save aggregated results
        summary_file = output_dir / "benchmark_summary.json"
        summary_file.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Benchmark complete. %d/%d configs succeeded.", len(results), total)
        return results

    def _run_single(self, config: dict[str, Any]) -> MetricsReport:
        """Run a single pipeline configuration and compute metrics.

        This method wires together the full RAG pipeline:
        1. Configure chunking/retrieval per the config dict
        2. Ingest documents (chunk + embed + index)
        3. For each eval question: retrieve → generate
        4. Evaluate with RAGAS + custom metrics
        """
        settings = self._apply_config(config)
        report = MetricsReport()

        answers: list[str] = []
        contexts: list[list[str]] = []
        retrieval_latencies: list[float] = []
        generation_latencies: list[float] = []

        # Lazy imports — these modules may not exist yet during development
        try:
            from src.generation.generator import RAGGenerator
            from src.ingestion.chunker import Chunker
            from src.ingestion.embedder import EmbeddingService
            from src.ingestion.indexer import QdrantIndexer
            from src.ingestion.loader import DocumentLoader
            from src.retrieval.retriever import RetrieverFactory
        except ImportError as exc:
            logger.error("Pipeline modules not available: %s", exc)
            raise

        # Step 1: Ingest
        loader = DocumentLoader()
        docs = loader.load(settings.corpus_dir)
        chunker = Chunker(settings.chunking)
        chunks = chunker.chunk(docs)

        embedding_service = EmbeddingService(settings.embedding)
        texts = [c.page_content for c in chunks]
        embeddings = embedding_service.embed_texts(texts)

        strategy_val = settings.chunking.strategy.value
        collection_name = f"{settings.qdrant.collection_name}_{strategy_val}"
        settings.qdrant.collection_name = collection_name
        bm25_path = Path("data") / f"bm25_{strategy_val}.pkl"

        indexer = QdrantIndexer(
            settings.qdrant,
            vector_dim=embedding_service.dimensions,
            bm25_path=bm25_path,
        )
        indexer.create_collection(recreate=True)
        indexer.index(chunks, embeddings)

        # Step 2: Retrieve + Generate per query
        retriever = RetrieverFactory.create(settings.retrieval, settings.qdrant, settings.embedding)
        generator = RAGGenerator(settings.generation)

        assert self.eval_dataset is not None
        for item in self.eval_dataset.items:
            # Retrieve
            retrieved, ret_latency = compute_latency(retriever.retrieve, item.question)
            retrieval_latencies.append(ret_latency)

            ctx_texts = [r.text for r in retrieved]
            contexts.append(ctx_texts)
            item.retrieved_contexts = ctx_texts

            # Generate
            answer, gen_latency = compute_latency(generator.generate, item.question, ctx_texts)
            generation_latencies.append(gen_latency)

            answers.append(answer)
            item.generated_answer = answer

            report.per_query.append(
                {
                    "question": item.question,
                    "answer": answer,
                    "retrieval_latency": ret_latency,
                    "generation_latency": gen_latency,
                }
            )

        # Step 3: Evaluate
        ragas_dataset = self.eval_dataset.to_ragas_format()
        ragas_scores = compute_ragas_metrics(ragas_dataset)
        report.faithfulness = ragas_scores["faithfulness"]
        report.answer_relevancy = ragas_scores["answer_relevancy"]
        report.context_precision = ragas_scores["context_precision"]
        report.context_recall = ragas_scores["context_recall"]

        report.hallucination_rate = compute_hallucination_rate(answers, contexts)
        report.citation_accuracy = compute_citation_accuracy(answers, contexts)

        if retrieval_latencies:
            report.avg_retrieval_latency = sum(retrieval_latencies) / len(retrieval_latencies)
        if generation_latencies:
            report.avg_generation_latency = sum(generation_latencies) / len(generation_latencies)
        report.avg_total_latency = report.avg_retrieval_latency + report.avg_generation_latency

        return report

    def _apply_config(self, config: dict[str, Any]) -> Settings:
        """Create a Settings instance with overrides from the config dict."""
        chunking_overrides = {}
        retrieval_overrides = {}

        if "chunking_strategy" in config:
            chunking_overrides["strategy"] = ChunkingStrategy(config["chunking_strategy"])
        if "chunk_size" in config:
            chunking_overrides["chunk_size"] = config["chunk_size"]
        if "chunk_overlap" in config:
            chunking_overrides["chunk_overlap"] = config["chunk_overlap"]

        if "retrieval_method" in config:
            retrieval_overrides["method"] = RetrievalMethod(config["retrieval_method"])
        if "use_reranker" in config:
            retrieval_overrides["use_reranker"] = config["use_reranker"]
        if "top_k" in config:
            retrieval_overrides["top_k"] = config["top_k"]

        return get_settings(
            openai_api_key=self.settings.openai_api_key,
            cohere_api_key=self.settings.cohere_api_key,
            corpus_dir=self.settings.corpus_dir,
            ground_truth_dir=self.settings.ground_truth_dir,
            results_dir=self.settings.results_dir,
            chunking=ChunkingConfig(**chunking_overrides),
            retrieval=RetrievalConfig(**retrieval_overrides),
        )


def main() -> None:
    """CLI entry point: python -m src.evaluation.benchmark"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Run RAG benchmark across config matrix")
    parser.add_argument(
        "--configs",
        default="all",
        help="'all' for full 24-config matrix, or path to a JSON file with config list",
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Output directory for benchmark results (default: results/)",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to ground-truth directory (overrides settings)",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.ground_truth:
        settings = get_settings(ground_truth_dir=Path(args.ground_truth))

    eval_dataset = None
    gt_path = settings.ground_truth_dir
    if gt_path.exists():
        eval_dataset = EvalDataset.from_path(gt_path)

    benchmark = RAGBenchmark(settings=settings, eval_dataset=eval_dataset)

    configs: list[dict[str, Any]] | str = "all"
    if args.configs != "all":
        config_path = Path(args.configs)
        if config_path.exists():
            configs = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            logger.error("Config file not found: %s", config_path)
            sys.exit(1)

    benchmark.run(configs=configs, output_dir=Path(args.output))


if __name__ == "__main__":
    main()
