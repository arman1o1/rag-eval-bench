"""Ablation study runner — systematic sweeps over individual RAG parameters.

Isolates the effect of each parameter (chunk size, overlap, top-k, reranker,
query transform) by holding everything else constant and sweeping one variable.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.evaluation.benchmark import RAGBenchmark
from src.evaluation.dataset import EvalDataset

logger = logging.getLogger(__name__)

# Ablation parameter grids
CHUNK_SIZES = [256, 512, 1024, 2048]
OVERLAP_PERCENTS = [0, 10, 20, 50]
TOP_K_VALUES = [3, 5, 10, 20]
RERANKER_OPTIONS = [True, False]
QUERY_TRANSFORMS = ["none", "rewrite", "hyde", "multi_query"]


class AblationRunner:
    """Run controlled ablation studies over RAG pipeline parameters.

    Each ablation study varies a single parameter while holding all others
    at their default values. Results are saved as JSON for visualization.

    Args:
        settings: Base pipeline settings.
        eval_dataset: Ground-truth evaluation dataset.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        eval_dataset: EvalDataset | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.eval_dataset = eval_dataset
        self._benchmark = RAGBenchmark(settings=self.settings, eval_dataset=self.eval_dataset)

    def run_all(self, output_dir: Path | str = Path("results/ablations")) -> dict[str, Any]:
        """Run all ablation studies and save results.

        Args:
            output_dir: Root directory for ablation result files.

        Returns:
            Dictionary mapping study_name → list of result dicts.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_results: dict[str, Any] = {}

        studies = [
            ("chunk_size", self.run_chunk_size_ablation),
            ("overlap_ratio", self.run_overlap_ablation),
            ("top_k", self.run_top_k_ablation),
            ("reranker", self.run_reranker_ablation),
            ("query_transform", self.run_query_transform_ablation),
        ]

        for name, runner in studies:
            logger.info("=== Running ablation study: %s ===", name)
            try:
                results = runner(output_dir=output_dir)
                all_results[name] = results
                logger.info("Completed ablation: %s (%d runs)", name, len(results))
            except Exception:
                logger.exception("Ablation study '%s' failed", name)
                all_results[name] = {"error": True}

        # Save combined summary
        summary_path = output_dir / "ablation_summary.json"
        summary_path.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("All ablation studies complete. Summary: %s", summary_path)
        return all_results

    def run_chunk_size_ablation(
        self, output_dir: Path | None = None
    ) -> list[dict[str, Any]]:
        """Sweep chunk_size over [256, 512, 1024, 2048]."""
        configs = [
            {
                "chunking_strategy": self.settings.chunking.strategy.value,
                "retrieval_method": self.settings.retrieval.method.value,
                "use_reranker": self.settings.retrieval.use_reranker,
                "chunk_size": size,
            }
            for size in CHUNK_SIZES
        ]
        return self._run_study("chunk_size", configs, output_dir)

    def run_overlap_ablation(
        self, output_dir: Path | None = None
    ) -> list[dict[str, Any]]:
        """Sweep overlap as percentage of chunk_size over [0, 10, 20, 50]%."""
        base_size = self.settings.chunking.chunk_size
        configs = [
            {
                "chunking_strategy": self.settings.chunking.strategy.value,
                "retrieval_method": self.settings.retrieval.method.value,
                "use_reranker": self.settings.retrieval.use_reranker,
                "chunk_size": base_size,
                "chunk_overlap": int(base_size * pct / 100),
            }
            for pct in OVERLAP_PERCENTS
        ]
        return self._run_study("overlap_ratio", configs, output_dir)

    def run_top_k_ablation(
        self, output_dir: Path | None = None
    ) -> list[dict[str, Any]]:
        """Sweep top_k over [3, 5, 10, 20]."""
        configs = [
            {
                "chunking_strategy": self.settings.chunking.strategy.value,
                "retrieval_method": self.settings.retrieval.method.value,
                "use_reranker": self.settings.retrieval.use_reranker,
                "top_k": k,
            }
            for k in TOP_K_VALUES
        ]
        return self._run_study("top_k", configs, output_dir)

    def run_reranker_ablation(
        self, output_dir: Path | None = None
    ) -> list[dict[str, Any]]:
        """Compare with vs. without reranker."""
        configs = [
            {
                "chunking_strategy": self.settings.chunking.strategy.value,
                "retrieval_method": self.settings.retrieval.method.value,
                "use_reranker": use_rr,
            }
            for use_rr in RERANKER_OPTIONS
        ]
        return self._run_study("reranker", configs, output_dir)

    def run_query_transform_ablation(
        self, output_dir: Path | None = None
    ) -> list[dict[str, Any]]:
        """Sweep query transformation strategy: none, rewrite, HyDE, multi-query."""
        configs = [
            {
                "chunking_strategy": self.settings.chunking.strategy.value,
                "retrieval_method": self.settings.retrieval.method.value,
                "use_reranker": self.settings.retrieval.use_reranker,
                "query_transform": qt,
            }
            for qt in QUERY_TRANSFORMS
        ]
        return self._run_study("query_transform", configs, output_dir)

    def _run_study(
        self,
        study_name: str,
        configs: list[dict[str, Any]],
        output_dir: Path | None,
    ) -> list[dict[str, Any]]:
        """Execute a batch of configs for a single ablation study."""
        study_dir = (output_dir or Path("results/ablations")) / study_name
        study_dir.mkdir(parents=True, exist_ok=True)

        results = self._benchmark.run(configs=configs, output_dir=study_dir)

        # Save study-level summary
        summary_path = study_dir / f"{study_name}_results.json"
        summary_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return results


def main() -> None:
    """CLI entry point: python -m src.evaluation.ablation"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Run RAG ablation studies")
    parser.add_argument(
        "--output",
        default="results/ablations/",
        help="Output directory for ablation results",
    )
    parser.add_argument(
        "--study",
        default="all",
        choices=["all", "chunk_size", "overlap_ratio", "top_k", "reranker", "query_transform"],
        help="Which ablation study to run (default: all)",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to ground-truth directory",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.ground_truth:
        settings = get_settings(ground_truth_dir=Path(args.ground_truth))

    eval_dataset = None
    if settings.ground_truth_dir.exists():
        eval_dataset = EvalDataset.from_path(settings.ground_truth_dir)

    runner = AblationRunner(settings=settings, eval_dataset=eval_dataset)

    output_dir = Path(args.output)
    if args.study == "all":
        runner.run_all(output_dir=output_dir)
    else:
        study_method = getattr(runner, f"run_{args.study}_ablation")
        study_method(output_dir=output_dir)


if __name__ == "__main__":
    main()
