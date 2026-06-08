"""Tests for the evaluation framework.

Tests EvalDataset load/save roundtrip, MetricsReport structure,
and citation accuracy computation with synthetic data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.evaluation.dataset import EvalDataset, EvalItem
from src.evaluation.metrics import (
    MetricsReport,
    compute_citation_accuracy,
    compute_latency,
)

# ---------------------------------------------------------------------------
# EvalDataset load/save roundtrip
# ---------------------------------------------------------------------------

class TestEvalDatasetRoundtrip:
    """Test saving and loading evaluation items preserves data."""

    def test_save_and_load_roundtrip(self, sample_eval_items: list[EvalItem]):
        """Items saved to JSON and reloaded should be identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_qa.json"
            EvalDataset.save(sample_eval_items, path)

            assert path.exists()

            loaded = EvalDataset.load(path)
            assert len(loaded) == len(sample_eval_items)

            for original, restored in zip(sample_eval_items, loaded):
                assert original.question == restored.question
                assert original.ground_truth_answer == restored.ground_truth_answer
                assert original.source_doc == restored.source_doc
                assert original.source_page == restored.source_page
                assert original.context_texts == restored.context_texts

    def test_save_creates_parent_dirs(self, sample_eval_items: list[EvalItem]):
        """save() should create parent directories automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "qa.json"
            EvalDataset.save(sample_eval_items, path)
            assert path.exists()

    def test_load_from_directory(self, sample_eval_items: list[EvalItem]):
        """load() should handle a directory of JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save two separate files
            path1 = Path(tmpdir) / "qa_part1.json"
            path2 = Path(tmpdir) / "qa_part2.json"
            EvalDataset.save(sample_eval_items[:2], path1)
            EvalDataset.save(sample_eval_items[2:], path2)

            loaded = EvalDataset.load(Path(tmpdir))
            assert len(loaded) == len(sample_eval_items)

    def test_load_nonexistent_path_raises(self):
        """load() should raise FileNotFoundError for missing paths."""
        with pytest.raises(FileNotFoundError):
            EvalDataset.load(Path("/nonexistent/path/qa.json"))

    def test_load_malformed_json_raises(self):
        """load() should raise ValueError for invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text("not valid json {{{", encoding="utf-8")
            with pytest.raises(ValueError, match="Malformed"):
                EvalDataset.load(path)


# ---------------------------------------------------------------------------
# EvalDataset operations
# ---------------------------------------------------------------------------

class TestEvalDatasetOperations:
    """Test EvalDataset methods beyond load/save."""

    def test_split_preserves_total_count(self, sample_eval_items: list[EvalItem]):
        """Train + test split should contain all items."""
        ds = EvalDataset(sample_eval_items)
        train, test = ds.split(test_size=0.4)
        assert len(train) + len(test) == len(sample_eval_items)

    def test_split_is_deterministic(self, sample_eval_items: list[EvalItem]):
        """Same seed should produce identical splits."""
        ds = EvalDataset(sample_eval_items)
        train1, test1 = ds.split(test_size=0.2, seed=123)
        train2, test2 = ds.split(test_size=0.2, seed=123)

        assert [i.question for i in train1.items] == [i.question for i in train2.items]
        assert [i.question for i in test1.items] == [i.question for i in test2.items]

    def test_split_invalid_test_size(self, sample_eval_items: list[EvalItem]):
        """test_size outside (0, 1) should raise ValueError."""
        ds = EvalDataset(sample_eval_items)
        with pytest.raises(ValueError):
            ds.split(test_size=0.0)
        with pytest.raises(ValueError):
            ds.split(test_size=1.0)

    def test_to_ragas_format(self, sample_eval_items: list[EvalItem]):
        """to_ragas_format should produce a Dataset with expected columns."""
        ds = EvalDataset(sample_eval_items)
        ragas_ds = ds.to_ragas_format()

        assert "question" in ragas_ds.column_names
        assert "answer" in ragas_ds.column_names
        assert "contexts" in ragas_ds.column_names
        assert "ground_truth" in ragas_ds.column_names
        assert len(ragas_ds) == len(sample_eval_items)

    def test_len_and_getitem(self, sample_eval_items: list[EvalItem]):
        """__len__ and __getitem__ should work."""
        ds = EvalDataset(sample_eval_items)
        assert len(ds) == len(sample_eval_items)
        assert ds[0].question == sample_eval_items[0].question


# ---------------------------------------------------------------------------
# MetricsReport
# ---------------------------------------------------------------------------

class TestMetricsReport:
    """Test MetricsReport structure and serialization."""

    def test_default_values(self):
        """Default MetricsReport should have zero values."""
        report = MetricsReport()
        assert report.faithfulness == 0.0
        assert report.hallucination_rate == 0.0
        assert report.per_query == []

    def test_to_dict_keys(self):
        """to_dict should contain all expected metric keys."""
        report = MetricsReport(
            faithfulness=0.85,
            answer_relevancy=0.90,
            context_precision=0.78,
            context_recall=0.82,
            hallucination_rate=0.05,
            citation_accuracy=0.95,
            avg_retrieval_latency=0.12,
            avg_generation_latency=1.5,
        )
        d = report.to_dict()

        expected_keys = {
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "hallucination_rate",
            "citation_accuracy",
            "avg_retrieval_latency",
            "avg_generation_latency",
            "avg_total_latency",
            "num_queries",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match(self):
        """to_dict values should match the report's attributes."""
        report = MetricsReport(faithfulness=0.85, hallucination_rate=0.15)
        d = report.to_dict()
        assert d["faithfulness"] == 0.85
        assert d["hallucination_rate"] == 0.15


# ---------------------------------------------------------------------------
# Citation accuracy
# ---------------------------------------------------------------------------

class TestCitationAccuracy:
    """Test citation accuracy computation with synthetic data."""

    def test_all_valid_citations(self):
        """All citations mapping to valid chunks → accuracy 1.0."""
        answers = ["The answer is [1] and also [2]."]
        contexts = [["Context chunk 1", "Context chunk 2", "Context chunk 3"]]
        accuracy = compute_citation_accuracy(answers, contexts)
        assert accuracy == 1.0

    def test_all_invalid_citations(self):
        """Citations referencing non-existent chunks → accuracy 0.0."""
        answers = ["The answer is [5] and [10]."]
        contexts = [["Only one chunk"]]
        accuracy = compute_citation_accuracy(answers, contexts)
        assert accuracy == 0.0

    def test_mixed_citations(self):
        """Mix of valid and invalid citations."""
        answers = ["See [1] and [3] and [5]."]
        contexts = [["Chunk 1", "Chunk 2", "Chunk 3"]]
        # [1] valid, [3] valid, [5] invalid → 2/3
        accuracy = compute_citation_accuracy(answers, contexts)
        assert abs(accuracy - 2 / 3) < 1e-6

    def test_no_citations(self):
        """Answer with no citation markers → accuracy 1.0 (nothing wrong)."""
        answers = ["This answer has no citations at all."]
        contexts = [["Some chunk"]]
        accuracy = compute_citation_accuracy(answers, contexts)
        assert accuracy == 1.0

    def test_empty_answers(self):
        """Empty answer list → accuracy 0.0."""
        accuracy = compute_citation_accuracy([], [])
        assert accuracy == 0.0

    def test_multiple_answers(self):
        """Citation accuracy across multiple answer-context pairs."""
        answers = [
            "Answer one [1] here.",   # valid
            "Answer two [2] here.",   # invalid (only 1 chunk)
        ]
        contexts = [
            ["Chunk A"],
            ["Chunk B"],
        ]
        # Answer 1: [1] valid → 1/1
        # Answer 2: [2] invalid → 0/1
        # Total: 1/2 = 0.5
        accuracy = compute_citation_accuracy(answers, contexts)
        assert accuracy == 0.5


# ---------------------------------------------------------------------------
# Latency wrapper
# ---------------------------------------------------------------------------

class TestComputeLatency:
    """Test the latency measurement wrapper."""

    def test_returns_result_and_time(self):
        """compute_latency should return (result, elapsed_seconds)."""
        def dummy(x: int) -> int:
            return x * 2

        result, elapsed = compute_latency(dummy, 5)
        assert result == 10
        assert elapsed >= 0.0
        assert isinstance(elapsed, float)

    def test_latency_is_nonnegative(self):
        """Elapsed time should always be non-negative."""
        result, elapsed = compute_latency(lambda: "hello")
        assert result == "hello"
        assert elapsed >= 0.0
