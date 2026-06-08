"""Evaluation dataset — load, save, and convert ground-truth Q&A pairs.

Handles the ground-truth JSON schema and provides RAGAS-compatible
dataset conversion for the evaluation framework.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class EvalItem:
    """Single evaluation item pairing a question with its ground truth."""

    question: str
    ground_truth_answer: str
    source_doc: str = ""
    source_page: int = 0
    source_section: str = ""
    context_texts: list[str] = field(default_factory=list)
    # Populated during evaluation pipeline runs
    generated_answer: str = ""
    retrieved_contexts: list[str] = field(default_factory=list)


class EvalDataset:
    """Manages ground-truth Q&A evaluation datasets.

    Loads from JSON files, converts to RAGAS format, and supports
    train/test splitting for evaluation runs.
    """

    def __init__(self, items: list[EvalItem] | None = None) -> None:
        self.items: list[EvalItem] = items or []

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> EvalItem:
        return self.items[idx]

    @staticmethod
    def load(path: Path) -> list[EvalItem]:
        """Load evaluation items from a JSON file or directory of JSON files.

        Args:
            path: Path to a single JSON file or directory containing JSON files.

        Returns:
            List of EvalItem objects parsed from the JSON data.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If JSON data is malformed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Ground-truth path not found: {path}")

        json_files: list[Path] = []
        if path.is_file():
            json_files = [path]
        elif path.is_dir():
            json_files = sorted(path.glob("*.json"))
            if not json_files:
                logger.warning("No JSON files found in %s", path)
                return []
        else:
            raise ValueError(f"Path is neither a file nor directory: {path}")

        items: list[EvalItem] = []
        for fp in json_files:
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
                entries = raw if isinstance(raw, list) else [raw]
                for entry in entries:
                    items.append(_parse_eval_item(entry))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.error("Failed to parse %s: %s", fp, exc)
                raise ValueError(f"Malformed ground-truth file {fp}: {exc}") from exc

        logger.info("Loaded %d evaluation items from %s", len(items), path)
        return items

    @classmethod
    def from_path(cls, path: Path) -> EvalDataset:
        """Convenience: create an EvalDataset by loading from path."""
        return cls(items=cls.load(path))

    @staticmethod
    def save(items: list[EvalItem], path: Path) -> None:
        """Persist evaluation items to a JSON file.

        Args:
            items: List of EvalItem objects to save.
            path: Output file path (parent dirs created automatically).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(item) for item in items]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved %d evaluation items to %s", len(items), path)

    def to_ragas_format(self) -> Dataset:
        """Convert to a HuggingFace Dataset compatible with RAGAS evaluation.

        RAGAS expects columns: question, answer, contexts, ground_truth.

        Returns:
            HuggingFace Dataset with RAGAS-compatible schema.
        """
        records: list[dict[str, Any]] = []
        for item in self.items:
            contexts = item.retrieved_contexts if item.retrieved_contexts else item.context_texts
            records.append(
                {
                    "question": item.question,
                    "answer": item.generated_answer or item.ground_truth_answer,
                    "contexts": contexts,
                    "ground_truth": item.ground_truth_answer,
                }
            )
        return Dataset.from_list(records)

    def split(
        self, test_size: float = 0.2, seed: int = 42
    ) -> tuple[EvalDataset, EvalDataset]:
        """Split into train/test datasets.

        Args:
            test_size: Fraction of items for the test split (0.0–1.0).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {test_size}")

        indices = list(range(len(self.items)))
        rng = random.Random(seed)
        rng.shuffle(indices)

        split_idx = int(len(indices) * (1 - test_size))
        train_items = [self.items[i] for i in indices[:split_idx]]
        test_items = [self.items[i] for i in indices[split_idx:]]

        logger.info("Split dataset: %d train, %d test", len(train_items), len(test_items))
        return EvalDataset(train_items), EvalDataset(test_items)


def _parse_eval_item(entry: dict[str, Any]) -> EvalItem:
    """Parse a single dictionary into an EvalItem, handling missing fields."""
    return EvalItem(
        question=entry["question"],
        ground_truth_answer=entry["ground_truth_answer"],
        source_doc=entry.get("source_doc", ""),
        source_page=entry.get("source_page", 0),
        source_section=entry.get("source_section", ""),
        context_texts=entry.get("context_texts", []),
        generated_answer=entry.get("generated_answer", ""),
        retrieved_contexts=entry.get("retrieved_contexts", []),
    )
