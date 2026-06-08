"""Download the CUAD (Contract Understanding Atticus Dataset) for evaluation.

Downloads from HuggingFace, extracts contract texts, and saves them as
individual files in the corpus directory.

Usage:
    python scripts/download_cuad.py --output-dir data/corpus
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_cuad(output_dir: Path, max_contracts: int | None = None) -> None:
    """Download CUAD dataset and extract contract documents.

    Args:
        output_dir: Directory to save extracted contract files.
        max_contracts: Optional limit on number of contracts to download.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("'datasets' library required: pip install datasets")
        raise

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading CUAD dataset from HuggingFace...")
    try:
        dataset = load_dataset("theatticusproject/cuad-qa", split="test")
    except Exception:
        logger.warning("Primary dataset ID failed, trying fallback 'cuad'...")
        try:
            dataset = load_dataset("cuad", split="test")
        except Exception:
            logger.error(
                "Failed to download CUAD. You may need to accept terms at "
                "https://huggingface.co/datasets/theatticusproject/cuad-qa"
            )
            raise

    # Extract unique contract documents
    # CUAD has 'context' field with the contract text and 'title' for the contract name
    seen_titles: set[str] = set()
    saved_count = 0

    for item in dataset:
        title = item.get("title", f"contract_{saved_count}")
        if title in seen_titles:
            continue
        seen_titles.add(title)

        context = item.get("context", "")
        if not context.strip():
            continue

        # Sanitize filename
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
        safe_name = safe_name.strip()[:100]
        filepath = output_dir / f"{safe_name}.txt"

        filepath.write_text(context, encoding="utf-8")
        saved_count += 1

        if max_contracts and saved_count >= max_contracts:
            break

    logger.info(
        "Downloaded %d contracts to %s",
        saved_count,
        output_dir,
    )
    print(f"\n✅ Downloaded {saved_count} contract documents to {output_dir}")
    print(f"   Unique contracts found: {len(seen_titles)}")

    # Print summary
    total_bytes = sum(f.stat().st_size for f in output_dir.glob("*.txt"))
    print(f"   Total size: {total_bytes / 1024:.1f} KB")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Download CUAD dataset for RAG evaluation"
    )
    parser.add_argument(
        "--output-dir",
        default="data/corpus",
        help="Output directory for contract files (default: data/corpus)",
    )
    parser.add_argument(
        "--max-contracts",
        type=int,
        default=None,
        help="Maximum number of contracts to download (default: all)",
    )
    args = parser.parse_args()

    download_cuad(
        output_dir=Path(args.output_dir),
        max_contracts=args.max_contracts,
    )


if __name__ == "__main__":
    main()
