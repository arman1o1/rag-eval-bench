"""End-to-end ingestion pipeline: load → chunk → embed → index."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from src.config import (
    ChunkingConfig,
    ChunkingStrategy,
    QdrantConfig,
    Settings,
    get_settings,
)
from src.ingestion.chunker import get_chunker
from src.ingestion.embedder import EmbeddingService
from src.ingestion.indexer import QdrantIndexer
from src.ingestion.loader import DocumentLoader

logger = logging.getLogger(__name__)
console = Console()


class IngestionPipeline:
    """Orchestrate the full ingestion flow.

    1. **Load** PDFs from a corpus directory.
    2. **Chunk** them using the chosen strategy.
    3. **Embed** the resulting chunks.
    4. **Index** vectors into Qdrant + build a BM25 sparse index.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.loader = DocumentLoader()
        self.embedding_service = EmbeddingService(self.settings.embedding)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        corpus_dir: Path | None = None,
        strategy: ChunkingStrategy | None = None,
    ) -> None:
        """Execute the pipeline for a single chunking strategy.

        Args:
            corpus_dir: Override for ``settings.corpus_dir``.
            strategy: Override for ``settings.chunking.strategy``.
        """
        corpus_dir = corpus_dir or self.settings.corpus_dir
        strategy = strategy or self.settings.chunking.strategy

        console.rule(f"[bold green]Ingestion — {strategy.value}")

        # 1. Load
        console.print("[cyan]Loading documents …[/]")
        docs = self.loader.load(corpus_dir)
        if not docs:
            console.print("[red]No documents loaded — aborting.[/]")
            return
        console.print(f"  ✓ {len(docs)} page-documents loaded")

        # 2. Chunk
        console.print(f"[cyan]Chunking ({strategy.value}) …[/]")
        chunking_config = ChunkingConfig(strategy=strategy)
        chunker = get_chunker(
            chunking_config, embedding_service=self.embedding_service
        )
        chunks = chunker.chunk(docs)
        console.print(f"  ✓ {len(chunks)} chunks produced")

        # 3. Embed
        console.print("[cyan]Embedding chunks …[/]")
        texts = [c.page_content for c in chunks]
        embeddings = self.embedding_service.embed_texts(texts)
        console.print(f"  ✓ {len(embeddings)} embeddings computed")

        # 4. Index
        console.print("[cyan]Indexing into Qdrant + BM25 …[/]")
        collection_name = f"{self.settings.qdrant.collection_name}_{strategy.value}"
        qdrant_config = QdrantConfig(
            url=self.settings.qdrant.url,
            collection_name=collection_name,
            in_memory=self.settings.qdrant.in_memory,
        )
        bm25_path = Path("data") / f"bm25_{strategy.value}.pkl"
        indexer = QdrantIndexer(
            qdrant_config,
            vector_dim=self.embedding_service.dimensions,
            bm25_path=bm25_path,
        )
        indexer.create_collection(recreate=True)
        indexer.index(chunks, embeddings)
        console.print(
            f"  ✓ Indexed into [bold]{collection_name}[/] + "
            f"BM25 at [bold]{bm25_path}[/]"
        )
        console.rule("[bold green]Done")

    def run_all(self, corpus_dir: Path | None = None) -> None:
        """Run the pipeline for every chunking strategy.

        Creates separate Qdrant collections suffixed by strategy name.
        """
        for strategy in ChunkingStrategy:
            self.run(corpus_dir=corpus_dir, strategy=strategy)


# ======================================================================
# CLI
# ======================================================================


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG ingestion pipeline — load, chunk, embed, index.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Directory containing PDF files (default: from settings).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="recursive",
        choices=[s.value for s in ChunkingStrategy] + ["all"],
        help="Chunking strategy to use, or 'all' to run every strategy.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )

    args = _parse_args(argv)
    pipeline = IngestionPipeline()

    if args.strategy == "all":
        pipeline.run_all(corpus_dir=args.corpus_dir)
    else:
        strategy = ChunkingStrategy(args.strategy)
        pipeline.run(corpus_dir=args.corpus_dir, strategy=strategy)


if __name__ == "__main__":
    main()
