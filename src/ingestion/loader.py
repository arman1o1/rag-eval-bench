"""Document loader — extracts text from PDFs using PyMuPDF (fitz)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Heuristic: lines that look like section headers (short, possibly numbered).
_HEADER_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 :,\-–—]{2,80}$"
)


class DocumentLoader:
    """Load PDF files from a directory and return LangChain ``Document`` objects.

    Each page becomes one ``Document`` with metadata containing:
    - ``source``: file path
    - ``page``: 1-based page number
    - ``section``: last detected section header (best-effort heuristic)
    - ``total_pages``: total number of pages in the source PDF
    """

    def load(self, corpus_dir: Path) -> list[Document]:
        """Load all PDFs under *corpus_dir* (non-recursive by default).

        Args:
            corpus_dir: Directory containing ``.pdf`` files.

        Returns:
            List of :class:`Document` objects — one per page per file.
        """
        corpus_dir = Path(corpus_dir)
        if not corpus_dir.is_dir():
            logger.error("Corpus directory does not exist: %s", corpus_dir)
            return []

        pdf_paths = sorted(corpus_dir.glob("**/*.pdf"))
        if not pdf_paths:
            logger.warning("No PDF files found in %s", corpus_dir)
            return []

        logger.info("Found %d PDF(s) in %s", len(pdf_paths), corpus_dir)
        documents: list[Document] = []

        for path in pdf_paths:
            try:
                docs = self._load_pdf(path)
                documents.extend(docs)
                logger.info(
                    "Loaded %s — %d page(s) extracted", path.name, len(docs)
                )
            except Exception:
                logger.exception("Failed to load %s — skipping", path.name)

        logger.info("Total documents loaded: %d", len(documents))
        return documents

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_pdf(self, path: Path) -> list[Document]:
        """Extract per-page documents from a single PDF."""
        docs: list[Document] = []
        current_section: str = ""

        with fitz.open(str(path)) as pdf:
            total_pages = len(pdf)
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                if not text or not text.strip():
                    logger.debug(
                        "%s page %d: empty — skipped", path.name, page_num
                    )
                    continue

                # Try to detect section headers in this page.
                current_section = self._detect_section(text, current_section)

                docs.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": str(path),
                            "page": page_num,
                            "section": current_section,
                            "total_pages": total_pages,
                        },
                    )
                )
        return docs

    @staticmethod
    def _detect_section(text: str, fallback: str) -> str:
        """Return the last section-header-like line found in *text*.

        Uses a simple heuristic: short lines starting with an uppercase letter,
        optionally preceded by a number (e.g. ``3.2 Methodology``).  Falls back
        to *fallback* if nothing matches.
        """
        lines = text.split("\n")
        best = fallback
        for line in lines:
            stripped = line.strip()
            if _HEADER_PATTERN.match(stripped):
                best = stripped
        return best
