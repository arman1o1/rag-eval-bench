"""Generate ground-truth Q&A pairs from CUAD annotations and/or LLM augmentation.

Reads CUAD-style contract annotations, extracts question–answer pairs per
annotated clause, and optionally uses an LLM to generate additional Q&A
pairs from un-annotated sections.

# Usage:
#     python scripts/generate_qa.py --corpus-dir data/corpus
#         --output-dir data/ground_truth --num-pairs 50
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# CUAD clause categories used as question templates
CUAD_CLAUSE_TYPES: list[dict[str, str]] = [
    {"type": "Document Name", "question": "What is the name of this contract?"},
    {"type": "Parties", "question": "Who are the parties involved in this contract?"},
    {"type": "Agreement Date", "question": "What is the effective date of this agreement?"},
    {"type": "Expiration Date", "question": "When does this contract expire?"},
    {"type": "Renewal Term", "question": "What are the renewal terms?"},
    {"type": "Governing Law", "question": "Which jurisdiction's law governs this contract?"},
    {
        "type": "Termination For Convenience",
        "question": "Can either party terminate for convenience?",
    },
    {"type": "Non-Compete", "question": "Does this contract contain a non-compete clause?"},
    {"type": "Exclusivity", "question": "Is there an exclusivity provision?"},
    {"type": "Indemnification", "question": "What are the indemnification obligations?"},
    {"type": "Limitation Of Liability", "question": "What limitations of liability exist?"},
    {
        "type": "IP Ownership Assignment",
        "question": "How is intellectual property ownership assigned?",
    },
    {"type": "Confidentiality", "question": "What are the confidentiality obligations?"},
]


def generate_from_cuad_annotations(corpus_dir: Path, num_pairs: int = 50) -> list[dict[str, Any]]:
    """Generate Q&A pairs from CUAD-style annotations.

    If the CUAD dataset is available via HuggingFace, extracts annotated
    question-answer pairs. Otherwise, generates synthetic pairs from
    contract text files using heuristics.

    Args:
        corpus_dir: Directory containing contract documents.
        num_pairs: Target number of Q&A pairs to generate.

    Returns:
        List of ground-truth Q&A dicts.
    """
    qa_pairs: list[dict[str, Any]] = []

    # Try loading CUAD annotations from HuggingFace
    try:
        from datasets import load_dataset

        logger.info("Loading CUAD annotations from HuggingFace...")
        dataset = load_dataset("theatticusproject/cuad-qa", split="test")

        seen_questions: set[str] = set()
        for item in dataset:
            if len(qa_pairs) >= num_pairs:
                break

            question = item.get("question", "")
            answers = item.get("answers", {})
            answer_texts = answers.get("text", [])
            context = item.get("context", "")
            title = item.get("title", "unknown")

            if not answer_texts or not answer_texts[0].strip():
                continue

            # Deduplicate
            q_key = f"{title}::{question}"
            if q_key in seen_questions:
                continue
            seen_questions.add(q_key)

            qa_pairs.append(
                {
                    "question": question,
                    "ground_truth_answer": answer_texts[0],
                    "source_doc": title,
                    "source_page": 0,
                    "source_section": "",
                    "context_texts": [context] if context else [],
                }
            )

        logger.info("Extracted %d Q&A pairs from CUAD annotations", len(qa_pairs))

    except (ImportError, Exception) as exc:
        logger.warning("Could not load CUAD annotations: %s", exc)
        logger.info("Falling back to synthetic generation from corpus files...")
        qa_pairs = _generate_synthetic(corpus_dir, num_pairs)

    return qa_pairs[:num_pairs]


def _generate_synthetic(corpus_dir: Path, num_pairs: int) -> list[dict[str, Any]]:
    """Generate synthetic Q&A pairs from contract text files using heuristics.

    Creates Q&A pairs by extracting key sentences from contract documents
    and framing them as questions.
    """
    qa_pairs: list[dict[str, Any]] = []
    files = list(corpus_dir.glob("*.txt")) + list(corpus_dir.glob("*.pdf"))

    if not files:
        logger.warning("No corpus files found in %s", corpus_dir)
        return qa_pairs

    for filepath in files:
        if len(qa_pairs) >= num_pairs:
            break

        try:
            text = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if not text.strip():
            continue

        # Split into paragraphs and create Q&A from first substantial paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]

        for clause_info in CUAD_CLAUSE_TYPES:
            if len(qa_pairs) >= num_pairs:
                break

            # Search for paragraphs mentioning the clause type keywords
            keywords = clause_info["type"].lower().split()
            for para in paragraphs:
                if any(kw in para.lower() for kw in keywords):
                    # Use first sentence as answer
                    sentences = [s.strip() for s in para.split(".") if s.strip()]
                    if sentences:
                        qa_pairs.append(
                            {
                                "question": clause_info["question"],
                                "ground_truth_answer": sentences[0] + ".",
                                "source_doc": filepath.name,
                                "source_page": 0,
                                "source_section": clause_info["type"],
                                "context_texts": [para],
                            }
                        )
                        break

    return qa_pairs


def generate_with_llm(
    corpus_dir: Path,
    num_pairs: int = 20,
    model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    """Generate additional Q&A pairs using an LLM.

    Sends contract text to the LLM and asks it to generate diverse Q&A pairs
    covering different legal aspects.

    Args:
        corpus_dir: Directory containing contract documents.
        num_pairs: Number of additional pairs to generate.
        model: OpenAI model to use.

    Returns:
        List of LLM-generated Q&A dicts.
    """
    try:
        from openai import OpenAI

        from src.config import get_settings
    except ImportError:
        logger.error("openai library required for LLM generation")
        return []

    settings = get_settings()
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY required for LLM-based Q&A generation")
        return []

    client = OpenAI(api_key=settings.openai_api_key)
    qa_pairs: list[dict[str, Any]] = []

    files = list(corpus_dir.glob("*.txt"))[:5]  # Limit to 5 docs for cost

    for filepath in files:
        if len(qa_pairs) >= num_pairs:
            break

        text = filepath.read_text(encoding="utf-8", errors="ignore")
        # Truncate to fit context window
        text_truncated = text[:8000]

        prompt = (
            "You are a legal Q&A dataset generator. Given the following contract text, "
            "generate 5 diverse question-answer pairs. Each pair should test understanding "
            "of different legal concepts (parties, dates, obligations, liabilities, etc.).\n\n"
            "Return JSON array with objects having keys: question, answer, section.\n\n"
            f"Contract text:\n{text_truncated}"
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            pairs = parsed.get("pairs", parsed.get("questions", []))
            if isinstance(parsed, list):
                pairs = parsed

            for p in pairs:
                if len(qa_pairs) >= num_pairs:
                    break
                qa_pairs.append(
                    {
                        "question": p.get("question", ""),
                        "ground_truth_answer": p.get("answer", ""),
                        "source_doc": filepath.name,
                        "source_page": 0,
                        "source_section": p.get("section", ""),
                        "context_texts": [text_truncated[:2000]],
                    }
                )

        except Exception:
            logger.exception("LLM generation failed for %s", filepath.name)

    logger.info("LLM generated %d Q&A pairs", len(qa_pairs))
    return qa_pairs


def save_qa_pairs(pairs: list[dict[str, Any]], output_dir: Path) -> Path:
    """Save Q&A pairs as a JSON file.

    Args:
        pairs: List of Q&A dictionaries.
        output_dir: Output directory.

    Returns:
        Path to the saved JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ground_truth_qa.json"
    output_path.write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved %d Q&A pairs to %s", len(pairs), output_path)
    return output_path


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate ground-truth Q&A pairs for RAG evaluation"
    )
    parser.add_argument(
        "--corpus-dir",
        default="data/corpus",
        help="Directory containing contract documents",
    )
    parser.add_argument(
        "--output-dir",
        default="data/ground_truth",
        help="Output directory for Q&A JSON files",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=50,
        help="Number of Q&A pairs to generate (default: 50)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Also generate additional pairs using an LLM",
    )
    parser.add_argument(
        "--llm-pairs",
        type=int,
        default=20,
        help="Number of LLM-generated pairs (default: 20)",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)

    # Generate from annotations
    pairs = generate_from_cuad_annotations(corpus_dir, num_pairs=args.num_pairs)
    print(f"\n📝 Generated {len(pairs)} Q&A pairs from annotations/heuristics")

    # Optionally augment with LLM
    if args.use_llm:
        llm_pairs = generate_with_llm(corpus_dir, num_pairs=args.llm_pairs)
        pairs.extend(llm_pairs)
        print(f"🤖 Added {len(llm_pairs)} LLM-generated pairs")

    # Save
    output_path = save_qa_pairs(pairs, output_dir)
    print(f"\n✅ Saved {len(pairs)} total Q&A pairs to {output_path}")


if __name__ == "__main__":
    main()
