"""Central configuration for the RAG pipeline.

All settings are loaded from environment variables (via .env) and can be
overridden programmatically. Uses Pydantic BaseSettings for validation.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class EmbeddingProvider(StrEnum):
    OPENAI = "openai"
    LOCAL = "local"


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SENTENCE_WINDOW = "sentence_window"


class RetrievalMethod(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class EmbeddingConfig(BaseSettings):
    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    openai_model: str = "text-embedding-3-large"
    local_model: str = "BAAI/bge-m3"
    dimensions: int = 3072  # text-embedding-3-large default
    batch_size: int = 64

    model_config = {"env_prefix": "EMBEDDING_"}


class ChunkingConfig(BaseSettings):
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 50
    # Semantic chunking
    similarity_threshold: float = 0.75
    # Sentence window
    window_size: int = 3

    model_config = {"env_prefix": "CHUNKING_"}


class RetrievalConfig(BaseSettings):
    method: RetrievalMethod = RetrievalMethod.HYBRID
    top_k: int = 10
    # Hybrid RRF
    rrf_alpha: float = 0.5
    rrf_k: int = 60  # RRF constant
    # Reranking
    use_reranker: bool = True
    reranker_model: str = "rerank-v3.5"
    reranker_top_k: int = 5
    # Local reranker fallback
    local_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model_config = {"env_prefix": "RETRIEVAL_"}


class GenerationConfig(BaseSettings):
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 2048
    # Context assembly
    max_context_tokens: int = 6000
    min_relevance_score: float = 0.3

    model_config = {"env_prefix": "GENERATION_"}


class QdrantConfig(BaseSettings):
    url: str = "http://localhost:6333"
    collection_name: str = "rag_eval_bench"
    # Use in-memory storage for tests
    in_memory: bool = False

    model_config = {"env_prefix": "QDRANT_"}


class Settings(BaseSettings):
    """Top-level settings aggregating all sub-configs."""

    # API Keys
    openai_api_key: str = ""
    cohere_api_key: str | None = None

    # Paths
    corpus_dir: Path = Field(default=Path("data/corpus"))
    ground_truth_dir: Path = Field(default=Path("data/ground_truth"))
    results_dir: Path = Field(default=Path("results"))

    # Sub-configs
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)

    model_config = {"env_prefix": "", "env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings(**overrides) -> Settings:
    """Create settings with optional overrides."""
    return Settings(**overrides)
