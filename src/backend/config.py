"""Configuration management for Data Room Graph."""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    # OpenAI Configuration (for embeddings)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    embedding_model: str = Field(
        default="text-embedding-3-large", alias="EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(default=3072, alias="EMBEDDING_DIMENSIONS")

    # Anthropic Configuration (for LLM/Agents)
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    claude_model: str = Field(default="claude-opus-4-20250514", alias="CLAUDE_MODEL")

    # Unstructured.io Configuration
    unstructured_api_key: str = Field(default="", alias="UNSTRUCTURED_API_KEY")
    unstructured_api_url: str = Field(
        default="https://api.unstructuredapp.io/general/v0/general",
        alias="UNSTRUCTURED_API_URL",
    )

    # Application Paths
    app_data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".data-room-graph",
        alias="APP_DATA_DIR",
    )

    # Ingestion Settings
    embedding_batch_size: int = Field(default=100, alias="EMBEDDING_BATCH_SIZE")
    max_chunk_tokens: int = Field(default=512, alias="MAX_CHUNK_TOKENS")
    max_concurrent_files: int = Field(default=4, alias="MAX_CONCURRENT_FILES")

    # Retrieval Settings
    default_top_k: int = Field(default=10, alias="DEFAULT_TOP_K")
    vector_top_k_multiplier: int = Field(default=2, alias="VECTOR_TOP_K_MULTIPLIER")
    fulltext_weight: float = Field(default=0.8, alias="FULLTEXT_WEIGHT")

    # Text Processing Thresholds
    max_text_chars_embedding: int = Field(default=30000, alias="MAX_TEXT_CHARS_EMBEDDING")
    max_text_chars_extraction: int = Field(default=10000, alias="MAX_TEXT_CHARS_EXTRACTION")
    snippet_context_chars: int = Field(default=100, alias="SNIPPET_CONTEXT_CHARS")

    # Graph Expansion Scores
    sibling_expansion_score: float = Field(default=0.6, alias="SIBLING_EXPANSION_SCORE")
    entity_cooccurrence_base_score: float = Field(default=0.4, alias="ENTITY_COOCCURRENCE_BASE_SCORE")
    entity_cooccurrence_multiplier: float = Field(default=0.1, alias="ENTITY_COOCCURRENCE_MULTIPLIER")

    # Entity Extraction
    min_entity_confidence: float = Field(default=0.5, alias="MIN_ENTITY_CONFIDENCE")
    co_occurrence_threshold: int = Field(default=2, alias="CO_OCCURRENCE_THRESHOLD")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def ensure_app_data_dir(self) -> Path:
        """Ensure the application data directory exists."""
        self.app_data_dir.mkdir(parents=True, exist_ok=True)
        return self.app_data_dir

    @property
    def datarooms_dir(self) -> Path:
        """Directory for storing data room metadata."""
        path = self.app_data_dir / "datarooms"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logs_dir(self) -> Path:
        """Directory for storing log files."""
        path = self.app_data_dir / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment/file."""
    global _settings
    _settings = Settings()
    return _settings
