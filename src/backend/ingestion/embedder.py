"""Embedding generation using OpenAI API."""

import logging
import asyncio
from typing import Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings using OpenAI's embedding models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """Initialize the embedder.

        Args:
            api_key: OpenAI API key. Defaults to settings.
            model: Embedding model name. Defaults to settings.
            dimensions: Embedding dimensions. Defaults to settings.
        """
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        self.batch_size = settings.embedding_batch_size

        if not self.api_key:
            logger.warning(
                "No OpenAI API key configured. "
                "Set OPENAI_API_KEY in .env"
            )

        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        """Get the OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        if not text.strip():
            return [0.0] * self.dimensions

        # Truncate if too long (model limit is ~8191 tokens)
        if len(text) > 30000:
            text = text[:30000]

        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
        )

        return response.data[0].embedding

    async def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            show_progress: Whether to log progress.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            if show_progress:
                logger.info(f"Embedding batch {batch_num}/{total_batches}")

            batch_embeddings = await self._embed_batch(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: Batch of texts.

        Returns:
            List of embeddings.
        """
        # Clean and truncate texts
        cleaned = []
        for text in texts:
            text = text.strip() if text else ""
            if len(text) > 30000:
                text = text[:30000]
            cleaned.append(text if text else " ")  # API doesn't accept empty strings

        response = await self.client.embeddings.create(
            model=self.model,
            input=cleaned,
            dimensions=self.dimensions,
        )

        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def embed_with_context(
        self,
        text: str,
        context: str,
        context_weight: float = 0.3,
    ) -> list[float]:
        """Generate embedding with section context prepended.

        Args:
            text: Main text to embed.
            context: Context (e.g., section headers) to prepend.
            context_weight: Not used, but could weight context differently.

        Returns:
            Embedding vector.
        """
        if context:
            combined = f"{context}\n\n{text}"
        else:
            combined = text

        return await self.embed_text(combined)

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text.

        Uses a rough estimate of 4 characters per token.

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count.
        """
        return len(text) // 4

    async def close(self):
        """Close the client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None


# Singleton instance
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Get the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


async def close_embedder() -> None:
    """Close the global embedder instance."""
    global _embedder
    if _embedder is not None:
        await _embedder.close()
        _embedder = None
