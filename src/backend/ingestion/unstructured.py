"""Unstructured.io API client for document parsing using the official SDK."""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional, Literal

from unstructured_client import UnstructuredClient as UnstructuredSDK
from unstructured_client.models import shared, operations, errors
from unstructured_client.utils import BackoffStrategy, RetryConfig

from ..config import get_settings

logger = logging.getLogger(__name__)


# Strategy type alias
StrategyType = Literal["hi_res", "fast", "auto", "ocr_only", "vlm"]


class UnstructuredElement:
    """Represents a parsed element from Unstructured.io."""

    def __init__(self, data: dict):
        """Initialize from API response data.

        Args:
            data: Element data from Unstructured.io API.
        """
        self.element_id: str = data.get("element_id", "")
        self.type: str = data.get("type", "NarrativeText")
        self.text: str = data.get("text", "")

        # Metadata
        metadata = data.get("metadata", {})
        self.filename: str = metadata.get("filename", "")
        self.file_directory: str = metadata.get("file_directory", "")
        self.filetype: str = metadata.get("filetype", "")

        # Hierarchy info
        self.parent_id: Optional[str] = metadata.get("parent_id")
        self.category_depth: int = metadata.get("category_depth", 0)

        # Location info
        self.page_number: Optional[int] = metadata.get("page_number")
        self.coordinates: Optional[dict] = metadata.get("coordinates")

        # For text offsets (we'll compute these)
        self.offset_start: int = 0
        self.offset_end: int = 0

    @property
    def is_title(self) -> bool:
        """Check if this element is a title/header."""
        return self.type == "Title"

    @property
    def is_text(self) -> bool:
        """Check if this element contains text content."""
        return self.type in ("NarrativeText", "ListItem", "UncategorizedText")

    @property
    def is_table(self) -> bool:
        """Check if this element is a table."""
        return self.type == "Table"

    def __repr__(self) -> str:
        return f"UnstructuredElement(type={self.type}, text={self.text[:50]}...)"


class UnstructuredClient:
    """Client for the Unstructured.io API using the official SDK."""

    # Supported file types
    SUPPORTED_EXTENSIONS = {
        ".pdf", ".docx", ".doc", ".pptx", ".ppt",
        ".xlsx", ".xls", ".txt", ".html", ".htm",
        ".md", ".rtf", ".odt", ".epub", ".xml",
        ".json", ".csv",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        max_workers: int = 4,
    ):
        """Initialize the client.

        Args:
            api_key: Unstructured.io API key. Defaults to settings.
            api_url: API URL. Defaults to settings.
            max_workers: Maximum number of worker threads for parallel processing.
        """
        settings = get_settings()
        self.api_key = api_key or settings.unstructured_api_key
        self.api_url = api_url or settings.unstructured_api_url
        self.max_workers = max_workers

        if not self.api_key:
            logger.warning(
                "No Unstructured.io API key configured. "
                "Set UNSTRUCTURED_API_KEY in .env"
            )

        # Thread pool for parallel processing - SDK is synchronous so we use threads
        self._executor: Optional[ThreadPoolExecutor] = None

        # Store base URL for creating per-thread SDK clients
        self._base_url = self._extract_base_url(self.api_url)
        self._retry_config = RetryConfig(
            strategy="backoff",
            retry_connection_errors=True,
            backoff=BackoffStrategy(
                initial_interval=1000,      # 1 second
                max_interval=60000,         # 60 seconds max between retries
                exponent=1.5,
                max_elapsed_time=300000,    # 5 minutes total max retry time
            ),
        )

    def _extract_base_url(self, url: str) -> str:
        """Extract base URL from full endpoint URL.

        Args:
            url: Full API URL like https://api.unstructuredapp.io/general/v0/general

        Returns:
            Base URL like https://api.unstructuredapp.io
        """
        # The SDK handles the endpoint path, so we just need the base
        if "/general/v0/general" in url:
            return url.replace("/general/v0/general", "")
        return url

    @property
    def executor(self) -> ThreadPoolExecutor:
        """Get the thread pool executor, creating it if needed."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="unstructured_worker",
            )
        return self._executor

    def _create_sdk_client(self) -> UnstructuredSDK:
        """Create a new SDK client instance.

        Creating new clients per-call ensures thread safety since the SDK
        may have internal state that isn't thread-safe.
        """
        return UnstructuredSDK(
            api_key_auth=self.api_key,
            server_url=self._base_url,
            retry_config=self._retry_config,
        )

    def shutdown(self):
        """Shutdown the thread pool executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def is_supported(self, file_path: Path) -> bool:
        """Check if a file type is supported.

        Args:
            file_path: Path to the file.

        Returns:
            True if the file type is supported.
        """
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _parse_document_sync(
        self,
        file_path: Path,
        strategy: StrategyType = "hi_res",
        include_page_breaks: bool = True,
        languages: Optional[list[str]] = None,
        use_vlm: bool = False,
        vlm_model: str = "gpt-4o",
        vlm_model_provider: str = "openai",
    ) -> list[UnstructuredElement]:
        """Synchronously parse a document. Called from thread pool.

        Creates a new SDK client per-call to ensure thread safety.
        """
        logger.info(f"Parsing document: {file_path}")

        # Create a new SDK client for this thread to ensure thread safety
        client = self._create_sdk_client()

        # Read file content
        with open(file_path, "rb") as f:
            file_content = f.read()

        # Prepare files object
        files = shared.Files(
            content=file_content,
            file_name=file_path.name,
        )

        # Build partition parameters
        partition_params = {
            "files": files,
            "include_page_breaks": include_page_breaks,
            "coordinates": True,
        }

        # Set strategy
        if use_vlm or strategy == "vlm":
            partition_params["strategy"] = shared.Strategy.VLM
            partition_params["vlm_model"] = vlm_model
            partition_params["vlm_model_provider"] = vlm_model_provider
        else:
            strategy_map = {
                "hi_res": shared.Strategy.HI_RES,
                "fast": shared.Strategy.FAST,
                "auto": shared.Strategy.AUTO,
                "ocr_only": shared.Strategy.OCR_ONLY,
            }
            partition_params["strategy"] = strategy_map.get(strategy, shared.Strategy.HI_RES)

        # Add hi_res model name for non-VLM hi_res strategy
        if strategy == "hi_res" and not use_vlm:
            partition_params["hi_res_model_name"] = "yolox"

        # Add languages if specified
        if languages:
            partition_params["languages"] = languages

        # For PDFs, enable page splitting for better performance
        if file_path.suffix.lower() == ".pdf":
            partition_params["split_pdf_page"] = True
            partition_params["split_pdf_allow_failed"] = True
            partition_params["split_pdf_concurrency_level"] = 10

        # Create request
        request = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(**partition_params)
        )

        try:
            # Synchronous SDK call - this runs in a worker thread
            response = client.general.partition(request=request)

            # Convert to UnstructuredElement objects
            elements = [UnstructuredElement(el) for el in response.elements]

            # Compute text offsets
            self._compute_offsets(elements)

            logger.info(f"Parsed {len(elements)} elements from {file_path.name}")
            return elements

        except errors.HTTPValidationError as e:
            logger.error(f"Validation error parsing {file_path}: {e}")
            raise UnstructuredAPIError(f"Validation error: {e}", status_code=422) from e
        except errors.ServerError as e:
            logger.error(f"Server error parsing {file_path}: {e}")
            raise UnstructuredAPIError(f"Server error: {e}", status_code=500) from e
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            raise UnstructuredAPIError(f"API error: {e}") from e

    async def parse_document(
        self,
        file_path: Path,
        strategy: StrategyType = "hi_res",
        include_page_breaks: bool = True,
        languages: Optional[list[str]] = None,
        use_vlm: bool = False,
        vlm_model: str = "gpt-4o",
        vlm_model_provider: str = "openai",
    ) -> list[UnstructuredElement]:
        """Parse a document using the Unstructured.io API.

        Runs the synchronous SDK call in a thread pool to avoid blocking
        the asyncio event loop. Each call uses a separate SDK client instance
        to ensure thread safety.

        Args:
            file_path: Path to the document file.
            strategy: Parsing strategy ('hi_res', 'fast', 'ocr_only', 'auto', 'vlm').
            include_page_breaks: Whether to include page break elements.
            languages: List of languages in the document (e.g., ['eng', 'spa']).
            use_vlm: Whether to use Vision Language Model for better accuracy.
            vlm_model: VLM model to use (default: gpt-4o).
            vlm_model_provider: VLM provider (default: openai).

        Returns:
            List of parsed elements.

        Raises:
            ValueError: If file type is not supported.
            FileNotFoundError: If file does not exist.
            UnstructuredAPIError: If API request fails.
        """
        # Validate inputs synchronously (fast operations)
        if not self.is_supported(file_path):
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Run the blocking SDK call in a thread pool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            partial(
                self._parse_document_sync,
                file_path,
                strategy,
                include_page_breaks,
                languages,
                use_vlm,
                vlm_model,
                vlm_model_provider,
            ),
        )

    def _compute_offsets(self, elements: list[UnstructuredElement]):
        """Compute text offsets for elements.

        Args:
            elements: List of elements to compute offsets for.
        """
        current_offset = 0
        for element in elements:
            element.offset_start = current_offset
            element.offset_end = current_offset + len(element.text)
            current_offset = element.offset_end + 1  # +1 for newline separator

    async def parse_documents_batch(
        self,
        file_paths: list[Path],
        max_concurrent: int = 5,
        **kwargs,
    ) -> dict[Path, list[UnstructuredElement]]:
        """Parse multiple documents concurrently.

        Args:
            file_paths: List of file paths to parse.
            max_concurrent: Maximum concurrent requests.
            **kwargs: Additional arguments passed to parse_document.

        Returns:
            Dictionary mapping file paths to their parsed elements.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}

        async def parse_with_semaphore(path: Path):
            async with semaphore:
                try:
                    elements = await self.parse_document(path, **kwargs)
                    return path, elements
                except Exception as e:
                    logger.error(f"Failed to parse {path}: {e}")
                    return path, []

        tasks = [parse_with_semaphore(path) for path in file_paths]
        completed = await asyncio.gather(*tasks)

        for path, elements in completed:
            results[path] = elements

        return results


class UnstructuredAPIError(Exception):
    """Custom exception for Unstructured API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        """Initialize the error.

        Args:
            message: Error message.
            status_code: HTTP status code if available.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
