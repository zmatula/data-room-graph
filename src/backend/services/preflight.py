"""Preflight validation service for checking service connectivity before ingestion."""

import logging
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from ..config import get_settings
from ..database.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Supported file extensions for ingestion
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".txt", ".md", ".html", ".htm", ".xml", ".json", ".csv",
}


@dataclass
class PreflightResult:
    """Result of a single preflight validation check."""

    service: str  # "neo4j", "unstructured", "openai", "folder"
    status: str  # "ok", "error", "warning"
    message: str
    details: Optional[dict] = field(default_factory=dict)


@dataclass
class PreflightReport:
    """Complete preflight validation report."""

    ready: bool
    results: list[PreflightResult]
    blocking_errors: list[str]

    @classmethod
    def from_results(cls, results: list[PreflightResult]) -> "PreflightReport":
        """Create a report from a list of results."""
        blocking_errors = [
            f"{r.service}: {r.message}"
            for r in results
            if r.status == "error"
        ]
        return cls(
            ready=len(blocking_errors) == 0,
            results=results,
            blocking_errors=blocking_errors,
        )


class PreflightValidator:
    """Validates all required services before starting ingestion."""

    def __init__(self):
        """Initialize the validator."""
        self._settings = get_settings()

    async def validate_all(self, folder_path: Optional[str] = None) -> PreflightReport:
        """Run all preflight validation checks.

        Args:
            folder_path: Optional path to folder being ingested.

        Returns:
            PreflightReport with all validation results.
        """
        results: list[PreflightResult] = []

        # Run all validations
        results.append(await self.validate_neo4j())
        results.append(await self.validate_unstructured())
        results.append(await self.validate_openai())

        if folder_path:
            results.append(await self.validate_folder(folder_path))

        return PreflightReport.from_results(results)

    async def validate_neo4j(self) -> PreflightResult:
        """Validate Neo4j connectivity.

        Returns:
            PreflightResult for Neo4j validation.
        """
        try:
            client = Neo4jClient()
            if client.verify_connectivity():
                # Get server info for details
                with client.session() as session:
                    result = session.run("CALL dbms.components() YIELD name, versions")
                    components = list(result)
                    version = components[0]["versions"][0] if components else "unknown"

                client.close()
                return PreflightResult(
                    service="neo4j",
                    status="ok",
                    message=f"Connected (v{version})",
                    details={"version": version, "uri": self._settings.neo4j_uri},
                )
            else:
                client.close()
                return PreflightResult(
                    service="neo4j",
                    status="error",
                    message="Connection verification failed",
                    details={
                        "suggestion": "Ensure Neo4j is running: python scripts/setup_neo4j.py start",
                        "uri": self._settings.neo4j_uri,
                    },
                )
        except Exception as e:
            error_msg = str(e)
            suggestion = "Ensure Neo4j is running: python scripts/setup_neo4j.py start"

            if "authentication" in error_msg.lower() or "auth" in error_msg.lower():
                suggestion = "Check NEO4J_PASSWORD in your .env file"
            elif "connection refused" in error_msg.lower():
                suggestion = "Neo4j is not running. Start it with: python scripts/setup_neo4j.py start"

            return PreflightResult(
                service="neo4j",
                status="error",
                message=f"Connection failed: {error_msg}",
                details={"suggestion": suggestion, "uri": self._settings.neo4j_uri},
            )

    async def validate_unstructured(self) -> PreflightResult:
        """Validate Unstructured.io API connectivity.

        Multi-stage validation:
        1. Config check - API key present
        2. DNS resolution - Can resolve api.unstructuredapp.io
        3. HTTPS connectivity - Can reach the endpoint
        4. Auth validation - Check for 401 vs other responses

        Returns:
            PreflightResult for Unstructured validation.
        """
        # Stage 1: Config check
        if not self._settings.unstructured_api_key:
            return PreflightResult(
                service="unstructured",
                status="error",
                message="API key not configured",
                details={"suggestion": "Set UNSTRUCTURED_API_KEY in your .env file"},
            )

        # Stage 2: DNS resolution
        hostname = "api.unstructuredapp.io"
        try:
            socket.getaddrinfo(hostname, 443)
        except socket.gaierror as e:
            return PreflightResult(
                service="unstructured",
                status="error",
                message=f"DNS resolution failed for {hostname}",
                details={
                    "error": str(e),
                    "suggestion": "Check network connection or DNS settings. "
                    "If behind a corporate proxy, ensure proxy settings are configured.",
                },
            )

        # Stage 3: HTTPS connectivity + Auth validation
        # Use httpx for a lightweight connectivity check
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Use GET request to test connectivity
                # The API will return 405 Method Not Allowed for GET (expects POST)
                # but that still proves connectivity and auth
                response = await client.get(
                    self._settings.unstructured_api_url,
                    headers={
                        "unstructured-api-key": self._settings.unstructured_api_key,
                    },
                )

                if response.status_code == 401:
                    return PreflightResult(
                        service="unstructured",
                        status="error",
                        message="Invalid API key (401 Unauthorized)",
                        details={
                            "suggestion": "Check UNSTRUCTURED_API_KEY in your .env file",
                            "status_code": 401,
                        },
                    )
                elif response.status_code == 403:
                    return PreflightResult(
                        service="unstructured",
                        status="error",
                        message="API key forbidden (403)",
                        details={
                            "suggestion": "Your API key may have been revoked or has insufficient permissions",
                            "status_code": 403,
                        },
                    )
                elif response.status_code in (200, 405, 422):
                    # 200 = OK, 405 = Method Not Allowed (expected for GET), 422 = Validation Error
                    # All indicate the API is reachable and auth is working
                    return PreflightResult(
                        service="unstructured",
                        status="ok",
                        message="API reachable (using official SDK)",
                        details={
                            "url": self._settings.unstructured_api_url,
                            "status_code": response.status_code,
                            "sdk": "unstructured-client",
                        },
                    )
                else:
                    # Unexpected status code - treat as warning, not error
                    return PreflightResult(
                        service="unstructured",
                        status="warning",
                        message=f"Unexpected status code: {response.status_code}",
                        details={
                            "url": self._settings.unstructured_api_url,
                            "status_code": response.status_code,
                            "suggestion": "The API may still work, but returned an unexpected response",
                        },
                    )

        except httpx.ConnectError as e:
            return PreflightResult(
                service="unstructured",
                status="error",
                message=f"Connection failed: {e}",
                details={
                    "suggestion": "Check firewall/proxy settings. "
                    "If behind a corporate network, ensure HTTPS traffic to api.unstructuredapp.io is allowed.",
                    "url": self._settings.unstructured_api_url,
                },
            )
        except httpx.TimeoutException:
            return PreflightResult(
                service="unstructured",
                status="error",
                message="Connection timed out",
                details={
                    "suggestion": "The API server may be slow or unreachable. "
                    "Check your network connection.",
                    "url": self._settings.unstructured_api_url,
                },
            )
        except Exception as e:
            return PreflightResult(
                service="unstructured",
                status="error",
                message=f"Unexpected error: {e}",
                details={
                    "url": self._settings.unstructured_api_url,
                    "error_type": type(e).__name__,
                },
            )

    async def validate_openai(self) -> PreflightResult:
        """Validate OpenAI API connectivity for embeddings.

        Returns:
            PreflightResult for OpenAI validation.
        """
        # Stage 1: Config check
        if not self._settings.openai_api_key:
            return PreflightResult(
                service="openai",
                status="error",
                message="API key not configured",
                details={"suggestion": "Set OPENAI_API_KEY in your .env file"},
            )

        # Stage 2: API connectivity check
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Use the models endpoint to verify connectivity and auth
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={
                        "Authorization": f"Bearer {self._settings.openai_api_key}",
                    },
                )

                if response.status_code == 401:
                    return PreflightResult(
                        service="openai",
                        status="error",
                        message="Invalid API key (401 Unauthorized)",
                        details={"suggestion": "Check OPENAI_API_KEY in your .env file"},
                    )
                elif response.status_code == 429:
                    return PreflightResult(
                        service="openai",
                        status="warning",
                        message="Rate limited (429)",
                        details={
                            "suggestion": "You may be hitting rate limits. "
                            "Ingestion may be slower than usual."
                        },
                    )
                elif response.status_code == 200:
                    return PreflightResult(
                        service="openai",
                        status="ok",
                        message="API key valid",
                        details={"model": self._settings.embedding_model},
                    )
                else:
                    return PreflightResult(
                        service="openai",
                        status="warning",
                        message=f"Unexpected status: {response.status_code}",
                        details={"status_code": response.status_code},
                    )

        except httpx.ConnectError as e:
            return PreflightResult(
                service="openai",
                status="error",
                message=f"Connection failed: {e}",
                details={
                    "suggestion": "Check network connection and firewall settings",
                },
            )
        except httpx.TimeoutException:
            return PreflightResult(
                service="openai",
                status="error",
                message="Connection timed out",
                details={"suggestion": "Check network connection"},
            )
        except Exception as e:
            return PreflightResult(
                service="openai",
                status="error",
                message=f"Unexpected error: {e}",
                details={"error_type": type(e).__name__},
            )

    async def validate_folder(self, path: str) -> PreflightResult:
        """Validate the folder to be ingested.

        Args:
            path: Path to the folder.

        Returns:
            PreflightResult for folder validation.
        """
        folder_path = Path(path)

        # Check if path exists
        if not folder_path.exists():
            return PreflightResult(
                service="folder",
                status="error",
                message=f"Path does not exist: {path}",
                details={"suggestion": "Check the folder path"},
            )

        # Check if it's a directory
        if not folder_path.is_dir():
            return PreflightResult(
                service="folder",
                status="error",
                message=f"Path is not a directory: {path}",
                details={"suggestion": "Drop a folder, not a file"},
            )

        # Count supported files
        supported_files = []
        unsupported_files = []

        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    supported_files.append(file_path.name)
                else:
                    unsupported_files.append(file_path.name)

        if not supported_files:
            return PreflightResult(
                service="folder",
                status="error",
                message="No supported files found",
                details={
                    "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
                    "unsupported_files": unsupported_files[:10],  # Show first 10
                    "suggestion": f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                },
            )

        # Success - found supported files
        return PreflightResult(
            service="folder",
            status="ok",
            message=f"{len(supported_files)} supported file(s) found",
            details={
                "supported_files": len(supported_files),
                "unsupported_files": len(unsupported_files),
                "folder": folder_path.name,
            },
        )
