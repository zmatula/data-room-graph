"""Citation and source resolution for retrieval results."""

import logging
import subprocess
import sys
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from ..database.neo4j_client import Neo4jClient, get_neo4j_client

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A citation linking to a source document."""

    chunk_id: str
    document_path: str
    document_name: str
    page: Optional[int] = None
    offset_start: Optional[int] = None
    offset_end: Optional[int] = None
    snippet: str = ""
    entities: Optional[list[str]] = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = []

    @property
    def display_text(self) -> str:
        """Human-readable citation text."""
        page_info = f", p.{self.page}" if self.page else ""
        return f"{self.document_name}{page_info}"

    @property
    def markdown_link(self) -> str:
        """Markdown formatted link."""
        return f"[{self.display_text}]({self.document_path})"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_path": self.document_path,
            "document_name": self.document_name,
            "page": self.page,
            "offset_start": self.offset_start,
            "offset_end": self.offset_end,
            "snippet": self.snippet,
            "entities": self.entities,
            "display_text": self.display_text,
        }


class CitationResolver:
    """Resolves and manages citations for retrieval results."""

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """Initialize the resolver.

        Args:
            neo4j_client: Neo4j client.
        """
        self.neo4j = neo4j_client or get_neo4j_client()

    def resolve_citation(
        self,
        chunk_id: str,
        document_path: str,
        document_name: str,
        page: Optional[int] = None,
        snippet: str = "",
    ) -> Citation:
        """Create a citation from chunk information.

        Args:
            chunk_id: ID of the source chunk.
            document_path: Path to the document.
            document_name: Name of the document.
            page: Optional page number.
            snippet: Text snippet.

        Returns:
            Citation object.
        """
        return Citation(
            chunk_id=chunk_id,
            document_path=document_path,
            document_name=document_name,
            page=page,
            snippet=snippet,
        )

    def get_chunk_citation(self, chunk_id: str) -> Optional[Citation]:
        """Get full citation information for a chunk.

        Args:
            chunk_id: ID of the chunk.

        Returns:
            Citation if found, None otherwise.
        """
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (c)<-[:HAS_ROOT|CONTAINS*]-(doc:Document)
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
        RETURN c.text AS text,
               c.page_start AS page,
               c.offset_start AS offset_start,
               c.offset_end AS offset_end,
               doc.full_path AS document_path,
               doc.filename AS document_name,
               collect(DISTINCT e.name) AS entities
        """

        results = self.neo4j.execute_read(query, {"chunk_id": chunk_id})

        if not results:
            return None

        result = results[0]
        snippet = result["text"][:200] if result["text"] else ""

        return Citation(
            chunk_id=chunk_id,
            document_path=result["document_path"],
            document_name=result["document_name"],
            page=result.get("page"),
            offset_start=result.get("offset_start"),
            offset_end=result.get("offset_end"),
            snippet=snippet,
            entities=result.get("entities", []),
        )

    def open_citation(self, citation: Citation) -> bool:
        """Open a citation in the default application.

        Args:
            citation: Citation to open.

        Returns:
            True if successful, False otherwise.
        """
        import os

        path = Path(citation.document_path).resolve()

        if not path.exists() or not path.is_file():
            logger.error(f"Document not found: {path}")
            return False

        try:
            if sys.platform == "win32":
                # Windows: use os.startfile which is safe
                os.startfile(str(path))
            elif sys.platform == "darwin":
                # macOS
                subprocess.run(["open", str(path)], check=False)
            else:
                # Linux
                subprocess.run(["xdg-open", str(path)], check=False)

            return True

        except Exception as e:
            logger.error(f"Failed to open document: {e}")
            return False

    def format_citations_markdown(
        self,
        citations: list[Citation],
        include_snippets: bool = False,
    ) -> str:
        """Format citations as markdown.

        Args:
            citations: List of citations to format.
            include_snippets: Whether to include text snippets.

        Returns:
            Markdown formatted citation list.
        """
        if not citations:
            return ""

        lines = ["### Sources\n"]

        for i, citation in enumerate(citations, 1):
            line = f"{i}. **{citation.display_text}**"

            if citation.entities:
                entities_str = ", ".join(citation.entities[:3])
                line += f" (mentions: {entities_str})"

            lines.append(line)

            if include_snippets and citation.snippet:
                snippet = citation.snippet.replace("\n", " ")
                lines.append(f"   > {snippet}...\n")

        return "\n".join(lines)

    def deduplicate_citations(
        self,
        citations: list[Citation],
    ) -> list[Citation]:
        """Remove duplicate citations.

        Args:
            citations: List of citations.

        Returns:
            Deduplicated list.
        """
        seen = set()
        unique = []

        for citation in citations:
            key = (citation.document_path, citation.page)
            if key not in seen:
                seen.add(key)
                unique.append(citation)

        return unique

    def group_by_document(
        self,
        citations: list[Citation],
    ) -> dict[str, list[Citation]]:
        """Group citations by source document.

        Args:
            citations: List of citations.

        Returns:
            Dictionary mapping document names to their citations.
        """
        groups: dict[str, list[Citation]] = {}

        for citation in citations:
            key = citation.document_name
            if key not in groups:
                groups[key] = []
            groups[key].append(citation)

        # Sort each group by page number
        for key in groups:
            groups[key].sort(key=lambda c: c.page or 0)

        return groups
