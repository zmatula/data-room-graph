"""Hierarchy-aware expansion for retrieval.

This module provides hierarchy-aware expansion for retrieval results.
It supports both:
- Legacy structure: Document -> Page -> Section -> Chunk
- GraphRAG structure: Document -> Chunk (with metadata properties)
"""

import logging
from typing import Optional, Literal

from ..database.neo4j_client import Neo4jClient, get_neo4j_client

logger = logging.getLogger(__name__)


# Schema type for query adaptation
SchemaType = Literal["legacy", "graphrag", "auto"]


class HierarchyExpander:
    """Expands retrieval results using document hierarchy.

    This expander supports both legacy and GraphRAG schema structures:
    - Legacy: Uses Section traversal via CONTAINS relationships
    - GraphRAG: Uses metadata-based filtering on section_path property
    """

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        schema_type: SchemaType = "auto",
    ):
        """Initialize the expander.

        Args:
            neo4j_client: Neo4j client.
            schema_type: Schema type to use for queries.
        """
        self.neo4j = neo4j_client or get_neo4j_client()
        self._schema_type = schema_type
        self._detected_schema: Optional[SchemaType] = None

    @property
    def schema_type(self) -> SchemaType:
        """Get the effective schema type (auto-detected if needed)."""
        if self._schema_type != "auto":
            return self._schema_type

        if self._detected_schema is None:
            self._detected_schema = self._detect_schema()

        return self._detected_schema

    def _detect_schema(self) -> SchemaType:
        """Detect which schema is in use based on graph structure."""
        query = "MATCH (s:Section) RETURN count(s) > 0 AS has_sections LIMIT 1"
        result = self.neo4j.execute_read(query, {})

        if result and result[0].get("has_sections"):
            return "legacy"
        return "graphrag"

    def expand_chunks(
        self,
        chunk_ids: list[str],
        max_expansion: int = 20,
    ) -> list[dict]:
        """Expand a set of chunks by traversing the graph.

        Expansion includes:
        - Sibling chunks (NEXT relationships within section)
        - Other chunks in the same section
        - Chunks mentioning the same entities

        Args:
            chunk_ids: IDs of chunks to expand from.
            max_expansion: Maximum number of expanded chunks.

        Returns:
            List of expanded chunk dictionaries.
        """
        if not chunk_ids:
            return []

        if self.schema_type == "legacy":
            return self._expand_chunks_legacy(chunk_ids, max_expansion)
        else:
            return self._expand_chunks_graphrag(chunk_ids, max_expansion)

    def _expand_chunks_legacy(
        self,
        chunk_ids: list[str],
        max_expansion: int = 20,
    ) -> list[dict]:
        """Expand chunks using legacy schema (Section-based).

        Legacy hierarchy: Document -> Page -> Section -> Chunk
        """
        sibling_section_query = """
        MATCH (c:Chunk)
        WHERE c.id IN $chunk_ids

        // Get siblings (chunks before and after in sequence)
        OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
        OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)

        // Get other chunks in the same section
        OPTIONAL MATCH (c)<-[:CONTAINS]-(section:Section)-[:CONTAINS]->(sibling:Chunk)
        WHERE sibling.id <> c.id

        // Get document info via chunk's page
        MATCH (c)-[:ON_PAGE]->(page:Page)<-[:HAS_PAGE]-(doc:Document)

        WITH collect(DISTINCT next) + collect(DISTINCT prev) + collect(DISTINCT sibling) AS expanded_chunks, doc, page
        UNWIND expanded_chunks AS exp
        WHERE exp IS NOT NULL

        RETURN DISTINCT exp.id AS chunk_id,
               exp.text AS text,
               0.6 AS score,
               doc.id AS document_id,
               doc.full_path AS document_path,
               doc.filename AS document_name,
               page.page_number AS page,
               'sibling_section' AS expansion_type
        LIMIT $max_expansion
        """

        entity_cooccurrence_query = """
        MATCH (c:Chunk)
        WHERE c.id IN $chunk_ids
        MATCH (c)-[:MENTIONS]->(e:Entity)
        MATCH (other:Chunk)-[:MENTIONS]->(e)
        WHERE NOT other.id IN $chunk_ids
        MATCH (other)<-[:CONTAINS]-(section:Section)
        MATCH (other)-[:ON_PAGE]->(page:Page)<-[:HAS_PAGE]-(doc:Document)

        WITH other, doc, page, count(DISTINCT e) AS shared_entities
        ORDER BY shared_entities DESC

        RETURN DISTINCT other.id AS chunk_id,
               other.text AS text,
               0.4 + (shared_entities * 0.1) AS score,
               doc.id AS document_id,
               doc.full_path AS document_path,
               doc.filename AS document_name,
               page.page_number AS page,
               'entity_cooccurrence' AS expansion_type
        LIMIT $max_expansion
        """

        expanded = []

        sibling_results = self.neo4j.execute_read(
            sibling_section_query, {"chunk_ids": chunk_ids, "max_expansion": max_expansion // 2}
        )
        entity_results = self.neo4j.execute_read(
            entity_cooccurrence_query, {"chunk_ids": chunk_ids, "max_expansion": max_expansion // 2}
        )

        seen_ids = set(chunk_ids)
        for result in sibling_results + entity_results:
            if result["chunk_id"] not in seen_ids:
                seen_ids.add(result["chunk_id"])
                expanded.append(result)

        expanded.sort(key=lambda x: x["score"], reverse=True)
        return expanded[:max_expansion]

    def _expand_chunks_graphrag(
        self,
        chunk_ids: list[str],
        max_expansion: int = 20,
    ) -> list[dict]:
        """Expand chunks using GraphRAG schema (metadata-based).

        GraphRAG structure: Document <- FROM_DOCUMENT - Chunk (with metadata)
        Uses section_path property for same-section expansion.
        """
        # Query for siblings via NEXT relationships and same section_path
        sibling_section_query = """
        MATCH (c:Chunk)
        WHERE c.id IN $chunk_ids

        // Get siblings (chunks before and after in sequence)
        OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
        OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)

        // Get other chunks with same section_path in same document
        OPTIONAL MATCH (c)-[:FROM_DOCUMENT]->(doc:Document)<-[:FROM_DOCUMENT]-(sibling:Chunk)
        WHERE sibling.id <> c.id
          AND sibling.section_path = c.section_path
          AND c.section_path IS NOT NULL

        WITH collect(DISTINCT next) + collect(DISTINCT prev) + collect(DISTINCT sibling) AS expanded_chunks
        UNWIND expanded_chunks AS exp
        WHERE exp IS NOT NULL
        MATCH (exp)-[:FROM_DOCUMENT]->(doc:Document)

        RETURN DISTINCT exp.id AS chunk_id,
               exp.text AS text,
               0.6 AS score,
               doc.id AS document_id,
               doc.full_path AS document_path,
               doc.filename AS document_name,
               exp.page_number AS page,
               'sibling_section' AS expansion_type
        LIMIT $max_expansion
        """

        # Query for chunks mentioning the same entities
        entity_cooccurrence_query = """
        MATCH (c:Chunk)
        WHERE c.id IN $chunk_ids
        MATCH (c)-[:MENTIONS]->(e:Entity)
        MATCH (other:Chunk)-[:MENTIONS]->(e)
        WHERE NOT other.id IN $chunk_ids
        MATCH (other)-[:FROM_DOCUMENT]->(doc:Document)

        WITH other, doc, count(DISTINCT e) AS shared_entities
        ORDER BY shared_entities DESC

        RETURN DISTINCT other.id AS chunk_id,
               other.text AS text,
               0.4 + (shared_entities * 0.1) AS score,
               doc.id AS document_id,
               doc.full_path AS document_path,
               doc.filename AS document_name,
               other.page_number AS page,
               'entity_cooccurrence' AS expansion_type
        LIMIT $max_expansion
        """

        expanded = []

        sibling_results = self.neo4j.execute_read(
            sibling_section_query, {"chunk_ids": chunk_ids, "max_expansion": max_expansion // 2}
        )
        entity_results = self.neo4j.execute_read(
            entity_cooccurrence_query, {"chunk_ids": chunk_ids, "max_expansion": max_expansion // 2}
        )

        seen_ids = set(chunk_ids)
        for result in sibling_results + entity_results:
            if result["chunk_id"] not in seen_ids:
                seen_ids.add(result["chunk_id"])
                expanded.append(result)

        expanded.sort(key=lambda x: x["score"], reverse=True)
        return expanded[:max_expansion]

    def get_document_context(
        self,
        chunk_id: str,
        context_window: int = 2,
    ) -> dict:
        """Get context around a chunk including nearby chunks.

        Args:
            chunk_id: ID of the chunk.
            context_window: Number of chunks before/after to include.

        Returns:
            Dictionary with chunk and context information.
        """
        if not isinstance(context_window, int) or context_window < 1 or context_window > 10:
            context_window = 2

        if self.schema_type == "legacy":
            return self._get_document_context_legacy(chunk_id, context_window)
        else:
            return self._get_document_context_graphrag(chunk_id, context_window)

    def _get_document_context_legacy(
        self,
        chunk_id: str,
        context_window: int = 2,
    ) -> dict:
        """Get document context using legacy schema."""
        query = f"""
        MATCH (c:Chunk {{id: $chunk_id}})

        // Get N previous chunks
        OPTIONAL MATCH path_prev = (c)<-[:NEXT*1..{context_window}]-(prev:Chunk)

        // Get N next chunks
        OPTIONAL MATCH path_next = (c)-[:NEXT*1..{context_window}]->(next:Chunk)

        // Get parent section
        OPTIONAL MATCH (c)<-[:CONTAINS]-(section:Section)

        // Get document via chunk's page
        MATCH (c)-[:ON_PAGE]->(page:Page)<-[:HAS_PAGE]-(doc:Document)

        RETURN c.text AS text,
               page.page_number AS page,
               c.element_type AS element_type,
               collect(DISTINCT prev.text) AS prev_texts,
               collect(DISTINCT next.text) AS next_texts,
               section.title AS section_title,
               doc.filename AS document_name,
               doc.full_path AS document_path
        """

        results = self.neo4j.execute_read(query, {"chunk_id": chunk_id})

        if not results:
            return {}

        result = results[0]
        return {
            "chunk_id": chunk_id,
            "text": result["text"],
            "page": result.get("page"),
            "element_type": result.get("element_type"),
            "previous_context": result.get("prev_texts", []),
            "next_context": result.get("next_texts", []),
            "section_title": result.get("section_title"),
            "document_name": result["document_name"],
            "document_path": result["document_path"],
        }

    def _get_document_context_graphrag(
        self,
        chunk_id: str,
        context_window: int = 2,
    ) -> dict:
        """Get document context using GraphRAG schema."""
        query = f"""
        MATCH (c:Chunk {{id: $chunk_id}})

        // Get N previous chunks
        OPTIONAL MATCH path_prev = (c)<-[:NEXT*1..{context_window}]-(prev:Chunk)

        // Get N next chunks
        OPTIONAL MATCH path_next = (c)-[:NEXT*1..{context_window}]->(next:Chunk)

        // Get document
        MATCH (c)-[:FROM_DOCUMENT]->(doc:Document)

        RETURN c.text AS text,
               c.page_number AS page,
               c.element_type AS element_type,
               c.section_title AS section_title,
               collect(DISTINCT prev.text) AS prev_texts,
               collect(DISTINCT next.text) AS next_texts,
               doc.filename AS document_name,
               doc.full_path AS document_path
        """

        results = self.neo4j.execute_read(query, {"chunk_id": chunk_id})

        if not results:
            return {}

        result = results[0]
        return {
            "chunk_id": chunk_id,
            "text": result["text"],
            "page": result.get("page"),
            "element_type": result.get("element_type"),
            "previous_context": result.get("prev_texts", []),
            "next_context": result.get("next_texts", []),
            "section_title": result.get("section_title"),
            "document_name": result["document_name"],
            "document_path": result["document_path"],
        }

    def get_section_chunks(
        self,
        chunk_id: str,
    ) -> list[dict]:
        """Get all chunks in the same section as the given chunk.

        Args:
            chunk_id: ID of the chunk.

        Returns:
            List of chunks in the same section.
        """
        if self.schema_type == "legacy":
            return self._get_section_chunks_legacy(chunk_id)
        else:
            return self._get_section_chunks_graphrag(chunk_id)

    def _get_section_chunks_legacy(self, chunk_id: str) -> list[dict]:
        """Get section chunks using legacy schema."""
        query = """
        MATCH (c:Chunk {id: $chunk_id})

        // Find the parent section
        MATCH (c)<-[:CONTAINS]-(section:Section)

        // Get all chunks under this section
        MATCH (section)-[:CONTAINS]->(sibling:Chunk)
        MATCH (sibling)-[:ON_PAGE]->(page:Page)<-[:HAS_PAGE]-(doc:Document)

        RETURN sibling.id AS chunk_id,
               sibling.text AS text,
               sibling.sequence_order AS sequence_order,
               sibling.element_type AS element_type,
               doc.id AS document_id,
               section.title AS section_title,
               page.page_number AS page
        ORDER BY sibling.sequence_order
        """

        results = self.neo4j.execute_read(query, {"chunk_id": chunk_id})

        return [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "sequence_order": r["sequence_order"],
                "element_type": r["element_type"],
                "document_id": r["document_id"],
                "section_title": r.get("section_title"),
                "page": r.get("page"),
            }
            for r in results
        ]

    def _get_section_chunks_graphrag(self, chunk_id: str) -> list[dict]:
        """Get section chunks using GraphRAG schema (metadata-based)."""
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        WHERE c.section_path IS NOT NULL

        // Find chunks with same section_path in same document
        MATCH (c)-[:FROM_DOCUMENT]->(doc:Document)<-[:FROM_DOCUMENT]-(sibling:Chunk)
        WHERE sibling.section_path = c.section_path

        RETURN sibling.id AS chunk_id,
               sibling.text AS text,
               sibling.index AS sequence_order,
               sibling.element_type AS element_type,
               doc.id AS document_id,
               sibling.section_title AS section_title,
               sibling.page_number AS page
        ORDER BY sibling.index
        """

        results = self.neo4j.execute_read(query, {"chunk_id": chunk_id})

        return [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "sequence_order": r.get("sequence_order", 0),
                "element_type": r.get("element_type"),
                "document_id": r["document_id"],
                "section_title": r.get("section_title"),
                "page": r.get("page"),
            }
            for r in results
        ]

    def get_related_chunks_by_entity(
        self,
        chunk_id: str,
        max_results: int = 10,
    ) -> list[dict]:
        """Get chunks related through shared entity mentions.

        Args:
            chunk_id: ID of the source chunk.
            max_results: Maximum results to return.

        Returns:
            List of related chunks with entity information.
        """
        if self.schema_type == "legacy":
            return self._get_related_chunks_by_entity_legacy(chunk_id, max_results)
        else:
            return self._get_related_chunks_by_entity_graphrag(chunk_id, max_results)

    def _get_related_chunks_by_entity_legacy(
        self,
        chunk_id: str,
        max_results: int = 10,
    ) -> list[dict]:
        """Get related chunks using legacy schema."""
        query = """
        MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e:Entity)
        MATCH (other:Chunk)-[:MENTIONS]->(e)
        WHERE other.id <> $chunk_id
        MATCH (other)<-[:CONTAINS]-(section:Section)
        MATCH (other)-[:ON_PAGE]->(page:Page)<-[:HAS_PAGE]-(doc:Document)

        WITH other, doc, section, page, collect(DISTINCT e.name) AS shared_entities
        ORDER BY size(shared_entities) DESC

        RETURN other.id AS chunk_id,
               other.text AS text,
               shared_entities,
               doc.filename AS document_name,
               doc.full_path AS document_path,
               page.page_number AS page,
               section.title AS section_title
        LIMIT $max_results
        """

        results = self.neo4j.execute_read(
            query, {"chunk_id": chunk_id, "max_results": max_results}
        )

        return [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "shared_entities": r["shared_entities"],
                "document_name": r["document_name"],
                "document_path": r["document_path"],
                "page": r.get("page"),
                "section_title": r.get("section_title"),
            }
            for r in results
        ]

    def _get_related_chunks_by_entity_graphrag(
        self,
        chunk_id: str,
        max_results: int = 10,
    ) -> list[dict]:
        """Get related chunks using GraphRAG schema."""
        query = """
        MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e:Entity)
        MATCH (other:Chunk)-[:MENTIONS]->(e)
        WHERE other.id <> $chunk_id
        MATCH (other)-[:FROM_DOCUMENT]->(doc:Document)

        WITH other, doc, collect(DISTINCT e.name) AS shared_entities
        ORDER BY size(shared_entities) DESC

        RETURN other.id AS chunk_id,
               other.text AS text,
               shared_entities,
               doc.filename AS document_name,
               doc.full_path AS document_path,
               other.page_number AS page,
               other.section_title AS section_title
        LIMIT $max_results
        """

        results = self.neo4j.execute_read(
            query, {"chunk_id": chunk_id, "max_results": max_results}
        )

        return [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "shared_entities": r["shared_entities"],
                "document_name": r["document_name"],
                "document_path": r["document_path"],
                "page": r.get("page"),
                "section_title": r.get("section_title"),
            }
            for r in results
        ]
