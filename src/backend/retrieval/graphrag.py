"""GraphRAG hybrid retrieval implementation.

This module provides hybrid retrieval using vector similarity + graph traversal.
It supports both:
- Legacy structure: Document -> Page -> Section -> Chunk
- GraphRAG structure: Document -> Chunk (with metadata properties)

The retriever auto-detects which structure is in use and adjusts queries accordingly.
"""

import logging
from typing import Optional, Literal
from dataclasses import dataclass, field

from ..config import get_settings
from ..database.neo4j_client import Neo4jClient, get_neo4j_client
from ..ingestion.embedder import Embedder, get_embedder
from .hierarchy import HierarchyExpander
from .citations import Citation, CitationResolver

logger = logging.getLogger(__name__)


# Schema type for query adaptation
SchemaType = Literal["legacy", "graphrag", "auto"]


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    chunk_id: str
    text: str
    score: float
    document_id: str
    document_path: str
    document_name: str
    page: Optional[int] = None
    section_title: Optional[str] = None
    entities: list[str] = field(default_factory=list)
    parent_context: Optional[str] = None
    sibling_context: Optional[str] = None
    source: str = "vector"  # vector, graph, fulltext


@dataclass
class RetrievalResponse:
    """Response from a retrieval operation."""

    results: list[RetrievalResult]
    query: str
    total_results: int
    expanded_results: int = 0


class GraphRAGRetriever:
    """Hybrid retrieval using vector similarity + graph traversal.

    This retriever supports both legacy and GraphRAG schema structures:
    - Legacy: Document -> Page -> Section -> Chunk (with CONTAINS relationships)
    - GraphRAG: Document -> Chunk (with FROM_DOCUMENT relationship and metadata properties)

    The schema_type parameter controls which queries are used. Use "auto" to
    detect the schema automatically based on the presence of Section nodes.
    """

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        embedder: Optional[Embedder] = None,
        schema_type: SchemaType = "auto",
    ):
        """Initialize the retriever.

        Args:
            neo4j_client: Neo4j client.
            embedder: Embedding generator.
            schema_type: Schema type to use for queries:
                - "legacy": Use Section-based traversal
                - "graphrag": Use metadata-based queries
                - "auto": Auto-detect based on graph structure
        """
        self.neo4j = neo4j_client or get_neo4j_client()
        self.embedder = embedder or get_embedder()
        self.hierarchy_expander = HierarchyExpander(self.neo4j, schema_type=schema_type)
        self.citation_resolver = CitationResolver(self.neo4j)
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
        """Detect which schema is in use based on graph structure.

        Returns:
            Detected schema type.
        """
        # Check if Section nodes exist
        query = "MATCH (s:Section) RETURN count(s) > 0 AS has_sections LIMIT 1"
        result = self.neo4j.execute_read(query, {})

        if result and result[0].get("has_sections"):
            logger.debug("Auto-detected legacy schema (Section nodes present)")
            return "legacy"

        logger.debug("Auto-detected graphrag schema (no Section nodes)")
        return "graphrag"

    async def retrieve(
        self,
        query: str,
        dataroom_id: str,
        top_k: int = 10,
        expand_graph: bool = True,
        include_entities: bool = True,
        doc_type_filter: Optional[list[str]] = None,
    ) -> RetrievalResponse:
        """Perform hybrid retrieval for a query.

        Args:
            query: User query text.
            dataroom_id: Data room to search.
            top_k: Number of results to return.
            expand_graph: Whether to expand results via graph traversal.
            include_entities: Whether to include entity information.
            doc_type_filter: Optional list of document types to filter.

        Returns:
            RetrievalResponse with results.
        """
        logger.info(f"Retrieving for query: {query[:50]}...")

        # Stage 1: Vector search
        query_embedding = await self.embedder.embed_text(query)
        vector_results = self._vector_search(
            query_embedding, dataroom_id, top_k * 2, doc_type_filter
        )

        # Stage 2: Full-text search (for exact matches)
        fulltext_results = self._fulltext_search(
            query, dataroom_id, top_k, doc_type_filter
        )

        # Merge and deduplicate results
        all_results = self._merge_results(vector_results, fulltext_results)

        # Stage 3: Graph expansion
        expanded_count = 0
        if expand_graph and all_results:
            chunk_ids = [r.chunk_id for r in all_results[:5]]  # Expand top 5
            expanded = self.hierarchy_expander.expand_chunks(chunk_ids)
            expanded_count = len(expanded)
            all_results = self._incorporate_expansion(all_results, expanded)

        # Add entity information
        if include_entities:
            self._add_entity_info(all_results)

        # Re-rank and limit
        all_results = self._rerank(all_results, query)[:top_k]

        return RetrievalResponse(
            results=all_results,
            query=query,
            total_results=len(all_results),
            expanded_results=expanded_count,
        )

    def _vector_search(
        self,
        query_embedding: list[float],
        dataroom_id: str,
        top_k: int,
        doc_type_filter: Optional[list[str]] = None,
    ) -> list[RetrievalResult]:
        """Perform vector similarity search.

        Args:
            query_embedding: Query embedding vector.
            dataroom_id: Data room ID.
            top_k: Number of results.
            doc_type_filter: Optional document type filter.

        Returns:
            List of retrieval results.
        """
        if self.schema_type == "legacy":
            return self._vector_search_legacy(
                query_embedding, dataroom_id, top_k, doc_type_filter
            )
        else:
            return self._vector_search_graphrag(
                query_embedding, dataroom_id, top_k, doc_type_filter
            )

    def _vector_search_legacy(
        self,
        query_embedding: list[float],
        dataroom_id: str,
        top_k: int,
        doc_type_filter: Optional[list[str]] = None,
    ) -> list[RetrievalResult]:
        """Vector search using legacy schema (Section-based).

        Legacy hierarchy: Document -> Page -> Section -> Chunk
        """
        query = """
        CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding)
        YIELD node AS chunk, score
        WHERE chunk.dataroom_id = $dataroom_id
        MATCH (chunk)<-[:CONTAINS]-(section:Section)
        MATCH (chunk)-[:ON_PAGE]->(page:Page)<-[:HAS_PAGE]-(doc:Document)
        WHERE CASE WHEN $doc_types IS NOT NULL THEN doc.doc_type IN $doc_types ELSE true END
        RETURN chunk.id AS chunk_id,
               chunk.text AS text,
               score,
               doc.id AS document_id,
               doc.full_path AS document_path,
               doc.filename AS document_name,
               page.page_number AS page,
               section.title AS section_title
        ORDER BY score DESC
        LIMIT $top_k
        """

        results = self.neo4j.execute_read(
            query,
            {
                "embedding": query_embedding,
                "dataroom_id": dataroom_id,
                "top_k": top_k,
                "doc_types": doc_type_filter,
            },
        )

        return [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
                document_id=r["document_id"],
                document_path=r["document_path"],
                document_name=r["document_name"],
                page=r.get("page"),
                section_title=r.get("section_title"),
                source="vector",
            )
            for r in results
        ]

    def _vector_search_graphrag(
        self,
        query_embedding: list[float],
        dataroom_id: str,
        top_k: int,
        doc_type_filter: Optional[list[str]] = None,
    ) -> list[RetrievalResult]:
        """Vector search using GraphRAG schema (metadata-based).

        GraphRAG structure: Document <- FROM_DOCUMENT - Chunk (with metadata properties)
        Section info is stored as chunk.section_title and chunk.section_path properties.
        """
        query = """
        CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding)
        YIELD node AS chunk, score
        WHERE chunk.dataroom_id = $dataroom_id
        MATCH (chunk)-[:FROM_DOCUMENT]->(doc:Document)
        WHERE CASE WHEN $doc_types IS NOT NULL THEN doc.doc_type IN $doc_types ELSE true END
        RETURN chunk.id AS chunk_id,
               chunk.text AS text,
               score,
               doc.id AS document_id,
               doc.full_path AS document_path,
               doc.filename AS document_name,
               chunk.page_number AS page,
               chunk.section_title AS section_title,
               chunk.section_path AS section_path
        ORDER BY score DESC
        LIMIT $top_k
        """

        results = self.neo4j.execute_read(
            query,
            {
                "embedding": query_embedding,
                "dataroom_id": dataroom_id,
                "top_k": top_k,
                "doc_types": doc_type_filter,
            },
        )

        return [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
                document_id=r["document_id"],
                document_path=r["document_path"],
                document_name=r["document_name"],
                page=r.get("page"),
                section_title=r.get("section_title"),
                source="vector",
            )
            for r in results
        ]

    def _fulltext_search(
        self,
        query: str,
        dataroom_id: str,
        top_k: int,
        doc_type_filter: Optional[list[str]] = None,
    ) -> list[RetrievalResult]:
        """Perform full-text search for exact matches.

        Args:
            query: Query text.
            dataroom_id: Data room ID.
            top_k: Number of results.
            doc_type_filter: Optional document type filter.

        Returns:
            List of retrieval results.
        """
        if self.schema_type == "legacy":
            return self._fulltext_search_legacy(query, dataroom_id, top_k, doc_type_filter)
        else:
            return self._fulltext_search_graphrag(query, dataroom_id, top_k, doc_type_filter)

    def _fulltext_search_legacy(
        self,
        query: str,
        dataroom_id: str,
        top_k: int,
        doc_type_filter: Optional[list[str]] = None,
    ) -> list[RetrievalResult]:
        """Fulltext search using legacy schema (Section-based).

        Legacy hierarchy: Document -> Page -> Section -> Chunk
        """
        cypher_query = """
        CALL db.index.fulltext.queryNodes('chunk_content', $query)
        YIELD node AS chunk, score
        WHERE chunk.dataroom_id = $dataroom_id
        MATCH (chunk)<-[:CONTAINS]-(section:Section)
        MATCH (chunk)-[:ON_PAGE]->(page:Page)<-[:HAS_PAGE]-(doc:Document)
        WHERE CASE WHEN $doc_types IS NOT NULL THEN doc.doc_type IN $doc_types ELSE true END
        RETURN chunk.id AS chunk_id,
               chunk.text AS text,
               score * 0.8 AS score,
               doc.id AS document_id,
               doc.full_path AS document_path,
               doc.filename AS document_name,
               page.page_number AS page,
               section.title AS section_title
        ORDER BY score DESC
        LIMIT $top_k
        """

        try:
            results = self.neo4j.execute_read(
                cypher_query,
                {"query": query, "dataroom_id": dataroom_id, "top_k": top_k, "doc_types": doc_type_filter},
            )

            return [
                RetrievalResult(
                    chunk_id=r["chunk_id"],
                    text=r["text"],
                    score=r["score"],
                    document_id=r["document_id"],
                    document_path=r["document_path"],
                    document_name=r["document_name"],
                    page=r.get("page"),
                    section_title=r.get("section_title"),
                    source="fulltext",
                )
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Fulltext search (legacy) failed: {e}")
            return []

    def _fulltext_search_graphrag(
        self,
        query: str,
        dataroom_id: str,
        top_k: int,
        doc_type_filter: Optional[list[str]] = None,
    ) -> list[RetrievalResult]:
        """Fulltext search using GraphRAG schema (metadata-based).

        GraphRAG structure: Document <- FROM_DOCUMENT - Chunk
        """
        cypher_query = """
        CALL db.index.fulltext.queryNodes('chunk_content', $query)
        YIELD node AS chunk, score
        WHERE chunk.dataroom_id = $dataroom_id
        MATCH (chunk)-[:FROM_DOCUMENT]->(doc:Document)
        WHERE CASE WHEN $doc_types IS NOT NULL THEN doc.doc_type IN $doc_types ELSE true END
        RETURN chunk.id AS chunk_id,
               chunk.text AS text,
               score * 0.8 AS score,
               doc.id AS document_id,
               doc.full_path AS document_path,
               doc.filename AS document_name,
               chunk.page_number AS page,
               chunk.section_title AS section_title
        ORDER BY score DESC
        LIMIT $top_k
        """

        try:
            results = self.neo4j.execute_read(
                cypher_query,
                {"query": query, "dataroom_id": dataroom_id, "top_k": top_k, "doc_types": doc_type_filter},
            )

            return [
                RetrievalResult(
                    chunk_id=r["chunk_id"],
                    text=r["text"],
                    score=r["score"],
                    document_id=r["document_id"],
                    document_path=r["document_path"],
                    document_name=r["document_name"],
                    page=r.get("page"),
                    section_title=r.get("section_title"),
                    source="fulltext",
                )
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Fulltext search (graphrag) failed: {e}")
            return []

    def _merge_results(
        self,
        vector_results: list[RetrievalResult],
        fulltext_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Merge and deduplicate results from different sources.

        Args:
            vector_results: Results from vector search.
            fulltext_results: Results from fulltext search.

        Returns:
            Merged and deduplicated results.
        """
        seen_ids = set()
        merged = []

        # Add vector results first (higher priority)
        for result in vector_results:
            if result.chunk_id not in seen_ids:
                seen_ids.add(result.chunk_id)
                merged.append(result)

        # Add fulltext results that aren't duplicates
        for result in fulltext_results:
            if result.chunk_id not in seen_ids:
                seen_ids.add(result.chunk_id)
                merged.append(result)
            else:
                # Boost score for results found by both methods
                for merged_result in merged:
                    if merged_result.chunk_id == result.chunk_id:
                        merged_result.score = min(1.0, merged_result.score * 1.1)
                        break

        return merged

    def _incorporate_expansion(
        self,
        results: list[RetrievalResult],
        expanded: list[dict],
    ) -> list[RetrievalResult]:
        """Add expanded chunks to results.

        Args:
            results: Original results.
            expanded: Expanded chunks from graph traversal.

        Returns:
            Results with expansion incorporated.
        """
        seen_ids = {r.chunk_id for r in results}

        for exp in expanded:
            if exp["chunk_id"] not in seen_ids:
                seen_ids.add(exp["chunk_id"])
                results.append(
                    RetrievalResult(
                        chunk_id=exp["chunk_id"],
                        text=exp["text"],
                        score=exp.get("score", 0.5),  # Default score for expanded
                        document_id=exp["document_id"],
                        document_path=exp["document_path"],
                        document_name=exp["document_name"],
                        page=exp.get("page"),
                        source="graph",
                    )
                )

        return results

    def _add_entity_info(self, results: list[RetrievalResult]):
        """Add entity information to results.

        Args:
            results: Results to enhance.
        """
        if not results:
            return

        chunk_ids = [r.chunk_id for r in results]

        query = """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE c.id IN $chunk_ids
        RETURN c.id AS chunk_id, collect(DISTINCT e.name) AS entities
        """

        entity_results = self.neo4j.execute_read(query, {"chunk_ids": chunk_ids})
        entity_map = {r["chunk_id"]: r["entities"] for r in entity_results}

        for result in results:
            result.entities = entity_map.get(result.chunk_id, [])

    def _rerank(
        self,
        results: list[RetrievalResult],
        query: str,
    ) -> list[RetrievalResult]:
        """Re-rank results based on multiple signals.

        Args:
            results: Results to re-rank.
            query: Original query.

        Returns:
            Re-ranked results.
        """
        # For now, just sort by score
        # TODO: Add more sophisticated re-ranking (e.g., entity centrality, doc type)
        return sorted(results, key=lambda r: r.score, reverse=True)

    def get_citations(
        self, results: list[RetrievalResult]
    ) -> list[Citation]:
        """Generate citations for retrieval results.

        Args:
            results: Retrieval results.

        Returns:
            List of citations.
        """
        return [
            self.citation_resolver.resolve_citation(
                chunk_id=r.chunk_id,
                document_path=r.document_path,
                document_name=r.document_name,
                page=r.page,
                snippet=r.text[:200],
            )
            for r in results
        ]

    async def entity_search(
        self,
        entity_name: str,
        dataroom_id: str,
        entity_type: Optional[str] = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Search for chunks mentioning a specific entity.

        Args:
            entity_name: Entity name to search.
            dataroom_id: Data room ID.
            entity_type: Optional entity type filter.
            top_k: Number of results.

        Returns:
            List of chunks mentioning the entity.
        """
        if self.schema_type == "legacy":
            return await self._entity_search_legacy(
                entity_name, dataroom_id, entity_type, top_k
            )
        else:
            return await self._entity_search_graphrag(
                entity_name, dataroom_id, entity_type, top_k
            )

    async def _entity_search_legacy(
        self,
        entity_name: str,
        dataroom_id: str,
        entity_type: Optional[str] = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Entity search using legacy schema (Section-based)."""
        query = """
        MATCH (e:Entity {dataroom_id: $dataroom_id})
        WHERE (e.name CONTAINS $entity_name OR e.canonical_name CONTAINS $entity_name)
        AND CASE WHEN $entity_type IS NOT NULL THEN e.entity_type = $entity_type ELSE true END
        MATCH (c:Chunk)-[r:MENTIONS]->(e)
        MATCH (c)<-[:CONTAINS]-(section:Section)
        MATCH (c)-[:ON_PAGE]->(page:Page)<-[:HAS_PAGE]-(doc:Document)
        RETURN DISTINCT c.id AS chunk_id,
               c.text AS text,
               r.confidence AS score,
               doc.id AS document_id,
               doc.full_path AS document_path,
               doc.filename AS document_name,
               page.page_number AS page,
               section.title AS section_title,
               collect(DISTINCT e.name) AS entities
        ORDER BY score DESC
        LIMIT $top_k
        """

        results = self.neo4j.execute_read(
            query,
            {"entity_name": entity_name, "dataroom_id": dataroom_id, "top_k": top_k, "entity_type": entity_type},
        )

        return [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
                document_id=r["document_id"],
                document_path=r["document_path"],
                document_name=r["document_name"],
                page=r.get("page"),
                section_title=r.get("section_title"),
                entities=r["entities"],
                source="entity",
            )
            for r in results
        ]

    async def _entity_search_graphrag(
        self,
        entity_name: str,
        dataroom_id: str,
        entity_type: Optional[str] = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Entity search using GraphRAG schema (metadata-based)."""
        query = """
        MATCH (e:Entity {dataroom_id: $dataroom_id})
        WHERE (e.name CONTAINS $entity_name OR e.canonical_name CONTAINS $entity_name)
        AND CASE WHEN $entity_type IS NOT NULL THEN e.entity_type = $entity_type ELSE true END
        MATCH (c:Chunk)-[r:MENTIONS]->(e)
        MATCH (c)-[:FROM_DOCUMENT]->(doc:Document)
        RETURN DISTINCT c.id AS chunk_id,
               c.text AS text,
               r.confidence AS score,
               doc.id AS document_id,
               doc.full_path AS document_path,
               doc.filename AS document_name,
               c.page_number AS page,
               c.section_title AS section_title,
               collect(DISTINCT e.name) AS entities
        ORDER BY score DESC
        LIMIT $top_k
        """

        results = self.neo4j.execute_read(
            query,
            {"entity_name": entity_name, "dataroom_id": dataroom_id, "top_k": top_k, "entity_type": entity_type},
        )

        return [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
                document_id=r["document_id"],
                document_path=r["document_path"],
                document_name=r["document_name"],
                page=r.get("page"),
                section_title=r.get("section_title"),
                entities=r["entities"],
                source="entity",
            )
            for r in results
        ]
