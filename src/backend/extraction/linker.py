"""Evidence linking between chunks and entities."""

import logging
from typing import Optional
from dataclasses import dataclass

from .entity_extractor import ExtractedEntity, EntityExtractor
from .canonicalizer import EntityCanonicalizer
from ..database.models import Entity, Evidence, Chunk
from ..database.neo4j_client import Neo4jClient, get_neo4j_client

logger = logging.getLogger(__name__)


@dataclass
class MentionLink:
    """A link between a chunk and an entity mention."""

    chunk_id: str
    entity_id: str
    offset_start: int
    offset_end: int
    confidence: float
    snippet: str


class EntityLinker:
    """Links extracted entities to chunks with evidence."""

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        extractor: Optional[EntityExtractor] = None,
        canonicalizer: Optional[EntityCanonicalizer] = None,
    ):
        """Initialize the linker.

        Args:
            neo4j_client: Neo4j client.
            extractor: Entity extractor.
            canonicalizer: Entity canonicalizer.
        """
        self.neo4j = neo4j_client or get_neo4j_client()
        self.extractor = extractor or EntityExtractor()
        self.canonicalizer = canonicalizer or EntityCanonicalizer(self.neo4j)

    def process_chunk(
        self,
        chunk: Chunk,
        dataroom_id: str,
    ) -> list[MentionLink]:
        """Extract entities from a chunk and create mention links.

        Args:
            chunk: Chunk to process.
            dataroom_id: Data room ID.

        Returns:
            List of created mention links.
        """
        # Extract entities from chunk text
        extracted = self.extractor.extract_entities(
            chunk.text,
            chunk.offset_start,
        )

        links = []
        for entity_data in extracted:
            # Create or find canonical entity
            entity = self.extractor.create_entity_node(entity_data, dataroom_id)
            canonical_id, is_new = self.canonicalizer.canonicalize_entity(
                entity, dataroom_id
            )

            if is_new:
                # Create new entity node
                self._create_entity_node(entity)
            else:
                # Update entity ID to canonical
                entity.id = canonical_id

            # Create mention relationship
            link = self._create_mention_link(chunk, entity_data, canonical_id)
            links.append(link)

        return links

    def _create_entity_node(self, entity: Entity):
        """Create an entity node in Neo4j.

        Args:
            entity: Entity to create.
        """
        # Determine labels based on entity type
        labels = ["Entity", entity.entity_type.value]

        self.neo4j.create_node(
            labels=labels,
            properties=entity.to_neo4j_properties(),
        )

        logger.debug(f"Created entity: {entity.canonical_name} ({entity.entity_type})")

    def _create_mention_link(
        self,
        chunk: Chunk,
        extracted: ExtractedEntity,
        entity_id: str,
    ) -> MentionLink:
        """Create a MENTIONS relationship between chunk and entity.

        Args:
            chunk: Source chunk.
            extracted: Extracted entity data.
            entity_id: Target entity ID.

        Returns:
            Created MentionLink.
        """
        # Create snippet for evidence
        snippet = self._create_snippet(
            chunk.text,
            extracted.offset_start - chunk.offset_start,
            extracted.offset_end - chunk.offset_start,
        )

        # Create relationship
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (c)-[r:MENTIONS]->(e)
        SET r.offset_start = $offset_start,
            r.offset_end = $offset_end,
            r.confidence = $confidence,
            r.snippet = $snippet
        WITH e, count(*) as mention_count
        SET e.mention_count = mention_count
        """

        self.neo4j.execute_write(
            query,
            {
                "chunk_id": chunk.id,
                "entity_id": entity_id,
                "offset_start": extracted.offset_start,
                "offset_end": extracted.offset_end,
                "confidence": extracted.confidence,
                "snippet": snippet,
            },
        )

        return MentionLink(
            chunk_id=chunk.id,
            entity_id=entity_id,
            offset_start=extracted.offset_start,
            offset_end=extracted.offset_end,
            confidence=extracted.confidence,
            snippet=snippet,
        )

    def _create_snippet(
        self,
        text: str,
        start: int,
        end: int,
        context_chars: int = 100,
    ) -> str:
        """Create a snippet with context around the mention.

        Args:
            text: Full text.
            start: Start offset in text.
            end: End offset in text.
            context_chars: Characters of context on each side.

        Returns:
            Snippet with context.
        """
        # Calculate snippet boundaries
        snippet_start = max(0, start - context_chars)
        snippet_end = min(len(text), end + context_chars)

        # Adjust to word boundaries
        if snippet_start > 0:
            space_idx = text.rfind(" ", snippet_start - 20, snippet_start)
            if space_idx > 0:
                snippet_start = space_idx + 1

        if snippet_end < len(text):
            space_idx = text.find(" ", snippet_end, snippet_end + 20)
            if space_idx > 0:
                snippet_end = space_idx

        snippet = text[snippet_start:snippet_end]

        # Add ellipsis if truncated
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(text):
            snippet = snippet + "..."

        return snippet

    def process_document_chunks(
        self,
        document_id: str,
        dataroom_id: str,
    ) -> int:
        """Process all chunks in a document for entity extraction.

        Args:
            document_id: Document ID.
            dataroom_id: Data room ID.

        Returns:
            Number of entities linked.
        """
        # Get all chunks for document
        # Hierarchy: Document -> Page -> Section -> Chunk
        query = """
        MATCH (d:Document {id: $document_id})-[:HAS_PAGE]->(p:Page)-[:HAS_SECTION]->(s:Section)-[:CONTAINS]->(c:Chunk)
        RETURN DISTINCT c.id as id, c.text as text, c.offset_start as offset_start, c.offset_end as offset_end, c.sequence_order as seq
        ORDER BY seq
        """

        chunks_data = self.neo4j.execute_read(query, {"document_id": document_id})

        total_links = 0
        for chunk_data in chunks_data:
            offset_start = chunk_data.get("offset_start", 0)
            # Use stored offset_end if available, otherwise calculate from text length
            offset_end = chunk_data.get("offset_end") or (offset_start + len(chunk_data["text"]))

            # Create a minimal chunk object
            chunk = Chunk(
                id=chunk_data["id"],
                dataroom_id=dataroom_id,
                document_id=document_id,
                text=chunk_data["text"],
                offset_start=offset_start,
                offset_end=offset_end,
            )

            links = self.process_chunk(chunk, dataroom_id)
            total_links += len(links)

        logger.info(
            f"Processed document {document_id}: {total_links} entity mentions"
        )
        return total_links

    def get_entity_evidence(
        self,
        entity_id: str,
    ) -> list[Evidence]:
        """Get all evidence (mentions) for an entity.

        Args:
            entity_id: Entity ID.

        Returns:
            List of Evidence objects.
        """
        # Hierarchy: Document -> Page -> Section -> Chunk
        query = """
        MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity {id: $entity_id})
        MATCH (c)-[:ON_PAGE]->(p:Page)<-[:HAS_PAGE]-(d:Document)
        RETURN c.id as chunk_id,
               d.id as document_id,
               p.page_number as page_start,
               p.page_number as page_end,
               r.offset_start as offset_start,
               r.offset_end as offset_end,
               c.coordinates as coordinates,
               r.snippet as snippet,
               r.confidence as confidence
        """

        results = self.neo4j.execute_read(query, {"entity_id": entity_id})

        evidence = []
        for record in results:
            evidence.append(
                Evidence(
                    chunk_id=record["chunk_id"],
                    document_id=record["document_id"],
                    page_start=record.get("page_start"),
                    page_end=record.get("page_end"),
                    offset_start=record["offset_start"],
                    offset_end=record["offset_end"],
                    coordinates=record.get("coordinates"),
                    snippet=record["snippet"],
                    confidence=record["confidence"],
                )
            )

        return evidence

    def create_entity_relationships(
        self,
        dataroom_id: str,
    ):
        """Create relationships between entities based on co-occurrence.

        Args:
            dataroom_id: Data room ID.
        """
        # Create AFFILIATED_WITH between managers and funds
        manager_fund_query = """
        MATCH (m:Manager {dataroom_id: $dataroom_id})<-[:MENTIONS]-(c:Chunk)
        MATCH (c)-[:MENTIONS]->(f:Fund {dataroom_id: $dataroom_id})
        WITH m, f, count(c) as co_occurrences
        WHERE co_occurrences >= 2
        MERGE (f)-[r:MANAGED_BY]->(m)
        SET r.co_occurrences = co_occurrences
        """

        # Create HAS_SERVICE_PROVIDER relationships
        service_provider_query = """
        MATCH (f:Fund {dataroom_id: $dataroom_id})<-[:MENTIONS]-(c:Chunk)
        MATCH (c)-[:MENTIONS]->(sp:ServiceProvider {dataroom_id: $dataroom_id})
        WITH f, sp, count(c) as co_occurrences
        WHERE co_occurrences >= 1
        MERGE (f)-[r:HAS_SERVICE_PROVIDER]->(sp)
        SET r.co_occurrences = co_occurrences
        """

        # Create HAS_ROLE relationships for persons
        person_role_query = """
        MATCH (p:Person {dataroom_id: $dataroom_id})<-[:MENTIONS]-(c:Chunk)
        MATCH (c)-[:MENTIONS]->(m:Manager {dataroom_id: $dataroom_id})
        WITH p, m, count(c) as co_occurrences
        WHERE co_occurrences >= 1
        MERGE (p)-[r:HAS_ROLE]->(m)
        SET r.co_occurrences = co_occurrences
        """

        self.neo4j.execute_batch([
            (manager_fund_query, {"dataroom_id": dataroom_id}),
            (service_provider_query, {"dataroom_id": dataroom_id}),
            (person_role_query, {"dataroom_id": dataroom_id}),
        ])

        logger.info(f"Created entity relationships for data room {dataroom_id}")
