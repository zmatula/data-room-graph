"""Cross-type entity resolution for Neo4j GraphRAG.

This module provides custom entity resolution that works across entity types,
preventing the same entity from being created with different types (e.g.,
"EnCap Investments LP" classified as both Fund and Manager).

The built-in FuzzyMatchResolver in neo4j-graphrag only matches within the
same label, so this custom resolver handles cross-type deduplication.
"""

import logging
import re
from typing import Optional
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..database.neo4j_client import Neo4jClient, get_neo4j_client
from .pe_schema import get_entity_labels

logger = logging.getLogger(__name__)


# Legal suffixes to normalize for comparison
LEGAL_SUFFIXES = [
    r"\bLLC\b",
    r"\bL\.L\.C\.\b",
    r"\bLLP\b",
    r"\bL\.L\.P\.\b",
    r"\bLP\b",
    r"\bL\.P\.\b",
    r"\bInc\.?\b",
    r"\bIncorporated\b",
    r"\bCorp\.?\b",
    r"\bCorporation\b",
    r"\bLtd\.?\b",
    r"\bLimited\b",
    r"\bPLC\b",
    r"\bP\.L\.C\.\b",
    r"\bGmbH\b",
    r"\bS\.A\.\b",
    r"\bS\.A\.R\.L\.\b",
]


@dataclass
class CrossTypeMatch:
    """Result of a cross-type matching operation."""

    canonical_id: str
    canonical_name: str
    canonical_label: str
    source_id: str
    source_name: str
    source_label: str
    similarity: float


class CrossTypeEntityResolver:
    """Resolves entities across different types to prevent duplicates.

    This resolver runs AFTER neo4j-graphrag's standard FuzzyMatchResolver
    to handle cases where the same entity was classified with different types.

    For example:
    - "EnCap Investments LP" might be extracted as both Fund and Manager
    - "Permian Basin Exploration" might be both PortfolioCompany and Location
    """

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        similarity_threshold: float = 0.85,
    ):
        """Initialize the resolver.

        Args:
            neo4j_client: Neo4j client.
            similarity_threshold: Minimum similarity for matching (0-1).
        """
        self.neo4j = neo4j_client or get_neo4j_client()
        self.similarity_threshold = similarity_threshold

        # Compile suffix pattern
        self.suffix_pattern = re.compile(
            "|".join(LEGAL_SUFFIXES), re.IGNORECASE
        )

        # Entity labels from schema
        self.entity_labels = get_entity_labels()

    def normalize_name(self, name: str) -> str:
        """Normalize an entity name for comparison.

        Args:
            name: Raw entity name.

        Returns:
            Normalized name string.
        """
        # Remove legal suffixes
        normalized = self.suffix_pattern.sub("", name)

        # Remove punctuation and extra whitespace
        normalized = re.sub(r"[,\.\-\(\)]", " ", normalized)
        normalized = " ".join(normalized.split()).strip().lower()

        return normalized

    def compute_similarity(self, name1: str, name2: str) -> float:
        """Compute similarity between two names.

        Args:
            name1: First name.
            name2: Second name.

        Returns:
            Similarity score (0-1).
        """
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)

        return SequenceMatcher(None, norm1, norm2).ratio()

    async def resolve(self, dataroom_id: str) -> list[CrossTypeMatch]:
        """Run cross-type resolution for a data room.

        This method:
        1. Queries all entities in the data room
        2. Compares entities across different labels
        3. Merges duplicates by moving relationships to canonical entity

        Args:
            dataroom_id: Data room ID to resolve.

        Returns:
            List of cross-type matches that were resolved.
        """
        logger.info(f"Running cross-type resolution for data room {dataroom_id}")

        # Get all entities grouped by label
        entities_by_label = await self._get_entities_by_label(dataroom_id)

        if not entities_by_label:
            logger.info("No entities found for cross-type resolution")
            return []

        # Find cross-type matches
        matches = self._find_cross_type_matches(entities_by_label)

        if not matches:
            logger.info("No cross-type duplicates found")
            return []

        logger.info(f"Found {len(matches)} cross-type matches to resolve")

        # Merge duplicates
        for match in matches:
            await self._merge_entities(match)

        return matches

    async def _get_entities_by_label(
        self,
        dataroom_id: str,
    ) -> dict[str, list[dict]]:
        """Get all entities grouped by label.

        Args:
            dataroom_id: Data room ID.

        Returns:
            Dictionary mapping label to list of entity dicts.
        """
        result = {}

        for label in self.entity_labels:
            query = f"""
            MATCH (e:{label})
            WHERE e.dataroom_id = $dataroom_id
            RETURN e.id AS id,
                   e.name AS name,
                   e.canonical_name AS canonical_name,
                   labels(e) AS labels
            """

            entities = self.neo4j.execute_read(
                query,
                {"dataroom_id": dataroom_id},
            )

            if entities:
                result[label] = [
                    {
                        "id": e["id"],
                        "name": e["name"] or e.get("canonical_name", ""),
                        "canonical_name": e.get("canonical_name", e["name"]),
                        "labels": e["labels"],
                    }
                    for e in entities
                ]

        return result

    def _find_cross_type_matches(
        self,
        entities_by_label: dict[str, list[dict]],
    ) -> list[CrossTypeMatch]:
        """Find entities that match across different types.

        Args:
            entities_by_label: Entities grouped by label.

        Returns:
            List of cross-type matches.
        """
        matches = []
        labels = list(entities_by_label.keys())

        # Compare entities across all label pairs
        for i, label1 in enumerate(labels):
            for label2 in labels[i + 1:]:
                for entity1 in entities_by_label[label1]:
                    for entity2 in entities_by_label[label2]:
                        similarity = self.compute_similarity(
                            entity1["name"],
                            entity2["name"],
                        )

                        if similarity >= self.similarity_threshold:
                            # Determine canonical entity (prefer the one with more mentions)
                            # For now, keep the first alphabetically by label
                            if label1 < label2:
                                canonical = entity1
                                canonical_label = label1
                                source = entity2
                                source_label = label2
                            else:
                                canonical = entity2
                                canonical_label = label2
                                source = entity1
                                source_label = label1

                            matches.append(CrossTypeMatch(
                                canonical_id=canonical["id"],
                                canonical_name=canonical["name"],
                                canonical_label=canonical_label,
                                source_id=source["id"],
                                source_name=source["name"],
                                source_label=source_label,
                                similarity=similarity,
                            ))

        # Sort by similarity descending
        matches.sort(key=lambda m: m.similarity, reverse=True)

        # Remove transitive matches (if A matches B and B matches C, only keep A->B)
        resolved_ids = set()
        filtered_matches = []

        for match in matches:
            if match.source_id not in resolved_ids:
                filtered_matches.append(match)
                resolved_ids.add(match.source_id)

        return filtered_matches

    async def _merge_entities(self, match: CrossTypeMatch):
        """Merge a duplicate entity into its canonical form.

        Args:
            match: Cross-type match to merge.
        """
        logger.info(
            f"Merging '{match.source_name}' ({match.source_label}) "
            f"into '{match.canonical_name}' ({match.canonical_label})"
        )

        # Move all MENTIONS relationships to canonical entity
        move_mentions_query = """
        MATCH (source {id: $source_id})<-[r:MENTIONS]-(chunk:Chunk)
        MATCH (canonical {id: $canonical_id})
        CREATE (chunk)-[:MENTIONS {
            offset_start: r.offset_start,
            offset_end: r.offset_end,
            confidence: r.confidence,
            snippet: r.snippet
        }]->(canonical)
        DELETE r
        """

        # Add source name as alias
        add_alias_query = """
        MATCH (canonical {id: $canonical_id})
        SET canonical.aliases = CASE
            WHEN canonical.aliases IS NULL THEN [$alias]
            WHEN NOT $alias IN canonical.aliases THEN canonical.aliases + $alias
            ELSE canonical.aliases
        END
        """

        # Update mention count
        update_count_query = """
        MATCH (canonical {id: $canonical_id})<-[:MENTIONS]-(chunk:Chunk)
        WITH canonical, count(chunk) AS mention_count
        SET canonical.mention_count = mention_count
        """

        # Delete source entity
        delete_source_query = """
        MATCH (source {id: $source_id})
        DETACH DELETE source
        """

        try:
            self.neo4j.execute_batch([
                (move_mentions_query, {
                    "source_id": match.source_id,
                    "canonical_id": match.canonical_id,
                }),
                (add_alias_query, {
                    "canonical_id": match.canonical_id,
                    "alias": match.source_name,
                }),
                (update_count_query, {
                    "canonical_id": match.canonical_id,
                }),
                (delete_source_query, {
                    "source_id": match.source_id,
                }),
            ])

            logger.debug(f"Successfully merged entity {match.source_id}")

        except Exception as e:
            logger.error(f"Failed to merge entity {match.source_id}: {e}")

    async def get_resolution_candidates(
        self,
        dataroom_id: str,
    ) -> list[CrossTypeMatch]:
        """Get potential cross-type matches without merging.

        Useful for review before committing resolution.

        Args:
            dataroom_id: Data room ID.

        Returns:
            List of potential cross-type matches.
        """
        entities_by_label = await self._get_entities_by_label(dataroom_id)
        return self._find_cross_type_matches(entities_by_label)


async def run_cross_type_resolution(
    dataroom_id: str,
    similarity_threshold: float = 0.85,
) -> list[CrossTypeMatch]:
    """Convenience function to run cross-type resolution.

    Args:
        dataroom_id: Data room ID.
        similarity_threshold: Minimum similarity for matching.

    Returns:
        List of resolved matches.
    """
    resolver = CrossTypeEntityResolver(similarity_threshold=similarity_threshold)
    return await resolver.resolve(dataroom_id)
