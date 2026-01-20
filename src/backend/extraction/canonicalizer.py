"""Entity canonicalization and deduplication."""

import logging
import re
from typing import Optional
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..database.models import EntityType, Entity
from ..database.neo4j_client import Neo4jClient, get_neo4j_client

logger = logging.getLogger(__name__)


# Legal suffixes to normalize
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

# Patterns to remove for comparison
CLEANUP_PATTERNS = [
    r"\bThe\b",
    r"\bA\b",
    r"\bAn\b",
    r"[,\.\-\(\)]",
    r"\s+",
]


@dataclass
class CanonicalMatch:
    """Result of a canonical matching operation."""

    canonical_id: str
    canonical_name: str
    similarity: float
    is_new: bool
    existing_type: Optional[EntityType] = None  # Type of existing entity if cross-type match


class EntityCanonicalizer:
    """Normalizes and deduplicates entity names."""

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        similarity_threshold: float = 0.85,
    ):
        """Initialize the canonicalizer.

        Args:
            neo4j_client: Neo4j client for checking existing entities.
            similarity_threshold: Minimum similarity for matching.
        """
        self.neo4j = neo4j_client or get_neo4j_client()
        self.similarity_threshold = similarity_threshold

        # Compile regex patterns
        self.suffix_pattern = re.compile(
            "|".join(LEGAL_SUFFIXES), re.IGNORECASE
        )
        self.cleanup_patterns = [
            re.compile(p, re.IGNORECASE) for p in CLEANUP_PATTERNS
        ]

    def normalize_name(self, name: str) -> str:
        """Normalize an entity name for comparison.

        Args:
            name: Raw entity name.

        Returns:
            Normalized name for matching.
        """
        # Remove legal suffixes
        normalized = self.suffix_pattern.sub("", name)

        # Apply cleanup patterns
        for pattern in self.cleanup_patterns:
            normalized = pattern.sub(" ", normalized)

        # Collapse whitespace and trim
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

        # Use SequenceMatcher for fuzzy matching
        return SequenceMatcher(None, norm1, norm2).ratio()

    def find_canonical_match(
        self,
        name: str,
        entity_type: EntityType,
        dataroom_id: str,
    ) -> Optional[CanonicalMatch]:
        """Find an existing canonical entity that matches this name.

        Args:
            name: Entity name to match.
            entity_type: Type of entity.
            dataroom_id: Data room to search in.

        Returns:
            CanonicalMatch if found, None otherwise.
        """
        # Get existing entities of this type
        query = """
        MATCH (e:Entity {dataroom_id: $dataroom_id, entity_type: $entity_type})
        RETURN e.id as id, e.canonical_name as canonical_name, e.aliases as aliases
        """
        results = self.neo4j.execute_read(
            query,
            {
                "dataroom_id": dataroom_id,
                "entity_type": entity_type.value,
            },
        )

        best_match: Optional[CanonicalMatch] = None
        best_similarity = 0.0

        for record in results:
            canonical_name = record["canonical_name"]
            aliases = record.get("aliases", []) or []

            # Check canonical name
            similarity = self.compute_similarity(name, canonical_name)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = CanonicalMatch(
                    canonical_id=record["id"],
                    canonical_name=canonical_name,
                    similarity=similarity,
                    is_new=False,
                )

            # Check aliases
            for alias in aliases:
                similarity = self.compute_similarity(name, alias)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = CanonicalMatch(
                        canonical_id=record["id"],
                        canonical_name=canonical_name,
                        similarity=similarity,
                        is_new=False,
                    )

        # Return match only if above threshold
        if best_match and best_match.similarity >= self.similarity_threshold:
            return best_match

        return None

    def find_cross_type_match(
        self,
        name: str,
        exclude_type: EntityType,
        dataroom_id: str,
    ) -> Optional[CanonicalMatch]:
        """Find an existing entity that matches this name across ALL entity types.

        This prevents duplicate entities when the same entity is classified
        differently (e.g., "EnCap Investments LP" as both Fund and Manager).

        Args:
            name: Entity name to match.
            exclude_type: Entity type to exclude (already checked by find_canonical_match).
            dataroom_id: Data room to search in.

        Returns:
            CanonicalMatch if found, None otherwise.
        """
        # Get existing entities of ALL types except the one already checked
        query = """
        MATCH (e:Entity {dataroom_id: $dataroom_id})
        WHERE e.entity_type <> $exclude_type
        RETURN e.id as id, e.canonical_name as canonical_name, e.aliases as aliases, e.entity_type as entity_type
        """
        results = self.neo4j.execute_read(
            query,
            {
                "dataroom_id": dataroom_id,
                "exclude_type": exclude_type.value,
            },
        )

        best_match: Optional[CanonicalMatch] = None
        best_similarity = 0.0

        for record in results:
            canonical_name = record["canonical_name"]
            aliases = record.get("aliases", []) or []

            # Check canonical name
            similarity = self.compute_similarity(name, canonical_name)
            if similarity > best_similarity:
                best_similarity = similarity
                # Map entity_type string back to EntityType enum
                existing_type = None
                try:
                    existing_type = EntityType(record["entity_type"])
                except ValueError:
                    pass
                best_match = CanonicalMatch(
                    canonical_id=record["id"],
                    canonical_name=canonical_name,
                    similarity=similarity,
                    is_new=False,
                    existing_type=existing_type,
                )

            # Check aliases
            for alias in aliases:
                similarity = self.compute_similarity(name, alias)
                if similarity > best_similarity:
                    best_similarity = similarity
                    existing_type = None
                    try:
                        existing_type = EntityType(record["entity_type"])
                    except ValueError:
                        pass
                    best_match = CanonicalMatch(
                        canonical_id=record["id"],
                        canonical_name=canonical_name,
                        similarity=similarity,
                        is_new=False,
                        existing_type=existing_type,
                    )

        # Return match only if above threshold
        if best_match and best_match.similarity >= self.similarity_threshold:
            return best_match

        return None

    def canonicalize_entity(
        self,
        entity: Entity,
        dataroom_id: str,
    ) -> tuple[str, bool]:
        """Canonicalize an entity, finding or creating a canonical version.

        This method first checks for same-type matches (fast path), then
        checks for cross-type duplicates to prevent the same entity from
        being created with different types.

        Args:
            entity: Entity to canonicalize.
            dataroom_id: Data room ID.

        Returns:
            Tuple of (canonical_id, is_new).
        """
        # Fast path: try to find existing match of the same type
        match = self.find_canonical_match(
            entity.name,
            entity.entity_type,
            dataroom_id,
        )

        if match:
            # Update alias list if this is a new variation
            if entity.name.lower() != match.canonical_name.lower():
                self._add_alias(match.canonical_id, entity.name)
            return match.canonical_id, False

        # Check for cross-type duplicates (same entity with different type)
        cross_type_match = self.find_cross_type_match(
            entity.name,
            entity.entity_type,
            dataroom_id,
        )

        if cross_type_match:
            # Found existing entity with different type - link to it instead of creating duplicate
            logger.warning(
                f"Cross-type match: '{entity.name}' (new type: {entity.entity_type.value}) "
                f"matches existing entity '{cross_type_match.canonical_name}' "
                f"(type: {cross_type_match.existing_type.value if cross_type_match.existing_type else 'unknown'}). "
                f"Linking to existing entity."
            )
            # Update alias list
            if entity.name.lower() != cross_type_match.canonical_name.lower():
                self._add_alias(cross_type_match.canonical_id, entity.name)
            return cross_type_match.canonical_id, False

        # No match found - this is a new canonical entity
        entity.canonical_name = self._select_canonical_form(entity.name)
        return entity.id, True

    def _select_canonical_form(self, name: str) -> str:
        """Select the canonical form of a name.

        Keeps legal suffixes but normalizes spacing/punctuation.

        Args:
            name: Raw entity name.

        Returns:
            Canonical form of the name.
        """
        # Clean up whitespace
        canonical = " ".join(name.split())

        # Capitalize properly (title case, but keep acronyms)
        words = []
        for word in canonical.split():
            if word.isupper() and len(word) <= 4:
                # Keep acronyms uppercase
                words.append(word)
            elif word.lower() in ("llc", "llp", "lp", "inc", "corp", "ltd", "plc"):
                # Legal suffixes uppercase
                words.append(word.upper())
            else:
                words.append(word.title())

        return " ".join(words)

    def _add_alias(self, entity_id: str, alias: str):
        """Add an alias to an existing entity.

        Args:
            entity_id: Entity ID.
            alias: Alias to add.
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        SET e.aliases = CASE
            WHEN e.aliases IS NULL THEN [$alias]
            WHEN NOT $alias IN e.aliases THEN e.aliases + $alias
            ELSE e.aliases
        END
        """
        self.neo4j.execute_write(query, {"entity_id": entity_id, "alias": alias})

    def merge_entities(
        self,
        source_id: str,
        target_id: str,
    ) -> bool:
        """Merge two entities, moving all relationships to target.

        Args:
            source_id: Entity to merge from (will be deleted).
            target_id: Entity to merge into.

        Returns:
            True if merge was successful.
        """
        # Move all MENTIONS relationships
        move_mentions_query = """
        MATCH (source:Entity {id: $source_id})<-[r:MENTIONS]-(chunk:Chunk)
        MATCH (target:Entity {id: $target_id})
        CREATE (chunk)-[:MENTIONS {
            offset_start: r.offset_start,
            offset_end: r.offset_end,
            confidence: r.confidence
        }]->(target)
        DELETE r
        """

        # Merge aliases
        merge_aliases_query = """
        MATCH (source:Entity {id: $source_id})
        MATCH (target:Entity {id: $target_id})
        SET target.aliases = CASE
            WHEN target.aliases IS NULL THEN source.aliases + [source.name]
            WHEN source.aliases IS NULL THEN target.aliases + [source.name]
            ELSE target.aliases + source.aliases + [source.name]
        END
        """

        # Update mention count
        update_count_query = """
        MATCH (target:Entity {id: $target_id})<-[:MENTIONS]-(chunk:Chunk)
        WITH target, count(chunk) as mention_count
        SET target.mention_count = mention_count
        """

        # Delete source entity
        delete_query = """
        MATCH (source:Entity {id: $source_id})
        DETACH DELETE source
        """

        try:
            self.neo4j.execute_batch([
                (move_mentions_query, {"source_id": source_id, "target_id": target_id}),
                (merge_aliases_query, {"source_id": source_id, "target_id": target_id}),
                (update_count_query, {"target_id": target_id}),
                (delete_query, {"source_id": source_id}),
            ])
            logger.info(f"Merged entity {source_id} into {target_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to merge entities: {e}")
            return False

    def get_duplicate_candidates(
        self,
        dataroom_id: str,
        entity_type: Optional[EntityType] = None,
    ) -> list[tuple[str, str, float]]:
        """Find potential duplicate entities.

        Args:
            dataroom_id: Data room to search.
            entity_type: Optional filter by entity type.

        Returns:
            List of (entity1_id, entity2_id, similarity) tuples.
        """
        type_filter = ""
        if entity_type:
            type_filter = "AND e.entity_type = $entity_type"

        query = f"""
        MATCH (e:Entity {{dataroom_id: $dataroom_id}})
        {type_filter.replace('AND', 'WHERE') if type_filter else 'WHERE true'}
        RETURN e.id as id, e.name as name, e.canonical_name as canonical_name
        """

        params = {"dataroom_id": dataroom_id}
        if entity_type:
            params["entity_type"] = entity_type.value

        results = self.neo4j.execute_read(query, params)

        candidates = []
        entities = list(results)

        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                similarity = self.compute_similarity(
                    e1["canonical_name"], e2["canonical_name"]
                )
                if similarity >= 0.7:  # Lower threshold for candidates
                    candidates.append((e1["id"], e2["id"], similarity))

        # Sort by similarity descending
        return sorted(candidates, key=lambda x: x[2], reverse=True)
