"""Neo4j schema management - constraints, indexes, and initialization."""

import logging
from typing import Optional

from .neo4j_client import Neo4jClient, get_neo4j_client

logger = logging.getLogger(__name__)


# Node constraint definitions
NODE_CONSTRAINTS = [
    ("dataroom_id", "DataRoom", "id"),
    ("folder_id", "Folder", "id"),
    ("document_id", "Document", "id"),
    ("page_id", "Page", "id"),
    ("section_id", "Section", "id"),
    ("chunk_id", "Chunk", "id"),
    ("fund_id", "Fund", "id"),
    ("manager_id", "Manager", "id"),
    ("person_id", "Person", "id"),
    ("vehicle_id", "Vehicle", "id"),
    ("serviceprovider_id", "ServiceProvider", "id"),
    ("investor_id", "Investor", "id"),
]

# Vector index definitions: (name, label, property, dimensions, similarity)
VECTOR_INDEXES = [
    ("chunk_embedding", "Chunk", "embedding", 3072, "cosine"),
    ("entity_embedding", "Entity", "embedding", 3072, "cosine"),
]

# Full-text index definitions: (name, label, properties)
FULLTEXT_INDEXES = [
    ("chunk_content", "Chunk", ["text"]),
    ("document_content", "Document", ["filename", "doc_type"]),
    ("entity_name", "Entity", ["name", "canonical_name"]),
]


class SchemaManager:
    """Manages Neo4j schema: constraints, indexes, migrations."""

    def __init__(self, client: Optional[Neo4jClient] = None):
        """Initialize the schema manager.

        Args:
            client: Neo4j client instance. Uses global client if not provided.
        """
        self.client = client or get_neo4j_client()

    def create_constraints(self) -> list[str]:
        """Create all uniqueness constraints.

        Returns:
            List of created constraint names.
        """
        created = []
        for constraint_name, label, property_name in NODE_CONSTRAINTS:
            try:
                query = f"""
                CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE
                """
                self.client.execute_write(query)
                created.append(constraint_name)
                logger.info(f"Created constraint: {constraint_name}")
            except Exception as e:
                logger.warning(f"Could not create constraint {constraint_name}: {e}")
        return created

    def create_vector_indexes(self) -> list[str]:
        """Create vector indexes for similarity search.

        Returns:
            List of created index names.
        """
        created = []
        for index_name, label, property_name, dimensions, similarity in VECTOR_INDEXES:
            try:
                query = f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:{label}) ON (n.{property_name})
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {dimensions},
                        `vector.similarity_function`: '{similarity}'
                    }}
                }}
                """
                self.client.execute_write(query)
                created.append(index_name)
                logger.info(f"Created vector index: {index_name}")
            except Exception as e:
                logger.warning(f"Could not create vector index {index_name}: {e}")
        return created

    def create_fulltext_indexes(self) -> list[str]:
        """Create full-text indexes for hybrid search.

        Returns:
            List of created index names.
        """
        created = []
        for index_name, label, properties in FULLTEXT_INDEXES:
            try:
                props_str = ", ".join([f"n.{p}" for p in properties])
                query = f"""
                CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                FOR (n:{label}) ON EACH [{props_str}]
                """
                self.client.execute_write(query)
                created.append(index_name)
                logger.info(f"Created fulltext index: {index_name}")
            except Exception as e:
                logger.warning(f"Could not create fulltext index {index_name}: {e}")
        return created

    def create_property_indexes(self) -> list[str]:
        """Create property indexes for common queries.

        Returns:
            List of created index names.
        """
        indexes = [
            ("idx_chunk_doc", "Chunk", "document_id"),
            ("idx_chunk_section", "Chunk", "section_id"),
            ("idx_chunk_page", "Chunk", "page_id"),
            ("idx_page_doc", "Page", "document_id"),
            ("idx_section_doc", "Section", "document_id"),
            ("idx_folder_dataroom", "Folder", "dataroom_id"),
            ("idx_document_folder", "Document", "folder_id"),
            ("idx_document_type", "Document", "doc_type"),
        ]
        created = []
        for index_name, label, property_name in indexes:
            try:
                query = f"""
                CREATE INDEX {index_name} IF NOT EXISTS
                FOR (n:{label}) ON (n.{property_name})
                """
                self.client.execute_write(query)
                created.append(index_name)
                logger.info(f"Created property index: {index_name}")
            except Exception as e:
                logger.warning(f"Could not create property index {index_name}: {e}")
        return created

    def initialize_schema(self) -> dict[str, list[str]]:
        """Initialize all schema components.

        Returns:
            Dictionary with lists of created constraints and indexes.
        """
        logger.info("Initializing Neo4j schema...")
        results = {
            "constraints": self.create_constraints(),
            "vector_indexes": self.create_vector_indexes(),
            "fulltext_indexes": self.create_fulltext_indexes(),
            "property_indexes": self.create_property_indexes(),
        }
        logger.info(f"Schema initialization complete: {results}")
        return results

    def drop_all_constraints(self) -> int:
        """Drop all constraints (use with caution).

        Returns:
            Number of constraints dropped.
        """
        query = "SHOW CONSTRAINTS YIELD name RETURN name"
        constraints = self.client.execute_read(query)
        dropped = 0
        for record in constraints:
            try:
                drop_query = f"DROP CONSTRAINT {record['name']}"
                self.client.execute_write(drop_query)
                dropped += 1
            except Exception as e:
                logger.warning(f"Could not drop constraint {record['name']}: {e}")
        return dropped

    def drop_all_indexes(self) -> int:
        """Drop all indexes (use with caution).

        Returns:
            Number of indexes dropped.
        """
        query = "SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name"
        indexes = self.client.execute_read(query)
        dropped = 0
        for record in indexes:
            try:
                drop_query = f"DROP INDEX {record['name']}"
                self.client.execute_write(drop_query)
                dropped += 1
            except Exception as e:
                logger.warning(f"Could not drop index {record['name']}: {e}")
        return dropped

    def get_schema_info(self) -> dict[str, list[dict]]:
        """Get current schema information.

        Returns:
            Dictionary with constraints and indexes information.
        """
        constraints_query = "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties"
        indexes_query = "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state"

        return {
            "constraints": self.client.execute_read(constraints_query),
            "indexes": self.client.execute_read(indexes_query),
        }

    def verify_schema(self) -> dict[str, bool]:
        """Verify that all required schema elements exist.

        Returns:
            Dictionary mapping schema element names to existence status.
        """
        schema_info = self.get_schema_info()
        existing_constraints = {c["name"] for c in schema_info["constraints"]}
        existing_indexes = {i["name"] for i in schema_info["indexes"]}

        results = {}

        # Check constraints
        for constraint_name, _, _ in NODE_CONSTRAINTS:
            results[constraint_name] = constraint_name in existing_constraints

        # Check vector indexes
        for index_name, _, _, _, _ in VECTOR_INDEXES:
            results[index_name] = index_name in existing_indexes

        # Check fulltext indexes
        for index_name, _, _ in FULLTEXT_INDEXES:
            results[index_name] = index_name in existing_indexes

        return results
