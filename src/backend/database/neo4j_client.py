"""Neo4j database client with connection pooling and transaction management."""

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

from neo4j import GraphDatabase, Driver, Session, Transaction, Result
from neo4j.exceptions import ServiceUnavailable, AuthError

from ..config import get_settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j database client wrapper with connection management."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """Initialize the Neo4j client.

        Args:
            uri: Neo4j connection URI. Defaults to settings.
            user: Neo4j username. Defaults to settings.
            password: Neo4j password. Defaults to settings.
            database: Neo4j database name. Defaults to settings.
        """
        settings = get_settings()
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password
        self._database = database or settings.neo4j_database
        self._driver: Optional[Driver] = None

    @property
    def driver(self) -> Driver:
        """Get or create the Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_timeout=30,  # 30 second connection timeout
                connection_acquisition_timeout=60,  # 60 second pool acquisition timeout
            )
        return self._driver

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def verify_connectivity(self) -> bool:
        """Verify that we can connect to Neo4j.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            self.driver.verify_connectivity()
            return True
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Create a session context manager.

        Yields:
            A Neo4j session.
        """
        session = self.driver.session(database=self._database)
        try:
            yield session
        finally:
            session.close()

    def execute_read(
        self, query: str, parameters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Execute a read query and return results as dictionaries.

        Uses managed transactions for automatic retry on transient failures.

        Args:
            query: Cypher query string.
            parameters: Query parameters.

        Returns:
            List of result records as dictionaries.
        """

        def _read_tx(tx: Transaction) -> list[dict[str, Any]]:
            result = tx.run(query, parameters or {})
            return [record.data() for record in result]

        with self.session() as session:
            return session.execute_read(_read_tx)

    def execute_write(
        self, query: str, parameters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Execute a write query and return results.

        Args:
            query: Cypher query string.
            parameters: Query parameters.

        Returns:
            List of result records as dictionaries.
        """

        def _write_tx(tx: Transaction) -> list[dict[str, Any]]:
            result = tx.run(query, parameters or {})
            return [record.data() for record in result]

        with self.session() as session:
            return session.execute_write(_write_tx)

    def execute_batch(
        self, queries: list[tuple[str, dict[str, Any]]]
    ) -> list[list[dict[str, Any]]]:
        """Execute multiple queries in a single transaction.

        Args:
            queries: List of (query, parameters) tuples.

        Returns:
            List of results for each query.
        """

        def _batch_tx(tx: Transaction) -> list[list[dict[str, Any]]]:
            results = []
            for query, params in queries:
                result = tx.run(query, params)
                results.append([record.data() for record in result])
            return results

        with self.session() as session:
            return session.execute_write(_batch_tx)

    def create_node(
        self,
        labels: list[str],
        properties: dict[str, Any],
        merge: bool = True,
        id_property: str = "id",
    ) -> dict[str, Any]:
        """Create or merge a node.

        Args:
            labels: List of node labels.
            properties: Node properties.
            merge: If True, use MERGE instead of CREATE.
            id_property: Property to use for MERGE matching.

        Returns:
            The created/merged node as a dictionary.
        """
        labels_str = ":".join(labels)
        operation = "MERGE" if merge else "CREATE"

        if merge and id_property in properties:
            query = f"""
            {operation} (n:{labels_str} {{{id_property}: $id_value}})
            SET n += $properties
            RETURN n
            """
            params = {
                "id_value": properties[id_property],
                "properties": properties,
            }
        else:
            query = f"""
            {operation} (n:{labels_str} $properties)
            RETURN n
            """
            params = {"properties": properties}

        results = self.execute_write(query, params)
        return results[0]["n"] if results else {}

    def create_relationship(
        self,
        start_id: str,
        start_label: str,
        end_id: str,
        end_label: str,
        rel_type: str,
        properties: Optional[dict[str, Any]] = None,
        merge: bool = True,
    ) -> dict[str, Any]:
        """Create or merge a relationship between two nodes.

        Args:
            start_id: ID of the start node.
            start_label: Label of the start node.
            end_id: ID of the end node.
            end_label: Label of the end node.
            rel_type: Relationship type.
            properties: Relationship properties.
            merge: If True, use MERGE instead of CREATE.

        Returns:
            The created relationship as a dictionary.
        """
        operation = "MERGE" if merge else "CREATE"
        props_clause = "SET r += $properties" if properties else ""

        query = f"""
        MATCH (a:{start_label} {{id: $start_id}})
        MATCH (b:{end_label} {{id: $end_id}})
        {operation} (a)-[r:{rel_type}]->(b)
        {props_clause}
        RETURN r
        """
        params = {
            "start_id": start_id,
            "end_id": end_id,
            "properties": properties or {},
        }

        results = self.execute_write(query, params)
        return results[0]["r"] if results else {}

    def vector_search(
        self,
        index_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Perform a vector similarity search.

        Args:
            index_name: Name of the vector index.
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            filter_query: Optional Cypher WHERE clause for filtering.

        Returns:
            List of matching nodes with scores.
        """
        where_clause = f"WHERE {filter_query}" if filter_query else ""

        query = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        {where_clause}
        RETURN node, score
        ORDER BY score DESC
        """
        params = {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": query_vector,
        }

        return self.execute_read(query, params)

    def get_node_by_id(self, node_id: str, label: str) -> Optional[dict[str, Any]]:
        """Get a node by its ID.

        Args:
            node_id: The node's ID property.
            label: The node's label.

        Returns:
            The node as a dictionary, or None if not found.
        """
        query = f"""
        MATCH (n:{label} {{id: $id}})
        RETURN n
        """
        results = self.execute_read(query, {"id": node_id})
        return results[0]["n"] if results else None

    def delete_node(self, node_id: str, label: str, detach: bool = True) -> bool:
        """Delete a node by its ID.

        Args:
            node_id: The node's ID property.
            label: The node's label.
            detach: If True, also delete all relationships.

        Returns:
            True if a node was deleted.
        """
        detach_clause = "DETACH " if detach else ""
        query = f"""
        MATCH (n:{label} {{id: $id}})
        {detach_clause}DELETE n
        RETURN count(n) as deleted
        """
        results = self.execute_write(query, {"id": node_id})
        return results[0]["deleted"] > 0 if results else False


# Singleton client instance
_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """Get the global Neo4j client instance."""
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client


def close_neo4j_client() -> None:
    """Close the global Neo4j client."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
