"""Graph visualization widget using QWebEngineView and vis.js."""

import logging
import json
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtCore import Qt, Signal, Slot, QObject, QUrl

from ...backend.database.neo4j_client import Neo4jClient, get_neo4j_client

logger = logging.getLogger(__name__)


class GraphBridge(QObject):
    """Bridge object for communication between Python and JavaScript."""

    # Signal when expansion is requested
    expansion_requested = Signal(str)  # node_id

    # Signal when document should be opened
    document_open_requested = Signal(str, int)  # path, page

    # Signal when full node details are requested
    full_node_details_requested = Signal(str)  # node_id

    # Signal when node relationships are requested
    node_relationships_requested = Signal(str)  # node_id

    def __init__(self, parent: Optional[QObject] = None):
        """Initialize the bridge.

        Args:
            parent: Parent QObject.
        """
        super().__init__(parent)

    @Slot(str)
    def requestExpansion(self, node_id: str):
        """Handle expansion request from JavaScript.

        Args:
            node_id: ID of node to expand.
        """
        logger.debug(f"Expansion requested for node: {node_id}")
        self.expansion_requested.emit(node_id)

    @Slot(str, int)
    def openDocument(self, path: str, page: int):
        """Handle document open request from JavaScript.

        Args:
            path: Path to document.
            page: Page number.
        """
        logger.debug(f"Open document requested: {path} page {page}")
        self.document_open_requested.emit(path, page)

    @Slot(str)
    def requestFullNodeDetails(self, node_id: str):
        """Handle full node details request from JavaScript.

        Args:
            node_id: ID of node to get full details for.
        """
        logger.debug(f"Full node details requested for: {node_id}")
        self.full_node_details_requested.emit(node_id)

    @Slot(str)
    def requestNodeRelationships(self, node_id: str):
        """Handle node relationships request from JavaScript.

        Args:
            node_id: ID of node to get relationships for.
        """
        logger.debug(f"Node relationships requested for: {node_id}")
        self.node_relationships_requested.emit(node_id)


class GraphViewWidget(QWidget):
    """Widget for visualizing the knowledge graph."""

    # Signal when a node is selected
    node_selected = Signal(str, str)  # node_id, node_type

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the widget.

        Args:
            neo4j_client: Neo4j client.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._dataroom_id: Optional[str] = None
        self.neo4j = neo4j_client or get_neo4j_client()
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 8, 8, 8)

        fit_btn = QPushButton("Fit to View")
        fit_btn.clicked.connect(self._on_fit)
        toolbar.addWidget(fit_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        toolbar.addWidget(refresh_btn)

        toolbar.addStretch()

        layout.addLayout(toolbar)

        # Web view
        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view, 1)

        # Setup web channel for JS-Python communication
        self.channel = QWebChannel()
        self.bridge = GraphBridge()
        self.channel.registerObject("graphBridge", self.bridge)
        self.web_view.page().setWebChannel(self.channel)

        # Enable loading remote content (for vis.js CDN)
        settings = self.web_view.settings()
        from PySide6.QtWebEngineCore import QWebEngineSettings
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)

        # Connect bridge signals
        self.bridge.expansion_requested.connect(self._on_expand_node)
        self.bridge.document_open_requested.connect(self._on_open_document)
        self.bridge.full_node_details_requested.connect(self._on_full_node_details)
        self.bridge.node_relationships_requested.connect(self._on_node_relationships)

        # Load the HTML template
        html_path = Path(__file__).parent.parent / "resources" / "graph.html"
        self.web_view.setUrl(QUrl.fromLocalFile(str(html_path)))

        # Wait for page to load before allowing refresh
        self._page_loaded = False
        self.web_view.loadFinished.connect(self._on_page_loaded)

    def _on_page_loaded(self, ok: bool):
        """Handle page load completion."""
        self._page_loaded = ok
        if ok:
            logger.info("Graph HTML page loaded successfully")
            # If we have a dataroom set, refresh the graph
            if self._dataroom_id:
                self._do_refresh()
        else:
            logger.error("Failed to load graph HTML page")

    def set_dataroom(self, dataroom_id: str):
        """Set the current data room.

        Args:
            dataroom_id: ID of the data room to visualize.
        """
        self._dataroom_id = dataroom_id
        self.refresh()

    def refresh(self):
        """Refresh the graph data."""
        if not self._dataroom_id:
            return

        # Only refresh if page is loaded
        if not getattr(self, '_page_loaded', False):
            logger.debug("Graph page not yet loaded, skipping refresh")
            return

        self._do_refresh()

    def _do_refresh(self):
        """Actually perform the refresh."""
        nodes, edges = self._load_graph_data()
        logger.info(f"Graph refresh: {len(nodes)} nodes, {len(edges)} edges")
        self._update_graph(nodes, edges)

    def _load_graph_data(self) -> tuple[list[dict], list[dict]]:
        """Load graph data from Neo4j.

        Returns:
            Tuple of (nodes, edges) lists.
        """
        # Check Neo4j connection is available
        if not self.neo4j:
            logger.warning("Neo4j client not available")
            return [], []

        try:
            nodes = []
            edges = []

            # Get DataRoom node
            dr_query = """
            MATCH (dr:DataRoom {id: $dataroom_id})
            RETURN dr.id as id, dr.name as name
            """
            dr_results = self.neo4j.execute_read(dr_query, {"dataroom_id": self._dataroom_id})

            if dr_results:
                dr = dr_results[0]
                nodes.append({
                    "id": dr["id"],
                    "name": dr["name"] or "Data Room",
                    "type": "DataRoom",
                    "size": 30,
                })

            # Get Documents directly connected to DataRoom
            doc_query = """
            MATCH (d:Document {dataroom_id: $dataroom_id})
            RETURN d.id as id, d.filename as name, d.doc_type as doc_type,
                   d.full_path as path, d.chunk_count as chunk_count
            LIMIT 50
            """
            doc_results = self.neo4j.execute_read(doc_query, {"dataroom_id": self._dataroom_id})

            for doc in doc_results or []:
                nodes.append({
                    "id": doc["id"],
                    "name": doc["name"],
                    "type": "Document",
                    "doc_type": doc.get("doc_type"),
                    "path": doc.get("path"),
                    "size": 18,
                })
                # Edge from DataRoom to Document
                edges.append({
                    "id": f"edge_dr_{doc['id']}",
                    "from": self._dataroom_id,
                    "to": doc["id"],
                    "type": "CONTAINS",
                })

            # Get Sections (via Page -> Section relationship)
            # First get unique sections
            section_query = """
            MATCH (s:Section {dataroom_id: $dataroom_id})<-[:HAS_SECTION]-(p:Page)<-[:HAS_PAGE]-(d:Document)
            RETURN DISTINCT s.id as id, s.title as title, s.description as description,
                   s.hierarchy_level as level, collect(DISTINCT p.id)[0] as page_id
            LIMIT 50
            """
            section_results = self.neo4j.execute_read(section_query, {"dataroom_id": self._dataroom_id})

            for section in section_results or []:
                nodes.append({
                    "id": section["id"],
                    "name": section["title"][:40] + "..." if len(section["title"]) > 40 else section["title"],
                    "type": "Section",
                    "title": section["title"],
                    "description": section.get("description"),
                    "level": section.get("level", 0),
                    "size": 14,
                })
                # Edge from Page to Section (use first page)
                if section.get("page_id"):
                    edges.append({
                        "id": f"edge_{section['page_id']}_{section['id']}",
                        "from": section["page_id"],
                        "to": section["id"],
                        "type": "HAS_SECTION",
                    })

            # Get Pages (limited sample)
            page_query = """
            MATCH (p:Page {dataroom_id: $dataroom_id})<-[:HAS_PAGE]-(d:Document)
            RETURN p.id as id, p.page_number as page_number, d.id as doc_id
            LIMIT 50
            """
            page_results = self.neo4j.execute_read(page_query, {"dataroom_id": self._dataroom_id})

            for page in page_results or []:
                nodes.append({
                    "id": page["id"],
                    "name": f"Page {page['page_number']}",
                    "type": "Page",
                    "page_number": page["page_number"],
                    "size": 8,
                })
                # Edge from Document to Page
                edges.append({
                    "id": f"edge_{page['doc_id']}_{page['id']}",
                    "from": page["doc_id"],
                    "to": page["id"],
                    "type": "HAS_PAGE",
                })

            # Get Chunks that have MENTIONS relationships (these connect to entities)
            # Hierarchy: Document -> Page -> Section -> Chunk
            chunk_query = """
            MATCH (c:Chunk {dataroom_id: $dataroom_id})-[:MENTIONS]->(e:Entity)
            MATCH (c)<-[:CONTAINS]-(s:Section)
            MATCH (c)-[:ON_PAGE]->(p:Page)
            RETURN DISTINCT c.id as id, c.text as text, c.element_type as element_type,
                   p.page_number as page, s.id as section_id
            LIMIT 100
            """
            chunk_results = self.neo4j.execute_read(chunk_query, {"dataroom_id": self._dataroom_id})

            for chunk in chunk_results or []:
                text = chunk.get("text", "")
                nodes.append({
                    "id": chunk["id"],
                    "name": text[:40] + "..." if len(text) > 40 else text,
                    "type": "Chunk",
                    "text": text,
                    "page": chunk.get("page"),
                    "size": 10,
                })
                # Edge from Section to Chunk
                if chunk.get("section_id"):
                    edges.append({
                        "id": f"edge_{chunk['section_id']}_{chunk['id']}",
                        "from": chunk["section_id"],
                        "to": chunk["id"],
                        "type": "CONTAINS",
                    })

            # Get Entities
            entity_query = """
            MATCH (e:Entity {dataroom_id: $dataroom_id})
            RETURN e.id as id, e.canonical_name as name, e.entity_type as type,
                   e.mention_count as mentions
            LIMIT 50
            """
            entity_results = self.neo4j.execute_read(entity_query, {"dataroom_id": self._dataroom_id})
            logger.info(f"Entity query returned {len(entity_results) if entity_results else 0} entities")

            for entity in entity_results or []:
                nodes.append({
                    "id": entity["id"],
                    "name": entity["name"],
                    "type": entity["type"],
                    "mentions": entity.get("mentions", 0),
                    "size": 14 + min(entity.get("mentions", 0) or 0, 20) * 0.5,
                })

            # Get MENTIONS relationships (Chunk -> Entity)
            mentions_query = """
            MATCH (c:Chunk {dataroom_id: $dataroom_id})-[r:MENTIONS]->(e:Entity)
            RETURN c.id as chunk_id, e.id as entity_id
            LIMIT 100
            """
            mentions_results = self.neo4j.execute_read(mentions_query, {"dataroom_id": self._dataroom_id})

            # Only add edges for chunks that are in our node set
            chunk_ids = {n["id"] for n in nodes if n["type"] == "Chunk"}
            mentions_added = 0
            for rel in mentions_results or []:
                if rel["chunk_id"] in chunk_ids:
                    edges.append({
                        "id": f"edge_mention_{rel['chunk_id']}_{rel['entity_id']}",
                        "from": rel["chunk_id"],
                        "to": rel["entity_id"],
                        "type": "MENTIONS",
                    })
                    mentions_added += 1
            logger.info(f"Added {mentions_added} MENTIONS edges (from {len(mentions_results) if mentions_results else 0} total)")

            # Log node type breakdown
            node_types = {}
            for n in nodes:
                t = n.get("type", "Unknown")
                node_types[t] = node_types.get(t, 0) + 1
            logger.info(f"Loaded {len(nodes)} nodes: {node_types}")
            logger.info(f"Loaded {len(edges)} edges")
            return nodes, edges

        except Exception as e:
            logger.error(f"Error loading graph data: {e}")
            return [], []

    def _update_graph(self, nodes: list[dict], edges: list[dict]):
        """Update the graph visualization.

        Args:
            nodes: List of node dictionaries.
            edges: List of edge dictionaries.
        """
        # Log entity count being sent to JS
        entity_types = ['Fund', 'Manager', 'Person', 'Vehicle', 'ServiceProvider', 'Investor']
        entity_count = sum(1 for n in nodes if n.get('type') in entity_types)
        logger.info(f"Sending to JS: {len(nodes)} nodes ({entity_count} entities), {len(edges)} edges")

        def js_callback(result):
            logger.info(f"Graph update JS result: {result}")

        js_code = f"""
        (function() {{
            if (typeof vis === 'undefined') {{
                return 'ERROR: vis.js not loaded';
            }}
            if (!window.graphAPI) {{
                return 'ERROR: graphAPI not ready';
            }}
            try {{
                window.graphAPI.clearGraph();
                window.graphAPI.initGraph({json.dumps(nodes)}, {json.dumps(edges)});
                return 'SUCCESS: ' + {len(nodes)} + ' nodes, ' + {len(edges)} + ' edges';
            }} catch(e) {{
                return 'ERROR: ' + e.message;
            }}
        }})();
        """
        self.web_view.page().runJavaScript(js_code, js_callback)

    def add_nodes(self, nodes: list[dict], edges: list[dict]):
        """Add nodes and edges to the existing graph.

        Args:
            nodes: New nodes to add.
            edges: New edges to add.
        """
        logger.info(f"Adding {len(nodes)} nodes and {len(edges)} edges to graph")

        if not nodes and not edges:
            logger.info("No new nodes or edges to add")
            return

        js_code = f"""
        if (window.graphAPI) {{
            window.graphAPI.addData({json.dumps(nodes)}, {json.dumps(edges)});
        }}
        """
        self.web_view.page().runJavaScript(js_code)

    @Slot()
    def _on_fit(self):
        """Fit graph to view."""
        self.web_view.page().runJavaScript(
            "if (window.graphAPI) window.graphAPI.fitGraph();"
        )

    @Slot(str)
    def _on_expand_node(self, node_id: str):
        """Handle node expansion request.

        Args:
            node_id: ID of the node to expand.
        """
        logger.info(f"Expand requested for node: {node_id}")

        # Determine node type from ID prefix
        if node_id.startswith("document:"):
            logger.info("Expanding as Document")
            self._expand_document(node_id)
        elif node_id.startswith("section:"):
            logger.info("Expanding as Section")
            self._expand_section(node_id)
        elif node_id.startswith("page:"):
            logger.info("Expanding as Page")
            self._expand_page(node_id)
        elif node_id.startswith("chunk:"):
            logger.info("Expanding as Chunk")
            self._expand_chunk(node_id)
        elif node_id.startswith(("fund:", "manager:", "person:", "vehicle:", "serviceprovider:", "investor:")):
            logger.info("Expanding as Entity")
            self._expand_entity(node_id)
        else:
            logger.warning(f"Unknown node type for expansion: {node_id}")

    def _expand_document(self, doc_id: str):
        """Expand a document node to show pages.

        Args:
            doc_id: Document ID.
        """
        if not self.neo4j:
            logger.warning("Neo4j client not available for document expansion")
            return

        nodes = []
        edges = []

        # Get pages (Documents now connect to Pages, which connect to Sections)
        page_query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_PAGE]->(p:Page)
        RETURN p.id as id, p.page_number as page_number
        ORDER BY p.page_number
        LIMIT 20
        """

        page_results = self.neo4j.execute_read(page_query, {"doc_id": doc_id})

        for page in page_results or []:
            nodes.append({
                "id": page["id"],
                "name": f"Page {page['page_number']}",
                "type": "Page",
                "page_number": page["page_number"],
                "size": 8,
            })
            edges.append({
                "id": f"edge_{doc_id}_{page['id']}",
                "from": doc_id,
                "to": page["id"],
                "type": "HAS_PAGE",
            })

        self.add_nodes(nodes, edges)

    def _expand_section(self, section_id: str):
        """Expand a section node to show chunks and nested sections.

        Args:
            section_id: Section ID.
        """
        if not self.neo4j:
            logger.warning("Neo4j client not available for section expansion")
            return

        # Get chunks in this section
        chunk_query = """
        MATCH (s:Section {id: $section_id})-[:CONTAINS]->(c:Chunk)
        OPTIONAL MATCH (c)-[:ON_PAGE]->(p:Page)
        RETURN c.id as id, c.text as text, c.element_type as type,
               p.page_number as page
        LIMIT 20
        """

        chunk_results = self.neo4j.execute_read(chunk_query, {"section_id": section_id})

        nodes = []
        edges = []

        for chunk in chunk_results or []:
            nodes.append({
                "id": chunk["id"],
                "name": chunk["text"][:50] + "..." if len(chunk["text"]) > 50 else chunk["text"],
                "type": "Chunk",
                "text": chunk["text"],
                "page": chunk.get("page"),
                "size": 10,
            })
            edges.append({
                "id": f"edge_{section_id}_{chunk['id']}",
                "from": section_id,
                "to": chunk["id"],
                "type": "CONTAINS",
            })

        # Get nested sections
        nested_query = """
        MATCH (s:Section {id: $section_id})-[:CONTAINS]->(nested:Section)
        RETURN nested.id as id, nested.title as title, nested.hierarchy_level as level
        LIMIT 10
        """

        nested_results = self.neo4j.execute_read(nested_query, {"section_id": section_id})

        for nested in nested_results or []:
            nodes.append({
                "id": nested["id"],
                "name": nested["title"][:40] + "..." if len(nested["title"]) > 40 else nested["title"],
                "type": "Section",
                "title": nested["title"],
                "level": nested.get("level", 0),
                "size": 12,
            })
            edges.append({
                "id": f"edge_{section_id}_{nested['id']}",
                "from": section_id,
                "to": nested["id"],
                "type": "CONTAINS",
            })

        self.add_nodes(nodes, edges)

    def _expand_page(self, page_id: str):
        """Expand a page node to show sections and chunks on this page.

        Args:
            page_id: Page ID.
        """
        if not self.neo4j:
            logger.warning("Neo4j client not available for page expansion")
            return

        # Get sections on this page (Page -[HAS_SECTION]-> Section)
        section_query = """
        MATCH (p:Page {id: $page_id})-[:HAS_SECTION]->(s:Section)
        RETURN s.id as id, s.title as title
        LIMIT 10
        """

        section_results = self.neo4j.execute_read(section_query, {"page_id": page_id})

        nodes = []
        edges = []

        for section in section_results or []:
            nodes.append({
                "id": section["id"],
                "name": section["title"][:40] + "..." if len(section["title"]) > 40 else section["title"],
                "type": "Section",
                "title": section["title"],
                "size": 14,
            })
            edges.append({
                "id": f"edge_{page_id}_{section['id']}",
                "from": page_id,
                "to": section["id"],
                "type": "HAS_SECTION",
            })

        # Get chunks on this page (Chunk -[ON_PAGE]-> Page)
        chunk_query = """
        MATCH (p:Page {id: $page_id})<-[:ON_PAGE]-(c:Chunk)
        RETURN c.id as id, c.text as text, c.element_type as type
        LIMIT 15
        """

        chunk_results = self.neo4j.execute_read(chunk_query, {"page_id": page_id})

        for chunk in chunk_results or []:
            nodes.append({
                "id": chunk["id"],
                "name": chunk["text"][:40] + "..." if len(chunk["text"]) > 40 else chunk["text"],
                "type": "Chunk",
                "text": chunk["text"],
                "size": 10,
            })
            edges.append({
                "id": f"edge_{chunk['id']}_{page_id}",
                "from": chunk["id"],
                "to": page_id,
                "type": "ON_PAGE",
            })

        self.add_nodes(nodes, edges)

    def _expand_chunk(self, chunk_id: str):
        """Expand a chunk node to show mentioned entities.

        Args:
            chunk_id: Chunk ID.
        """
        if not self.neo4j:
            logger.warning("Neo4j client not available for chunk expansion")
            return

        query = """
        MATCH (c:Chunk {id: $chunk_id})-[r:MENTIONS]->(e:Entity)
        RETURN e.id as id, e.canonical_name as name, e.entity_type as type,
               e.mention_count as mentions
        """

        results = self.neo4j.execute_read(query, {"chunk_id": chunk_id})

        nodes = []
        edges = []

        for entity in results:
            nodes.append({
                "id": entity["id"],
                "name": entity["name"],
                "type": entity["type"],
                "mentions": entity.get("mentions", 0),
                "size": 12,
            })
            edges.append({
                "id": f"edge_{chunk_id}_{entity['id']}",
                "from": chunk_id,
                "to": entity["id"],
                "type": "MENTIONS",
            })

        self.add_nodes(nodes, edges)

    def _expand_entity(self, entity_id: str):
        """Expand an entity node to show related chunks.

        Args:
            entity_id: Entity ID.
        """
        if not self.neo4j:
            logger.warning("Neo4j client not available for entity expansion")
            return

        # Hierarchy: Document -> Page -> Section -> Chunk
        query = """
        MATCH (e:Entity {id: $entity_id})<-[r:MENTIONS]-(c:Chunk)
        MATCH (c)-[:ON_PAGE]->(p:Page)<-[:HAS_PAGE]-(d:Document)
        RETURN c.id as chunk_id, c.text as text, d.id as doc_id, d.filename as doc_name
        LIMIT 10
        """

        results = self.neo4j.execute_read(query, {"entity_id": entity_id})
        logger.info(f"Entity expansion query returned {len(results) if results else 0} results")

        nodes = []
        edges = []

        for result in results or []:
            nodes.append({
                "id": result["chunk_id"],
                "name": result["text"][:50] + "...",
                "type": "Chunk",
                "text": result["text"],
                "size": 10,
            })
            edges.append({
                "id": f"edge_{result['chunk_id']}_{entity_id}",
                "from": result["chunk_id"],
                "to": entity_id,
                "type": "MENTIONS",
            })

        self.add_nodes(nodes, edges)

    @Slot(str, int)
    def _on_open_document(self, path: str, page: int):
        """Handle document open request.

        Args:
            path: Path to document.
            page: Page number.
        """
        import os
        import sys

        if not path:
            return

        # Validate path exists and is a file to prevent command injection
        resolved_path = Path(path).resolve()
        if not resolved_path.exists() or not resolved_path.is_file():
            logger.warning(f"Invalid document path: {path}")
            return

        try:
            if sys.platform == "win32":
                # Use os.startfile which is safe and doesn't use shell
                os.startfile(str(resolved_path))
            elif sys.platform == "darwin":
                import subprocess
                subprocess.run(["open", str(resolved_path)], check=False)
            else:
                import subprocess
                subprocess.run(["xdg-open", str(resolved_path)], check=False)
        except Exception as e:
            logger.error(f"Failed to open document: {e}")

    @Slot(str)
    def _on_full_node_details(self, node_id: str):
        """Handle full node details request.

        Args:
            node_id: ID of the node to get full details for.
        """
        logger.info(f"Full node details requested for: {node_id}")

        if not self.neo4j:
            logger.warning("Neo4j client not available for full node details")
            self._send_full_node_details({})
            return

        try:
            # Query for all properties of the node
            query = """
            MATCH (n {id: $node_id})
            RETURN properties(n) as props
            """
            results = self.neo4j.execute_read(query, {"node_id": node_id})

            if results and len(results) > 0:
                props = results[0].get("props", {})
                # Filter out embedding vectors (too large to display)
                filtered_props = {
                    k: v for k, v in props.items()
                    if not k.endswith("_embedding") and k != "embedding"
                }
                self._send_full_node_details(filtered_props)
            else:
                self._send_full_node_details({})

        except Exception as e:
            logger.error(f"Error fetching full node details: {e}")
            self._send_full_node_details({})

    def _send_full_node_details(self, properties: dict):
        """Send full node details to JavaScript.

        Args:
            properties: Dictionary of node properties.
        """
        js_code = f"""
        if (window.graphAPI && window.graphAPI.displayFullNodeDetails) {{
            window.graphAPI.displayFullNodeDetails({json.dumps(properties)});
        }}
        """
        self.web_view.page().runJavaScript(js_code)

    @Slot(str)
    def _on_node_relationships(self, node_id: str):
        """Handle node relationships request.

        Args:
            node_id: ID of the node to get relationships for.
        """
        logger.info(f"Node relationships requested for: {node_id}")

        if not self.neo4j:
            logger.warning("Neo4j client not available for node relationships")
            self._send_node_relationships([])
            return

        try:
            # Query for all relationships (both directions)
            query = """
            MATCH (n {id: $node_id})-[r]-(other)
            RETURN type(r) as rel_type,
                   properties(r) as rel_props,
                   CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END as direction,
                   other.id as other_id,
                   labels(other)[0] as other_type,
                   COALESCE(other.name, other.canonical_name, other.filename, other.text) as other_name
            LIMIT 50
            """
            results = self.neo4j.execute_read(query, {"node_id": node_id})

            relationships = []
            for row in results or []:
                rel_props = row.get("rel_props", {}) or {}
                # Filter out embedding vectors from relationship properties
                filtered_props = {
                    k: v for k, v in rel_props.items()
                    if not k.endswith("_embedding") and k != "embedding"
                }

                other_name = row.get("other_name")
                if other_name and len(other_name) > 100:
                    other_name = other_name[:100] + "..."

                relationships.append({
                    "type": row.get("rel_type"),
                    "direction": row.get("direction"),
                    "properties": filtered_props,
                    "other_node_id": row.get("other_id"),
                    "other_node_type": row.get("other_type"),
                    "other_node_name": other_name,
                })

            self._send_node_relationships(relationships)

        except Exception as e:
            logger.error(f"Error fetching node relationships: {e}")
            self._send_node_relationships([])

    def _send_node_relationships(self, relationships: list):
        """Send node relationships to JavaScript.

        Args:
            relationships: List of relationship dictionaries.
        """
        js_code = f"""
        if (window.graphAPI && window.graphAPI.displayNodeRelationships) {{
            window.graphAPI.displayNodeRelationships({json.dumps(relationships)});
        }}
        """
        self.web_view.page().runJavaScript(js_code)
