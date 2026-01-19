# Data Room Graph - Claude Code Project Memory

## Project Overview

Recursive Agentic GraphRAG System for Private Equity Fund Data Rooms (Proof of Concept).

A Windows desktop application that lets professional investors create and manage multiple private equity "data rooms" (document corpora), ingest folders of documents, and convert them into a graph-first + vector hybrid knowledge system in Neo4j.

## Tech Stack (Implemented)

- **Language:** Python 3.11+
- **UI Framework:** PySide6 with QWebEngineView for graph visualization
- **Graph Database:** Neo4j 5.x with vector indexes
- **Graph Visualization:** vis.js via QWebEngineView
- **Hierarchical Chunking:** Unstructured.io API
- **Embeddings:** OpenAI text-embedding-3-large (3072 dimensions)
- **LLM/Agents:** Claude Opus 4 via Anthropic SDK
- **Packaging:** PyInstaller + InstallForge (planned)

## Key Components (Implemented)

1. **Windows Desktop UI** - PySide6-based multi-pane layout with data room management, folder ingestion via drag-drop, agent chat console, and vis.js graph explorer
2. **Backend Services** - Ingestion pipeline, entity extraction, GraphRAG retrieval, citation resolution
3. **Neo4j Graph Store** - Full schema with vector indexes for chunks and entities
4. **Hybrid Retrieval Layer** - Vector similarity + graph traversal + hierarchy-aware expansion
5. **Citation/Source Resolver** - Maps chunks/entities to (file path, page, offsets)

## Agents (Implemented)

1. **Data Engineer Agent** - Structure optimization, duplicate detection, classification review
2. **Enrichment Agent** - Gap identification against PE due diligence checklist
3. **Investment Professional Agent** - Conversational co-pilot with hybrid search and citations

## Commands

```bash
# Setup Neo4j (Docker)
python scripts/setup_neo4j.py start

# Stop Neo4j
python scripts/setup_neo4j.py stop

# Run the application (from src/)
python -m ui.main

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
data-room-graph/
├── CLAUDE.md                    # This file - project memory
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
├── .env.example                # Environment variables template
├── src/
│   ├── backend/
│   │   ├── config.py           # Configuration management
│   │   ├── database/
│   │   │   ├── neo4j_client.py # Neo4j driver wrapper
│   │   │   ├── schema.py       # Schema initialization
│   │   │   └── models.py       # Pydantic node models
│   │   ├── ingestion/
│   │   │   ├── unstructured.py # Unstructured.io API client
│   │   │   ├── chunker.py      # Hierarchy builder
│   │   │   ├── classifier.py   # Document classification
│   │   │   ├── embedder.py     # OpenAI embeddings
│   │   │   └── pipeline.py     # Ingestion orchestrator
│   │   ├── extraction/
│   │   │   ├── entity_extractor.py  # Claude-based extraction
│   │   │   ├── canonicalizer.py     # Entity deduplication
│   │   │   └── linker.py            # Evidence linking
│   │   ├── retrieval/
│   │   │   ├── graphrag.py     # Hybrid retrieval
│   │   │   ├── hierarchy.py    # Graph expansion
│   │   │   └── citations.py    # Citation resolver
│   │   └── agents/
│   │       ├── base.py         # Base agent class
│   │       ├── data_engineer.py
│   │       ├── enrichment.py
│   │       └── investor.py
│   └── ui/
│       ├── main.py             # Application entry point
│       ├── main_window.py      # Main window layout
│       ├── widgets/
│       │   ├── dataroom_list.py
│       │   ├── dashboard.py
│       │   ├── chat.py
│       │   ├── drop_zone.py
│       │   └── graph_view.py
│       └── resources/
│           ├── graph.html      # vis.js template
│           └── styles.qss      # Qt stylesheet
├── scripts/
│   └── setup_neo4j.py          # Docker Neo4j management
└── tests/                      # Test suite (to be implemented)
```

## Graph Data Model

### Node Types
- **Structure:** DataRoom, Folder, Document, Chunk
- **Entities:** Fund, Manager, Person, Vehicle, ServiceProvider, Investor

### Relationship Types
- **Structure:** CONTAINS, HAS_ROOT, NEXT
- **Entity Context:** MENTIONS, MANAGED_BY, HAS_SERVICE_PROVIDER, HAS_ROLE

### Node ID Format
```
{node_type}:{data_room_id}:{deterministic_hash}
```

## Document Classifications

LPA, Side letter, PPM, Subscription agreement, Capital call notice, Quarterly report, Annual report, Financial statements, Fee schedule, Track record/marketing deck, Other, Unknown

## Current Status

**All phases implemented:**
- Phase 1: Neo4j schema, UI scaffold, ingestion skeleton
- Phase 2: Unstructured.io chunking, hierarchy builder, embeddings
- Phase 3: Entity extraction, canonicalization, evidence linking
- Phase 4: GraphRAG hybrid retrieval, citations
- Phase 5: All three agents (Data Engineer, Enrichment, Investment Professional)
- Phase 6: Graph Navigator with vis.js, layer toggles, node expansion

## Environment Setup

1. Copy `.env.example` to `.env` and configure API keys:
   - `NEO4J_PASSWORD` - Neo4j database password
   - `OPENAI_API_KEY` - For embeddings
   - `ANTHROPIC_API_KEY` - For Claude agents
   - `UNSTRUCTURED_API_KEY` - For document parsing

2. Start Neo4j: `python scripts/setup_neo4j.py start`

3. Install dependencies: `pip install -r requirements.txt`

4. Run the application: `python -m ui.main` (from src/ directory)

## Notes

- POC constraint: No full auditability or temporal schema evolution yet
- Keep interfaces/IDs/evidence structures for future auditability
- Entity extraction uses Claude for high accuracy
- Vector indexes use HNSW with cosine similarity
