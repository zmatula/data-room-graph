# Neo4j GraphRAG Migration Specification

**Version:** 1.0
**Date:** 2026-01-20
**Status:** Ready for Implementation Planning
**Target Branch:** `claude/neo4j-graphrag-migration`

---

## Executive Summary

Replace the current custom ingestion pipeline with Neo4j's official `neo4j-graphrag-python` package while preserving domain-specific PE (Private Equity) extraction logic and maintaining backward compatibility with existing graph schema and retrieval patterns.

**Key Decisions:**
- ✅ **Con #1 Resolution:** Store section and page metadata as properties on Chunk nodes (flat structure)
- ✅ **Con #2 Resolution:** Use custom PE-specific prompt in `LLMEntityRelationExtractor`
- ✅ **Con #3 Resolution:** Extend `FuzzyMatchResolver` with cross-type duplicate detection

**Migration Strategy:** Big bang replacement with feature parity validation

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Target Architecture](#2-target-architecture)
3. [Component Specifications](#3-component-specifications)
4. [Migration Plan](#4-migration-plan)
5. [Implementation Requirements](#5-implementation-requirements)
6. [Testing Strategy](#6-testing-strategy)
7. [Rollback Plan](#7-rollback-plan)
8. [Appendices](#8-appendices)

---

## 1. Current State Analysis

### 1.1 Current Pipeline Components

```
src/backend/ingestion/
├── pipeline.py              # IngestionPipeline orchestrator
├── unstructured.py          # UnstructuredClient (Unstructured.io API)
├── chunker.py               # SectionChunkBuilder (hierarchy builder)
├── classifier.py            # DocumentClassifier (PE doc types)
├── embedder.py              # Embedder (OpenAI embeddings)
└── (page_builder.py)        # PageBuilder (referenced but missing)

src/backend/extraction/
├── entity_extractor.py      # EntityExtractor (Claude-based)
├── canonicalizer.py         # EntityCanonicalizer (deduplication)
└── linker.py                # EntityLinker (chunk-entity relationships)

src/backend/retrieval/
├── graphrag.py              # GraphRAGRetriever (hybrid search)
├── hierarchy.py             # HierarchyExpander (graph traversal)
└── citations.py             # CitationResolver
```

### 1.2 Current Data Model

**Hierarchical Graph Structure:**
```
DataRoom
  └─ Folder (HAS_ROOT / CONTAINS)
      └─ Document (CONTAINS)
          ├─ Page (HAS_PAGE, NEXT)
          │   └─ Section (HAS_SECTION, CONTAINS)
          │       └─ Chunk (CONTAINS, ON_PAGE, NEXT)
          └─ Entity (MENTIONS from Chunk)
```

**Key Node Types:**
- `DataRoom`: Top-level container
- `Folder`: Directory structure with `narrative` (Claude-generated)
- `Document`: File with `doc_type`, `doc_type_confidence`, `ingestion_status`
- `Page`: Page number tracking with sequential NEXT relationships
- `Section`: Document sections with `title`, `description`, `hierarchy_level`, `hierarchy_path`, `parent_section_id`
- `Chunk`: Text chunks with `text`, `embedding`, `sequence_order`, `element_type`, `offset_start`, `offset_end`, `coordinates`
- `Entity`: 9 types (Fund, Manager, Person, Vehicle, ServiceProvider, Investor, PortfolioCompany, Location, Asset)

### 1.3 Current Pipeline Flow

```
1. Discovery: Find all supported files
2. Folder Creation: Create Folder nodes with hierarchy
3. Folder Narratives (optional): Generate folder descriptions
4. Parallel Document Processing:
   a. Parse with Unstructured.io → elements
   b. Classify document type → doc_type
   c. Build Pages → Page nodes
   d. Build Sections → Section nodes with hierarchy
   e. Build Chunks → Chunk nodes
   f. Generate section descriptions (optional)
   g. Generate embeddings → OpenAI
   h. Save to Neo4j (Document, Pages, Sections, Chunks)
5. Entity Extraction (optional):
   a. Extract entities from chunks → Claude
   b. Canonicalize entities → deduplication
   c. Link entities to chunks → MENTIONS
   d. Create entity relationships
6. Update stats: Document/chunk/entity counts
```

### 1.4 Critical Features to Preserve

**PE Domain Expertise:**
- 130-line entity extraction prompt with disambiguation rules:
  - Fund vs Manager (e.g., "EnCap Fund XII" vs "EnCap Investments LP")
  - Portfolio Company vs Fund
  - Location vs Company (e.g., "Permian Basin" vs company names)
- Document classification patterns (12 types: LPA, Side Letter, PPM, etc.)

**Cross-Type Entity Deduplication:**
- Detects when same entity is misclassified with different types
- Prevents "EnCap" from being created as both Fund and Manager

**Hierarchy-Aware Retrieval:**
- Section-based expansion (`HierarchyExpander.expand_chunks`)
- Sequential chunk traversal (NEXT relationships)
- Entity co-occurrence patterns

**Progress Tracking:**
- Real-time progress updates via `IngestionProgress`
- File-level and chunk-level progress
- Entity extraction progress

---

## 2. Target Architecture

### 2.1 Neo4j GraphRAG Components

**Package:** `neo4j-graphrag-python` (formerly `neo4j-genai`)

**Core Components:**
```python
from neo4j_graphrag.experimental.pipeline import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters import FixedSizeSplitter
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.experimental.components.resolver import FuzzyMatchResolver
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.schema import SchemaBuilder
```

### 2.2 Target Data Model

**Simplified Graph Structure (Con #1 Resolution):**
```
DataRoom
  └─ Folder (HAS_ROOT / CONTAINS)
      └─ Document (CONTAINS)
          ├─ Chunk (CONTAINS, NEXT)
          │   - Properties: text, embedding, section_title, section_path,
          │                 page_number, sequence_order, element_type,
          │                 offset_start, offset_end, coordinates
          └─ Entity (MENTIONS from Chunk)
```

**Changes from Current:**
- ❌ **Removed:** Page nodes (page_number becomes Chunk property)
- ❌ **Removed:** Section nodes (section_title, section_path become Chunk properties)
- ✅ **Preserved:** Folder hierarchy with narratives
- ✅ **Preserved:** Document classification
- ✅ **Preserved:** Chunk sequence order and NEXT relationships
- ✅ **Preserved:** Entity extraction and linking
- ✅ **Preserved:** All metadata needed for citations

### 2.3 Target Pipeline Flow

```
1. Discovery: Find all supported files (keep current logic)
2. Folder Creation: Create Folder nodes (keep current logic)
3. Folder Narratives: Generate descriptions (keep current logic)
4. Parallel Document Processing (NEW - Neo4j GraphRAG):
   a. Parse with Unstructured.io (via custom integration)
   b. Classify document type (via custom component)
   c. Custom Text Splitting:
      - Extract page numbers from elements
      - Extract section hierarchy from Title elements
      - Build chunks with flattened metadata
   d. Embed chunks (TextChunkEmbedder)
   e. Extract entities (LLMEntityRelationExtractor with PE prompt)
   f. Resolve duplicates (CrossTypeFuzzyResolver)
   g. Write to Neo4j (Neo4jWriter)
5. Entity Relationships: Create co-occurrence relationships
6. Update stats: Document/chunk/entity counts
```

### 2.4 New Components to Build

**Custom Components (extend Neo4j base classes):**
```
src/backend/ingestion/
├── neo4j_pipeline.py                 # NEW: Neo4jGraphRAGPipeline
├── hierarchical_splitter.py          # NEW: HierarchicalTextSplitter
├── pe_entity_extractor.py            # NEW: PEEntityExtractor (wraps LLMEntityRelationExtractor)
├── cross_type_resolver.py            # NEW: CrossTypeFuzzyResolver
├── document_classifier_component.py  # NEW: DocumentClassifierComponent
└── unstructured_parser.py            # NEW: UnstructuredParser (component wrapper)

# Keep but modify:
├── classifier.py                      # Keep classification patterns
├── embedder.py                        # Keep as-is (already compatible)
```

---

## 3. Component Specifications

### 3.1 HierarchicalTextSplitter

**Purpose:** Custom text splitter that preserves page/section metadata as chunk properties

**Base Class:** `neo4j_graphrag.experimental.components.text_splitters.TextSplitter`

**Input:** `UnstructuredElement[]` from Unstructured.io
**Output:** `TextChunk[]` with enriched metadata

**Core Logic:**
```python
class HierarchicalTextSplitter(TextSplitter):
    """
    Splits text while preserving document structure as chunk metadata.

    Extracts from Unstructured.io elements:
    - Page numbers → chunk.page_number
    - Section titles (from Title elements) → chunk.section_title, chunk.section_path
    - Element types → chunk.element_type
    - Coordinates → chunk.coordinates
    - Text offsets → chunk.offset_start, chunk.offset_end

    Builds chunks maintaining reading sequence order.
    """

    def __init__(self, max_chunk_tokens: int = 512):
        self.max_chunk_tokens = max_chunk_tokens
        self.max_chunk_chars = max_chunk_tokens * 4
        self.section_builder = SectionChunkBuilder()  # Reuse existing logic

    async def run(self, elements: list[UnstructuredElement]) -> list[TextChunk]:
        # 1. Build section tree (use current chunker.py logic)
        section_tree = self._build_section_tree(elements)

        # 2. Flatten sections into chunk metadata
        chunks = []
        for section_node in section_tree:
            section_metadata = {
                'section_title': section_node.title,
                'section_path': section_node.get_hierarchy_path(),
                'section_level': section_node.depth
            }

            # 3. Create chunks from section content
            for element in section_node.content_elements:
                if not element.text.strip():
                    continue

                chunk = TextChunk(
                    text=element.text,
                    metadata={
                        **section_metadata,
                        'page_number': element.page_number,
                        'element_type': element.type,
                        'offset_start': element.offset_start,
                        'offset_end': element.offset_end,
                        'coordinates': element.coordinates,
                        'sequence_order': global_order
                    }
                )
                chunks.append(chunk)
                global_order += 1

        # 4. Split oversized chunks
        return self._split_large_chunks(chunks)
```

**Files to Reference:**
- `src/backend/ingestion/chunker.py` (SectionChunkBuilder logic)
- `src/backend/ingestion/unstructured.py` (UnstructuredElement)

---

### 3.2 PEEntityExtractor

**Purpose:** Wrap Neo4j's LLMEntityRelationExtractor with PE-specific extraction prompt

**Base Class:** `neo4j_graphrag.experimental.components.entity_relation_extractor.LLMEntityRelationExtractor`

**Configuration:**
```python
class PEEntityExtractor(LLMEntityRelationExtractor):
    """
    Entity extractor with PE domain-specific prompt for accurate classification.

    Preserves the 130-line extraction prompt with:
    - Fund vs Manager disambiguation
    - Portfolio Company vs Fund rules
    - Location vs Company distinction (Permian Basin, etc.)
    - 9 entity types: Fund, Manager, Person, Vehicle, ServiceProvider,
                      Investor, PortfolioCompany, Location, Asset
    """

    def __init__(self, llm, neo4j_driver, schema):
        # Load PE-specific prompt
        prompt_template = self._load_pe_prompt_template()

        super().__init__(
            llm=llm,
            driver=neo4j_driver,
            create_lexical_graph=True,  # Auto-create Chunk-MENTIONS->Entity
            prompt_template=prompt_template,
            schema=schema,
            on_error="ignore",  # Continue on individual chunk failures
            max_concurrency=5
        )

    def _load_pe_prompt_template(self) -> str:
        """
        Load the PE extraction prompt from entity_extractor.py
        and adapt to Neo4j's expected format.
        """
        # Read EXTRACTION_PROMPT from entity_extractor.py
        # Adapt to Neo4j's input format (includes schema placeholders)
        return adapted_prompt
```

**Prompt Adaptation Required:**
- Current prompt expects `{text}` and `{context_section}`
- Neo4j format expects: `{examples}`, `{schema}`, `{text}`
- Need to merge PE prompt with Neo4j's schema injection

**Schema Definition:**
```python
from neo4j_graphrag.experimental.components.schema import SchemaBuilder, NodeType, RelationshipType

def build_pe_schema():
    schema = SchemaBuilder()

    # Define all 9 entity types
    schema.add_node_type(NodeType(
        label="Fund",
        properties=["name", "vintage_year", "strategy", "size_millions"]
    ))

    schema.add_node_type(NodeType(
        label="Manager",
        properties=["name", "aum", "headquarters"]
    ))

    # ... define all 9 types

    # Define relationships
    schema.add_relationship_type(RelationshipType(
        type="MANAGED_BY",
        from_node="Fund",
        to_node="Manager"
    ))

    # ... define standard relationships

    return schema.get_schema()
```

**Files to Reference:**
- `src/backend/extraction/entity_extractor.py` (EXTRACTION_PROMPT)
- `src/backend/database/models.py` (EntityType enum, entity models)

---

### 3.3 CrossTypeFuzzyResolver

**Purpose:** Extend Neo4j's FuzzyMatchResolver to detect cross-type duplicates

**Base Class:** `neo4j_graphrag.experimental.components.resolver.FuzzyMatchResolver`

**Core Logic:**
```python
class CrossTypeFuzzyResolver(FuzzyMatchResolver):
    """
    Entity resolver that checks for duplicates across all entity types.

    Prevents misclassified entities from creating duplicates:
    - "EnCap Investments LP" (Manager) vs "EnCap Investments" (Fund)
    - Uses RapidFuzz for fast fuzzy matching (5-10x faster than difflib)
    - Applies legal suffix normalization (LLC, LP, L.P., etc.)
    """

    def __init__(self, driver, threshold: float = 0.85, **kwargs):
        super().__init__(driver=driver, threshold=threshold, **kwargs)
        self.normalizer = EntityNameNormalizer()  # Port from canonicalizer.py

    async def resolve(self, entity):
        # Phase 1: Try same-type match (fast path)
        same_type_match = await super().resolve(entity)
        if same_type_match:
            return same_type_match

        # Phase 2: Cross-type check
        normalized_name = self.normalizer.normalize(entity.name)

        query = """
        MATCH (e:Entity {dataroom_id: $dataroom_id})
        WHERE e.entity_type <> $exclude_type
        WITH e,
             fuzz.ratio($normalized_name, $normalize_fn(e.name)) as similarity
        WHERE similarity > $threshold
        RETURN e, similarity
        ORDER BY similarity DESC
        LIMIT 1
        """

        result = await self.driver.execute_query(query, {
            "dataroom_id": entity.dataroom_id,
            "exclude_type": entity.entity_type,
            "normalized_name": normalized_name,
            "threshold": self.threshold
        })

        if result:
            logger.warning(
                f"Cross-type duplicate detected: "
                f"'{entity.name}' ({entity.entity_type}) → "
                f"'{result[0]['e'].name}' ({result[0]['e'].entity_type})"
            )
            return result[0]['e']

        return None

    def _install_cypher_functions(self):
        """
        Install custom Cypher functions for name normalization.
        Uses Neo4j's user-defined functions or pre-processing.
        """
        # Create Cypher function for legal suffix removal
        # Or pre-normalize in Python and compare
        pass
```

**Name Normalization (Port from canonicalizer.py):**
```python
class EntityNameNormalizer:
    """Normalizes entity names for comparison."""

    LEGAL_SUFFIXES = [r"\bLLC\b", r"\bL\.L\.C\.\b", r"\bLLP\b", ...]
    CLEANUP_PATTERNS = [r"\bThe\b", r"\bA\b", r"[,\.\-\(\)]", ...]

    def normalize(self, name: str) -> str:
        # Remove legal suffixes
        normalized = self.suffix_pattern.sub("", name)
        # Apply cleanup patterns
        for pattern in self.cleanup_patterns:
            normalized = pattern.sub(" ", normalized)
        # Collapse whitespace and lowercase
        return " ".join(normalized.split()).strip().lower()
```

**Files to Reference:**
- `src/backend/extraction/canonicalizer.py` (normalize_name, LEGAL_SUFFIXES)

---

### 3.4 Neo4jGraphRAGPipeline

**Purpose:** Main pipeline orchestrator replacing `IngestionPipeline`

**Architecture:**
```python
class Neo4jGraphRAGPipeline:
    """
    Orchestrates document ingestion using Neo4j GraphRAG components.

    Maintains backward compatibility with:
    - IngestionProgress callbacks
    - IngestionResult format
    - Folder hierarchy creation
    - Document classification
    - Error handling and retry logic
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        unstructured_client: UnstructuredClient,
        embedder: Embedder,
        extract_entities: bool = True,
        generate_narratives: bool = True,
        max_concurrent_files: int = 4
    ):
        self.neo4j = neo4j_client
        self.unstructured = unstructured_client
        self.embedder = embedder
        self.extract_entities = extract_entities
        self.generate_narratives = generate_narratives
        self.max_concurrent_files = max_concurrent_files

        # Build Neo4j GraphRAG pipeline
        self._build_pipeline()

        # Progress tracking (maintain compatibility)
        self._progress = IngestionProgress()
        self._progress_callback = None

    def _build_pipeline(self):
        """Construct Neo4j GraphRAG pipeline with custom components."""

        # 1. Text Splitter
        self.text_splitter = HierarchicalTextSplitter(max_chunk_tokens=512)

        # 2. Chunk Embedder
        self.chunk_embedder = TextChunkEmbedder(
            embedder=self.embedder  # Wrap existing OpenAI embedder
        )

        # 3. Schema Builder
        self.schema = build_pe_schema()

        # 4. Entity Extractor
        self.entity_extractor = PEEntityExtractor(
            llm=self._get_claude_llm(),
            neo4j_driver=self.neo4j.driver,
            schema=self.schema
        )

        # 5. Entity Resolver
        self.entity_resolver = CrossTypeFuzzyResolver(
            driver=self.neo4j.driver,
            threshold=0.85
        )

        # 6. KG Writer
        self.kg_writer = Neo4jWriter(
            driver=self.neo4j.driver
        )

        # Note: NOT using SimpleKGPipeline - building custom pipeline
        # to maintain control over folder hierarchy, classification, etc.

    async def ingest_folder(
        self,
        dataroom_id: str,
        folder_path: Path,
        recursive: bool = True
    ) -> IngestionResult:
        """
        Ingest all documents from a folder.

        Maintains same signature as current IngestionPipeline.ingest_folder()
        """
        start_time = datetime.utcnow()
        self._progress.status = "running"
        self._progress.start_time = start_time

        try:
            # 1. Discovery (keep current logic)
            files = self._discover_files(folder_path, recursive)
            self._update_progress(total_files=len(files))

            # 2. Create folder nodes (keep current logic)
            await self._create_folder_nodes(dataroom_id, folder_path, files)

            # 3. Generate folder narratives (keep current logic)
            if self.generate_narratives:
                await self._generate_folder_narratives(dataroom_id, folder_path, files)

            # 4. Process documents in parallel (NEW - Neo4j GraphRAG)
            results = await self._process_documents_parallel(
                dataroom_id, folder_path, files
            )

            # 5. Entity relationships (NEW - post-processing)
            if self.extract_entities:
                await self._create_entity_relationships(dataroom_id)

            # 6. Update stats (keep current logic)
            await self._update_dataroom_stats(dataroom_id)

            return IngestionResult(
                success=True,
                dataroom_id=dataroom_id,
                documents_processed=len(files),
                chunks_created=sum(r.chunks for r in results),
                entities_extracted=sum(r.entities for r in results),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds()
            )

        except Exception as e:
            self._progress.status = "failed"
            self._progress.error_message = str(e)
            return IngestionResult(
                success=False,
                dataroom_id=dataroom_id,
                errors=[str(e)],
                duration_seconds=(datetime.utcnow() - start_time).total_seconds()
            )

    async def _process_documents_parallel(
        self,
        dataroom_id: str,
        root_path: Path,
        files: list[Path]
    ) -> list[DocumentResult]:
        """Process multiple documents in parallel using Neo4j GraphRAG."""

        semaphore = asyncio.Semaphore(self.max_concurrent_files)

        async def process_single(file_path: Path):
            async with semaphore:
                return await self._process_document_graphrag(
                    dataroom_id, root_path, file_path
                )

        tasks = [process_single(fp) for fp in files]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_document_graphrag(
        self,
        dataroom_id: str,
        root_path: Path,
        file_path: Path
    ):
        """
        Process a single document using Neo4j GraphRAG pipeline.

        Flow:
        1. Parse with Unstructured.io
        2. Classify document type
        3. Split into chunks (with section/page metadata)
        4. Generate embeddings
        5. Extract entities (if enabled)
        6. Resolve duplicates
        7. Write to Neo4j
        """

        # 1. Parse document
        elements = await self.unstructured.parse_document(file_path)

        # 2. Classify document type
        full_text = "\n".join([el.text for el in elements])
        doc_type, confidence = self.classifier.classify(full_text, file_path.name)

        # 3. Create Document node
        folder_id = self._get_folder_id(dataroom_id, root_path, file_path)
        document = Document.create(
            dataroom_id=dataroom_id,
            folder_id=folder_id,
            full_path=str(file_path),
            filename=file_path.name,
            file_size=file_path.stat().st_size,
            file_type=file_path.suffix.lower(),
            doc_type=doc_type,
            doc_type_confidence=confidence,
            content_hash=self._compute_file_hash(file_path),
            ingestion_status="processing"
        )
        self.neo4j.create_node(labels=["Document"], properties=document.to_neo4j_properties())

        # 4. Split text with section/page metadata
        chunks = await self.text_splitter.run(elements)

        # Add document context to chunks
        for chunk in chunks:
            chunk.metadata.update({
                'document_id': document.id,
                'dataroom_id': dataroom_id,
                'doc_type': doc_type.value,
                'filename': file_path.name
            })

        # 5. Generate embeddings
        chunks_with_embeddings = await self.chunk_embedder.run(chunks)

        # 6. Extract entities (if enabled)
        entities = []
        if self.extract_entities:
            extraction_result = await self.entity_extractor.run(chunks_with_embeddings)
            entities = extraction_result.entities

            # 7. Resolve duplicates
            resolved_entities = []
            for entity in entities:
                canonical = await self.entity_resolver.resolve(entity)
                resolved_entities.append(canonical or entity)

        # 8. Write to Neo4j (chunks + entities + relationships)
        await self.kg_writer.run(
            document=document,
            chunks=chunks_with_embeddings,
            entities=resolved_entities if self.extract_entities else []
        )

        # 9. Create NEXT relationships between chunks
        await self._create_chunk_sequence(chunks_with_embeddings)

        # 10. Update document status
        self.neo4j.execute_write(
            "MATCH (d:Document {id: $id}) SET d.ingestion_status = 'completed', d.chunk_count = $count",
            {"id": document.id, "count": len(chunks)}
        )

        return DocumentResult(
            document_id=document.id,
            chunks=len(chunks),
            entities=len(entities) if self.extract_entities else 0
        )
```

---

### 3.5 Modified Retrieval Layer

**Purpose:** Update `GraphRAGRetriever` to work with flattened chunk metadata

**Changes Required:**

```python
class GraphRAGRetriever:
    """
    Hybrid retrieval adapted for flat chunk structure.

    Changes:
    - Section traversal replaced with metadata filtering
    - Page relationships replaced with page_number property
    - Maintains entity-based expansion
    """

    def _vector_search(self, query_embedding, dataroom_id, top_k, doc_type_filter):
        # Updated query: no Section/Page nodes
        query = """
        CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding)
        YIELD node AS chunk, score
        WHERE chunk.dataroom_id = $dataroom_id
        MATCH (doc:Document {id: chunk.document_id})
        WHERE CASE WHEN $doc_types IS NOT NULL THEN doc.doc_type IN $doc_types ELSE true END
        RETURN chunk.id AS chunk_id,
               chunk.text AS text,
               score,
               chunk.document_id AS document_id,
               doc.full_path AS document_path,
               doc.filename AS document_name,
               chunk.page_number AS page,
               chunk.section_title AS section_title
        ORDER BY score DESC
        LIMIT $top_k
        """
        # Same result structure as before!
```

**HierarchyExpander Updates:**

```python
class HierarchyExpander:
    """
    Adapted for flat chunk structure with metadata-based expansion.
    """

    def expand_chunks(self, chunk_ids, max_expansion=20):
        # Updated: Use section_title property instead of Section nodes
        query = """
        MATCH (c:Chunk)
        WHERE c.id IN $chunk_ids

        // Get siblings by section title + document
        MATCH (sibling:Chunk)
        WHERE sibling.section_title = c.section_title
          AND sibling.document_id = c.document_id
          AND sibling.id <> c.id

        // Get sequential neighbors (NEXT still exists!)
        OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
        OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)

        // Get entity co-occurrence (unchanged)
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(other:Chunk)
        WHERE other.id NOT IN $chunk_ids

        WITH collect(DISTINCT sibling) + collect(DISTINCT next) +
             collect(DISTINCT prev) + collect(DISTINCT other) AS expanded
        UNWIND expanded AS exp
        WHERE exp IS NOT NULL

        RETURN DISTINCT exp.id AS chunk_id,
               exp.text AS text,
               exp.document_id AS document_id,
               exp.page_number AS page,
               exp.section_title AS section_title
        LIMIT $max_expansion
        """
        # Still works! Metadata-based instead of graph-based
```

**Key Insight:** Most retrieval queries still work! Just replace:
- `(section:Section)` → `chunk.section_title`
- `(page:Page {page_number: X})` → `chunk.page_number = X`
- Section hierarchy traversal → metadata filtering

---

## 4. Migration Plan

### 4.1 Pre-Migration Validation

**Required Before Starting:**

1. ✅ **Install Neo4j GraphRAG package:**
   ```bash
   pip install neo4j-graphrag
   pip install rapidfuzz  # For fuzzy matching
   pip install spacy  # Optional for semantic matching
   ```

2. ✅ **Validate context7 MCP connectivity:**
   - Ensure context7 MCP is configured and accessible
   - Test validation endpoint
   - Confirm project metadata is synchronized

3. ✅ **Create feature branch:**
   ```bash
   git checkout -b claude/neo4j-graphrag-migration
   ```

4. ✅ **Backup test data:**
   - Export sample data room from Neo4j
   - Document expected outputs for validation

### 4.2 Implementation Phases

**Phase 1: Foundation (Parallel Work Streams)**

**Stream 1A: Schema & Models**
- Update `models.py` to add chunk metadata fields
- Remove Page/Section model classes (keep for backward compat initially)
- Add migration utilities

**Stream 1B: Custom Components**
- Implement `HierarchicalTextSplitter`
- Implement `EntityNameNormalizer` (port from canonicalizer)
- Implement `CrossTypeFuzzyResolver`

**Stream 1C: PE Entity Extraction**
- Adapt EXTRACTION_PROMPT for Neo4j format
- Implement `PEEntityExtractor`
- Build PE schema definition

**Phase 2: Core Pipeline (Sequential - Depends on Phase 1)**

**Stream 2A: Pipeline Orchestrator**
- Implement `Neo4jGraphRAGPipeline`
- Port folder creation logic
- Port narrative generation logic

**Stream 2B: Document Processing**
- Implement `_process_document_graphrag`
- Integrate all custom components
- Handle error cases

**Phase 3: Retrieval Updates (Parallel)**

**Stream 3A: Core Retrieval**
- Update `GraphRAGRetriever._vector_search()`
- Update `GraphRAGRetriever._fulltext_search()`
- Update result merging logic

**Stream 3B: Hierarchy Expansion**
- Update `HierarchyExpander.expand_chunks()`
- Update `HierarchyExpander.get_document_context()`
- Update `HierarchyExpander.get_section_chunks()` (metadata-based)

**Phase 4: Integration & Testing**
- Integration tests
- End-to-end validation
- Performance benchmarking

### 4.3 Parallelization Strategy

**Maximum Parallelization Map:**

```
Phase 1 (3 parallel streams):
├─ Stream 1A: Schema & Models (Agent A)
├─ Stream 1B: Custom Components (Agent B)
└─ Stream 1C: PE Entity Extraction (Agent C)

Phase 2 (2 parallel streams after Phase 1):
├─ Stream 2A: Pipeline Orchestrator (Agent A)
└─ Stream 2B: Document Processing (Agent B)

Phase 3 (2 parallel streams after Phase 2):
├─ Stream 3A: Core Retrieval (Agent A)
└─ Stream 3B: Hierarchy Expansion (Agent B)

Phase 4 (Single stream after Phase 3):
└─ Integration & Testing (All agents collaborate)
```

**Agent Coordination:**
- Each agent works on separate files (no merge conflicts)
- Shared interfaces defined upfront
- Integration points validated after each phase

---

## 5. Implementation Requirements

### 5.1 Development Environment

**Required:**
- Python 3.11+
- Neo4j 5.x running (Docker or local)
- API keys configured:
  - `ANTHROPIC_API_KEY` (Claude for entity extraction)
  - `OPENAI_API_KEY` (embeddings)
  - `UNSTRUCTURED_API_KEY` (document parsing)
- context7 MCP configured and running

**New Dependencies:**
```python
# requirements.txt additions
neo4j-graphrag>=0.10.0
rapidfuzz>=3.0.0
spacy>=3.7.0  # Optional
```

### 5.2 Code Standards

**Import Structure:**
```python
# Standard library
import asyncio
from pathlib import Path
from typing import Optional

# Third-party
from neo4j_graphrag.experimental import ...

# Local
from ..database.neo4j_client import Neo4jClient
from ..database.models import Chunk, Document
```

**Naming Conventions:**
- Neo4j GraphRAG files: `neo4j_*.py` prefix
- Custom components: Inherit from Neo4j base classes
- Maintain backward compatibility: Use same function signatures

**Documentation:**
```python
class Component:
    """
    Brief description.

    Neo4j GraphRAG Integration:
        - Base class: neo4j_graphrag.components.X
        - Custom behavior: Y

    Backward Compatibility:
        - Replaces: OldClass
        - Maintains: Same interface
    """
```

### 5.3 Testing Requirements

**Unit Tests:**
- Each custom component tested in isolation
- Mock Neo4j driver responses
- Test PE prompt with sample extractions
- Test cross-type resolver with known duplicates

**Integration Tests:**
- Full pipeline test with small document set
- Validate all node types and relationships created
- Check embedding dimensions
- Verify entity linking

**Validation Tests (Against Current System):**
- Run same documents through both pipelines
- Compare:
  - Chunk count
  - Entity count and types
  - Retrieval results for same queries
  - Citation accuracy

**Performance Tests:**
- Measure ingestion speed (docs/minute)
- Measure entity extraction time
- Measure retrieval latency
- Compare to current system benchmarks

### 5.4 context7 MCP Validation

**Implementation Plan Validation:**

Before starting implementation, the architect must:

1. **Submit Plan to context7 MCP:**
   ```python
   # Pseudo-code for context7 integration
   plan = {
       "phases": [...],
       "components": [...],
       "dependencies": [...],
       "risks": [...]
   }

   validation_result = context7_mcp.validate_plan(
       project="data-room-graph",
       plan=plan,
       check_dependencies=True,
       check_conflicts=True
   )

   if not validation_result.approved:
       raise ValidationError(validation_result.issues)
   ```

2. **Required Validations:**
   - ✅ No conflicts with existing codebase
   - ✅ All dependencies are available
   - ✅ Component interfaces are compatible
   - ✅ No circular dependencies
   - ✅ Test coverage plan is sufficient

3. **Ongoing Validation:**
   - Validate after each phase completion
   - Update context7 with progress
   - Flag any deviations from plan

---

## 6. Testing Strategy

### 6.1 Test Data

**Preparation:**
```bash
# Create test data room
python scripts/create_test_dataroom.py

# Backup for validation
python scripts/export_dataroom.py --id test_dataroom_001 --output /tmp/test_baseline.json
```

**Test Documents:**
- 1 LPA (Limited Partnership Agreement)
- 1 PPM (Private Placement Memorandum)
- 1 Quarterly Report
- Total ~100 pages, expect ~200 chunks, ~50 entities

### 6.2 Validation Criteria

**Functional Parity:**
- [ ] All documents ingested successfully
- [ ] Chunk count within 5% of baseline
- [ ] Entity count within 10% of baseline
- [ ] Entity types match baseline (Fund, Manager, etc.)
- [ ] No cross-type duplicates (validate key entities)
- [ ] Retrieval returns same top-10 results for test queries
- [ ] Citations point to correct page numbers
- [ ] Folder narratives generated successfully

**Performance Parity:**
- [ ] Ingestion speed within 20% of baseline
- [ ] Retrieval latency within 10% of baseline
- [ ] Memory usage acceptable

**Quality Improvements (Expected):**
- [ ] Faster entity resolution (RapidFuzz vs SequenceMatcher)
- [ ] Better entity extraction (schema-guided)
- [ ] Cleaner codebase (fewer custom components)

### 6.3 Test Execution

**Unit Tests:**
```bash
pytest tests/unit/test_hierarchical_splitter.py -v
pytest tests/unit/test_pe_entity_extractor.py -v
pytest tests/unit/test_cross_type_resolver.py -v
```

**Integration Tests:**
```bash
pytest tests/integration/test_neo4j_pipeline.py -v
```

**Validation Tests:**
```bash
# Run both pipelines on same data
python scripts/validate_migration.py \
    --test-dataroom test_dataroom_001 \
    --baseline /tmp/test_baseline.json \
    --output /tmp/migration_validation_report.json

# Check report
python scripts/check_validation_report.py /tmp/migration_validation_report.json
```

---

## 7. Rollback Plan

### 7.1 Rollback Triggers

**Automatic Rollback If:**
- Integration tests fail with >20% error rate
- Performance degradation >50%
- Data corruption detected
- Blocking bugs discovered

**Manual Rollback If:**
- Validation shows >30% accuracy degradation
- Critical feature broken
- Unresolvable Neo4j GraphRAG bug

### 7.2 Rollback Procedure

**Step 1: Stop Ingestion**
```python
# Mark feature flag as disabled
config.set("use_neo4j_graphrag", False)
```

**Step 2: Revert Code**
```bash
git checkout main
git branch -D claude/neo4j-graphrag-migration
```

**Step 3: Restore Data (if needed)**
```bash
# Restore from backup
python scripts/restore_dataroom.py --backup /tmp/test_baseline.json
```

**Step 4: Verify**
```bash
pytest tests/integration/test_current_pipeline.py -v
```

### 7.3 Post-Rollback Analysis

- Document failure reason
- Identify gaps in validation
- Plan remediation
- Re-attempt migration after fixes

---

## 8. Appendices

### 8.1 Neo4j GraphRAG References

**Documentation:**
- [GraphRAG for Python](https://neo4j.com/docs/neo4j-graphrag-python/current/)
- [API Documentation](https://neo4j.com/docs/neo4j-graphrag-python/current/api.html)
- [GitHub Repository](https://github.com/neo4j/neo4j-graphrag-python)

**Key Blog Posts:**
- [New Cypher AI procedures (Dec 2025)](https://medium.com/neo4j/new-cypher-ai-procedures-6b8c3177d56d)
- [Enhancing Hybrid Retrieval](https://neo4j.com/blog/developer/enhancing-hybrid-retrieval-graphrag-python-package/)
- [Entity Resolved Knowledge Graphs](https://neo4j.com/blog/developer/entity-resolved-knowledge-graphs/)

### 8.2 Component Interface Summary

**HierarchicalTextSplitter:**
```python
Input: list[UnstructuredElement]
Output: list[TextChunk]
Properties Added: section_title, section_path, section_level, page_number, element_type
```

**PEEntityExtractor:**
```python
Input: list[TextChunk]
Output: list[Entity]
Schema: 9 PE entity types (Fund, Manager, Person, etc.)
Prompt: 130-line PE-specific extraction rules
```

**CrossTypeFuzzyResolver:**
```python
Input: Entity
Output: Entity (canonical or None)
Algorithm: RapidFuzz + legal suffix normalization + cross-type search
```

**Neo4jGraphRAGPipeline:**
```python
Input: (dataroom_id, folder_path, recursive)
Output: IngestionResult
Components: Splitter → Embedder → Extractor → Resolver → Writer
```

### 8.3 File Change Summary

**New Files:**
```
src/backend/ingestion/
  neo4j_pipeline.py              (400 lines)
  hierarchical_splitter.py       (250 lines)
  pe_entity_extractor.py         (150 lines)
  cross_type_resolver.py         (200 lines)
  unstructured_parser.py         (100 lines)
  document_classifier_component.py (80 lines)

tests/unit/
  test_hierarchical_splitter.py  (150 lines)
  test_pe_entity_extractor.py    (120 lines)
  test_cross_type_resolver.py    (100 lines)

tests/integration/
  test_neo4j_pipeline.py         (200 lines)

scripts/
  validate_migration.py          (300 lines)
```

**Modified Files:**
```
src/backend/ingestion/
  pipeline.py                    (Add factory method for Neo4j pipeline)

src/backend/retrieval/
  graphrag.py                    (Update queries for flat structure)
  hierarchy.py                   (Update expansion logic)

src/backend/database/
  models.py                      (Add chunk metadata fields)

requirements.txt                 (Add neo4j-graphrag, rapidfuzz)
```

**Deprecated (Keep for Rollback):**
```
src/backend/ingestion/
  chunker.py                     (Logic moved to hierarchical_splitter)

src/backend/extraction/
  entity_extractor.py            (Wrapped by pe_entity_extractor)
  canonicalizer.py               (Logic moved to cross_type_resolver)
  linker.py                      (Replaced by Neo4j auto-linking)
```

### 8.4 Risk Assessment

**High Risk:**
- ❌ Prompt adaptation for Neo4j format may lose accuracy
- ❌ Flattened structure may impact retrieval quality
- **Mitigation:** Extensive validation tests, gradual rollout

**Medium Risk:**
- ⚠️ Neo4j GraphRAG package bugs or breaking changes
- ⚠️ Performance regression on large document sets
- **Mitigation:** Pin package versions, performance benchmarks

**Low Risk:**
- ✅ Cross-type resolver implementation
- ✅ Integration with existing embedder
- ✅ Folder/narrative generation (unchanged)

### 8.5 Success Metrics

**Must Have (Go/No-Go):**
- ✅ All integration tests pass
- ✅ Validation tests show <10% accuracy degradation
- ✅ No data corruption
- ✅ Feature parity with current system

**Should Have (Quality Gates):**
- ✅ Performance within 20% of baseline
- ✅ Code coverage >80%
- ✅ Documentation complete

**Nice to Have (Stretch Goals):**
- ✅ Performance improvement >10%
- ✅ Entity extraction accuracy improvement
- ✅ Cleaner architecture

---

## Implementation Checklist for Claude Code Architect

Before starting implementation:

- [ ] Review entire specification
- [ ] Validate understanding of current system
- [ ] Confirm access to all referenced files
- [ ] Test context7 MCP connectivity
- [ ] Create feature branch
- [ ] Build detailed implementation plan with:
  - [ ] Task breakdown by component
  - [ ] Dependency graph
  - [ ] Parallelization strategy
  - [ ] Time estimates
  - [ ] Risk mitigation steps
- [ ] Submit implementation plan to context7 MCP for validation
- [ ] Get approval before proceeding

After implementation plan approval:

- [ ] Spawn subagents for parallel work streams
- [ ] Coordinate agent work via shared interfaces
- [ ] Validate after each phase
- [ ] Run tests continuously
- [ ] Document any deviations from plan
- [ ] Create final validation report

---

**End of Specification**

**Next Steps:**
1. Claude Code architect reviews this spec
2. Builds detailed implementation plan
3. Validates plan with context7 MCP
4. Executes migration with parallel subagents
5. Validates results and delivers final report
