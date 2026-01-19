You are Claude Code acting as the lead engineer and architect. Choose the tech stack unless explicitly constrained. You must use the Context7 MCP server to validate every major step and every major dependency/API usage, and your plan must explicitly show what you validated. This is a proof of concept optimized for best possible output quality; latency and cost are not constraints.


## Specification for Claude Code (Engineer/Architect)

**Project:** Recursive Agentic GraphRAG System for Private Equity Fund Data Rooms (Proof of Concept)
**Primary language:** Python (mandatory)
**Primary datastore:** Neo4j + Neo4j GraphRAG (mandatory; graph-first)
**Hierarchical chunking engine:** Unstructured.io API (mandatory)
**Mandatory validation:** Claude Code must use the Context7 MCP server to validate implementation plan and all library/API usage
**Priority:** Best achievable output quality (accuracy/coverage/user usefulness). Latency/cost are not constraints.
**POC constraint:** Do not implement full auditability or temporal schema evolution yet; keep interfaces/IDs/evidence structures so auditability can be added later.

---

## 1) Purpose and Scope

### 1.1 Objective

Build a Windows desktop application that lets a professional investor create and manage multiple private equity “data rooms” (each is a document corpus), ingest folders of documents (including subfolders), and convert those documents into a **graph-first + vector** hybrid knowledge system in Neo4j that supports:

* Data-room structure navigation (folder → document → hierarchical sections)
* Entity-context navigation (fund, GP, people, vehicles, service providers, etc.)
* Hybrid retrieval (vector + graph traversal) for diligence Q&A
* Agentic workflows: data engineering suggestions, enrichment gap discovery, and an investor co-pilot

### 1.2 Target Users

* Professional investors performing due diligence on private equity funds.

### 1.3 Proof-of-Concept Constraints

* Demo quality > cost/latency.
* Do not build full governance/audit logging now; still implement stable IDs and evidence pointers for later production hardening.

---

## 2) Non-Negotiable Requirements

1. **Python-first** core implementation (ingestion, orchestration, retrieval).
2. **Neo4j is the primary data system** (“graph first”).
3. **Neo4j GraphRAG** patterns must be used for hybrid retrieval (vector + graph).
4. **Hierarchical semantic chunking** must be produced via **Unstructured.io API** and persisted as a hierarchy in Neo4j:

   * Document → Section → Subsection → Subsubsection → …
   * Explicit relationships between all hierarchy levels.
5. **Vectorize graph nodes** (at minimum: chunk nodes; optionally document/entity nodes) and index vectors in Neo4j.
6. **Source traceability**: every answer and every derived assertion must be linkable back to source documents (file path + page/anchor + snippet).
7. **Document classification** for common PE data room doc types (LPA, side letters, PPM, etc.) and use doc type to improve retrieval and extraction.
8. **Windows UI**: multi-data-room management, folder selection + drag/drop ingestion, progress/errors.
9. **Agentic structure per data room** (each data room has its own “namespace” and agents).
10. **Claude Code chooses the tech stack** except where constrained above.
11. **Context7 MCP usage is mandatory**: validate every major dependency, API, and approach; implementation plan must explicitly show what was validated.

---

## 3) Key Workflows

### 3.1 Data Room Management (UI)

* Create / rename / delete data rooms.
* Configure per-room settings: Neo4j connection, embedding model, Unstructured settings, classification policy.
* Dashboard for ingestion status, document inventory, and agent controls.

### 3.2 Ingestion (Folder → Graph + Vectors)

* User points to a folder or drags/drops a folder into a data room.
* System recursively traverses subfolders and ingests supported files.
* Pipeline must produce:

  * Folder/document structure nodes
  * Hierarchical chunk nodes (from Unstructured.io)
  * Classification per document
  * Embeddings for relevant nodes
  * Entity extraction + entity linking (fund/GP/people/etc.) with evidence pointers
  * Vector indexes and retrieval-ready graph

### 3.3 Data Engineer Agent (Structure Optimization)

* Reviews data room folder structure and document organization.
* Proposes improvements (renames, tagging/classification corrections, missing standard docs).
* Can apply edits **only with explicit user approval**.

### 3.4 Enrichment Agent (Gap Identification)

* Identifies missing diligence-critical info and low-confidence areas.
* Produces prioritized user prompts:

  * “Answer this question” or “Provide a document that addresses this.”
* On new info/docs, triggers incremental re-ingestion and graph updates.

### 3.5 Investment Professional Agent (Investor Co-pilot)

* Conversational diligence assistant that:

  * Answers questions grounded in retrieved chunks/entities
  * Uses entity context (Fund/GP/Person/etc.) for better precision
  * Always provides clickable citations that open source docs at the relevant location

---

## 4) System Architecture Requirements (High Level)

### 4.1 Components

1. **Windows Desktop UI**

   * Data room manager, ingestion console, agent console, graph explorer + graph visualization.
2. **Local Backend Service (Python)**

   * Ingestion orchestration, Unstructured.io API calls, classification, embedding, Neo4j upserts, retrieval API.
3. **Neo4j Graph Store**

   * Stores hierarchy, entities, relationships, evidence links, vectors, and vector indexes.
4. **Hybrid Retrieval Layer (GraphRAG)**

   * Vector similarity + graph traversal + hierarchy-aware expansion + doc-type routing.
5. **Citation/Source Resolver**

   * Maps chunks/entities/claims → (file path, page, anchor, offsets) so UI can open sources.

### 4.2 Parallelization Requirements (Two Distinct Kinds)

**A) Runtime parallelism (program execution):**
Parallelize ingestion steps safely (per file/per chunk/per embedding batch) with idempotent keys, transaction batching, and conflict-safe upserts.

**B) Implementation parallelism (building the system):**
Claude Code must use sub-agents in parallel to accelerate development. See §11.3.

---

## 5) Graph Data Model (Graph-First + Entity Context)

### 5.1 Required Node Types (Minimum)

**Data room structure**

* **DataRoom**: id, name, created_at
* **Folder**: id, path, name
* **Document**: id, filename, full_path, hash, doc_type, metadata
* **Chunk** (hierarchical): id, level, title/heading, order_index, text, page_range/anchors, offsets, embedding_vector

**Entity context (surrounding the data room)**

* **Fund**: id, name, vintage (if known), strategy (if known), domicile (if known)
* **Manager/GP**: id, legal_name, aliases
* **Person**: id, name, roles/titles (if known)
* **Vehicle/Entity** (SPV/feeder/blocker/management company): id, legal_name, entity_type, jurisdiction
* **ServiceProvider**: id, name, provider_type (auditor/admin/legal/etc.)
* **LP/Investor** (if present): id, name, type

**Optional high-value nodes (Claude Code may include)**

* **Clause** (Key Person, GP Removal, Fees, MFN, Recycling, etc.)
* **DefinedTerm** (term + definition text)
* **PortfolioCompany/Investment** (only if present in corpus)

### 5.2 Required Relationship Types (Minimum)

**Structure**

* (DataRoom)-[:CONTAINS]->(Folder)
* (Folder)-[:CONTAINS]->(Document)
* (Document)-[:HAS_ROOT]->(Chunk)
* (Chunk)-[:CONTAINS]->(ChunkChild)
* (Chunk)-[:NEXT]->(ChunkSibling)

**Entity context**

* (DataRoom)-[:ABOUT]->(Fund)
* (Fund)-[:MANAGED_BY]->(Manager/GP)
* (Person)-[:AFFILIATED_WITH]->(Manager/GP)
* (Person)-[:HAS_ROLE {role}]->(Fund or Manager/GP)
* (Fund)-[:USES_VEHICLE]->(Vehicle/Entity)
* (Fund)-[:HAS_SERVICE_PROVIDER {provider_type}]->(ServiceProvider)
* (Fund)-[:HAS_INVESTOR]->(LP/Investor) (only if present)
* (Document)-[:MENTIONS]->(Entity) with evidence pointers (chunk_id + offsets/page)
* (Chunk)-[:REFERENCES]->(Chunk or Document) (optional cross-reference detection)
* (Chunk)-[:EVIDENCES]->(Clause/DefinedTerm) (optional structured extraction)

### 5.3 Vectorization Requirements

* Embeddings on Chunk nodes at all meaningful hierarchy levels.
* Optional: Document-level embedding (summary embedding) for routing.
* Optional: Entity node embeddings (name + descriptions + key supporting snippets) to enable “entity-neighbor” discovery.
* Neo4j vector indexes must be created for the embedded node sets used in retrieval.

---

## 6) Hierarchical Semantic Chunking (Mandatory: Unstructured.io API)

### 6.1 Requirements

* Use **Unstructured.io API** to parse documents and produce structure-aware elements suitable for building a hierarchy.
* Preserve:

  * Headings/section titles
  * Ordering
  * Page numbers and/or coordinates when available
  * Offsets/anchors for “open at location” UX

### 6.2 Output Contract to Graph

Claude Code must define a deterministic mapping from Unstructured outputs to:

* Document node
* Chunk hierarchy nodes (levels inferred from headings/structure cues)
* Evidence metadata stored on chunks (page range, element IDs, offsets, coordinates where available)

---

## 7) Document Classification

### 7.1 Required Classes (Initial)

* LPA
* Side letter
* PPM
* Subscription agreement
* Capital call notice
* Quarterly report
* Annual report
* Financial statements
* Fee schedule
* Track record / marketing deck
* Other / Unknown

### 7.2 Usage

* Select chunking/extraction policies (e.g., LPAs emphasize defined terms/fee sections).
* Improve retrieval routing (“query → likely doc types first”).
* Power enrichment agent (“missing standard docs” detection).

---

## 8) Entity Extraction and Linking (Mandatory)

### 8.1 Requirements

* Extract entities (Fund/GP/Person/etc.) from documents and link them to supporting chunks with evidence pointers.
* Implement canonicalization/merging:

  * alias handling (e.g., “ABC Management” vs “ABC Mgmt, LLC”)
  * stable IDs per entity per data room
* Support “entity-first retrieval”:

  * queries that start from Fund/GP/Person nodes and traverse to relevant documents/clauses/chunks.

---

## 9) Windows UI Requirements

### 9.1 Core Screens

* **Data Rooms List**: create/select/manage
* **Data Room Dashboard**: ingestion status, document inventory, classification filters, agent controls
* **Chat/Agent Console**: investment co-pilot + citations with “open source”
* **Graph Explorer**: browse nodes/relationships
* **Graph Navigator (Interactive Visualization)**: required (see below)

### 9.2 Graph Navigator (Mandatory Visual Graph)

Interactive node-link visualization that supports:

* Toggle layers:

  * Structure graph (Folder/Document/Chunk hierarchy)
  * Entity graph (Fund/GP/People/Vehicles/Service providers)
  * Reference/mention edges
* Search + jump to node
* Expand N hops from selected node
* Node details panel:

  * properties
  * top supporting chunks
  * “open source” action for evidence
  * “similar nodes” (vector neighbors)

### 9.3 UX Requirements

* Drag-and-drop folder ingestion into selected data room
* Clear stage-by-stage progress + errors
* One-click open document at cited location

---

## 10) Quality and Acceptance Criteria

### 10.1 Quality Targets (POC)

* High precision for fund terms and governance clauses.
* All answers must include:

  * citations to chunks (and entities where relevant)
  * enough context to verify quickly
  * ability to open source at the location

### 10.2 Minimum Acceptance Criteria

* Multi-data-room UI
* Recursive folder ingestion
* Neo4j graph populated with:

  * structure hierarchy
  * entity context nodes + mention links
  * vectors + vector indexes
* Hybrid retrieval (GraphRAG)
* Investment professional agent with clickable citations
* Data engineer and enrichment agents functional per spec
* Graph visualization usable for navigation

---

## 11) Implementation Rules for Claude Code

### 11.1 Tech Stack Selection

Claude Code must choose and justify (validated via Context7 MCP):

* Windows UI framework (Python-compatible preferred)
* Backend/service layout (local app architecture)
* Embedding model/provider and LLM provider strategy
* Agent framework/orchestrator approach
* Neo4j deployment mode for demo (local vs managed)
* Unstructured.io integration details (auth, endpoints, payloads, rate handling)

### 11.2 Mandatory Context7 MCP Usage

Claude Code must:

* Use Context7 MCP to validate each major subsystem design and each major dependency/API usage.
* In the implementation plan, explicitly list what was validated for:

  * Neo4j vector indexes + GraphRAG retrieval patterns
  * Unstructured.io API usage for hierarchical chunking
  * chosen UI framework and packaging
  * concurrency patterns and Neo4j transaction batching
  * agent framework APIs and tool interfaces

### 11.3 Mandatory Implementation Parallelism (Sub-Agents for Building the System)

Claude Code must split development into parallel sub-agent workstreams, each with a clear contract and integration plan.

Minimum parallel workstreams (Claude Code may adjust):

* Sub-agent A: Neo4j schema + constraints/indexes + GraphRAG retrieval design
* Sub-agent B: Ingestion pipeline + Unstructured.io chunking integration + chunk hierarchy builder
* Sub-agent C: Entity extraction + canonicalization + mention/evidence linking
* Sub-agent D: Windows UI scaffolding + data room management + ingestion UX
* Sub-agent E: Graph Navigator visualization component (interactive)
* Sub-agent F: Agent orchestration + three agent roles + citations resolver

Claude Code must define shared contracts:

* node/relationship schemas, stable IDs, evidence schema
* backend API between UI and ingestion/retrieval services
* integration sequence + tests

---

## 12) Deliverables Claude Code Must Produce

1. Architecture doc (components, data flow, boundaries)
2. Neo4j schema: labels, relationships, constraints, indexes, vector indexes
3. Unstructured.io chunking integration spec: mapping outputs → chunk hierarchy → evidence metadata
4. Classification spec: classes, approach, evaluation checks
5. Entity extraction spec: entity types, canonicalization rules, evidence linking
6. Retrieval spec: hybrid GraphRAG strategy, hierarchy-aware expansion, doc-type routing
7. UI spec: screens, flows, Graph Navigator behaviors
8. Implementation plan: phased milestones + parallel sub-agent plan + integration testing
9. Demo runbook: step-by-step demonstration scenario

---

## 13) Required Implementation Plan Structure (Claude Code Output Format)

Claude Code must output:

* **Phase 0:** Context7 validation checklist + confirmed stack choices
* **Phase 1:** Neo4j schema + UI scaffold + ingestion skeleton
* **Phase 2:** Unstructured.io chunking → hierarchy → embeddings → vector indexes
* **Phase 3:** Entity extraction/linking + evidence + entity-first retrieval paths
* **Phase 4:** GraphRAG hybrid retrieval + citations + source opener integration
* **Phase 5:** Agents (data engineer, enrichment, investment professional)
* **Phase 6:** Graph Navigator polish + demo script + evaluation checks

---

## Paste-ready instruction header (place above this spec when sending to Claude Code)

“You are **Claude Code** acting as the lead engineer and architect. Choose the tech stack unless explicitly constrained. You must use the Context7 MCP server to validate every major step and every major dependency/API usage, and your plan must explicitly show what you validated. This is a proof of concept optimized for best possible output quality; latency and cost are not constraints.”
