"""Data Engineer Agent - Structure optimization and recommendations."""

import logging
from typing import Optional
from dataclasses import dataclass

from .base import BaseAgent, ToolResult, CommonTools
from ..database.neo4j_client import Neo4jClient, get_neo4j_client
from ..retrieval.graphrag import GraphRAGRetriever

logger = logging.getLogger(__name__)


@dataclass
class StructureRecommendation:
    """A recommendation for improving data room structure."""

    type: str  # rename, tag, reclassify, merge
    target: str  # filename or entity
    suggestion: str
    rationale: str
    confidence: float


class DataEngineerAgent(BaseAgent):
    """Agent for data room structure optimization."""

    @property
    def system_prompt(self) -> str:
        return """You are a Data Engineer agent specialized in organizing and optimizing private equity data room structures.

Your responsibilities:
1. Analyze the current data room structure (folders, documents, entities)
2. Identify organizational issues (misclassified documents, inconsistent naming, duplicate entities)
3. Propose specific improvements with clear rationale
4. Help users understand the data room contents

When making recommendations:
- Be specific and actionable
- Explain why each change would help
- Consider PE industry conventions
- Flag potential duplicates or misclassifications
- Suggest tags or metadata improvements

Available document types: LPA, Side letter, PPM, Subscription agreement, Capital call notice, Quarterly report, Annual report, Financial statements, Fee schedule, Track record/marketing deck, Other, Unknown

Always use your tools to gather information before making recommendations."""

    def _define_tools(self) -> list[dict]:
        return [
            {
                "name": "list_folders",
                "description": "List all folders in the data room with document counts",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "list_documents",
                "description": "List documents, optionally filtered by type or folder",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "doc_type": {
                            "type": "string",
                            "description": "Filter by document type",
                        },
                        "folder_path": {
                            "type": "string",
                            "description": "Filter by folder path",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "get_classification_confidence",
                "description": "Get document classification confidence scores",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "min_confidence": {
                            "type": "number",
                            "description": "Minimum confidence threshold (0-1)",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "find_duplicate_entities",
                "description": "Find potential duplicate entities",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity_type": {
                            "type": "string",
                            "description": "Filter by entity type",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "propose_rename",
                "description": "Propose a document rename",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "current_name": {
                            "type": "string",
                            "description": "Current filename",
                        },
                        "new_name": {
                            "type": "string",
                            "description": "Proposed new filename",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Reason for the rename",
                        },
                    },
                    "required": ["current_name", "new_name", "rationale"],
                },
            },
            {
                "name": "propose_reclassification",
                "description": "Propose a document type reclassification",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Document filename",
                        },
                        "current_type": {
                            "type": "string",
                            "description": "Current document type",
                        },
                        "proposed_type": {
                            "type": "string",
                            "description": "Proposed document type",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Reason for reclassification",
                        },
                    },
                    "required": ["filename", "current_type", "proposed_type", "rationale"],
                },
            },
            {
                "name": "analyze_document",
                "description": "Analyze a document's content and structure",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Document filename to analyze",
                        },
                    },
                    "required": ["filename"],
                },
            },
        ]

    async def _execute_tool(
        self, tool_name: str, tool_input: dict
    ) -> ToolResult:
        try:
            if tool_name == "list_folders":
                result = self._list_folders()
            elif tool_name == "list_documents":
                result = self._list_documents(
                    tool_input.get("doc_type"),
                    tool_input.get("folder_path"),
                )
            elif tool_name == "get_classification_confidence":
                result = self._get_classification_confidence(
                    tool_input.get("min_confidence", 0.5),
                )
            elif tool_name == "find_duplicate_entities":
                result = self._find_duplicate_entities(
                    tool_input.get("entity_type"),
                )
            elif tool_name == "propose_rename":
                result = self._propose_rename(
                    tool_input["current_name"],
                    tool_input["new_name"],
                    tool_input["rationale"],
                )
            elif tool_name == "propose_reclassification":
                result = self._propose_reclassification(
                    tool_input["filename"],
                    tool_input["current_type"],
                    tool_input["proposed_type"],
                    tool_input["rationale"],
                )
            elif tool_name == "analyze_document":
                result = self._analyze_document(tool_input["filename"])
            else:
                return ToolResult(
                    tool_use_id=tool_name,
                    content=f"Unknown tool: {tool_name}",
                    is_error=True,
                )

            return ToolResult(tool_use_id=tool_name, content=result)

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return ToolResult(
                tool_use_id=tool_name,
                content=f"Error: {str(e)}",
                is_error=True,
            )

    def _list_folders(self) -> str:
        query = """
        MATCH (f:Folder {dataroom_id: $dataroom_id})
        OPTIONAL MATCH (f)-[:CONTAINS]->(d:Document)
        RETURN f.path AS path,
               f.name AS name,
               count(d) AS doc_count
        ORDER BY f.path
        """

        results = self.neo4j.execute_read(
            query, {"dataroom_id": self.dataroom_id}
        )

        if not results:
            return "No folders found in this data room."

        output = "Folder Structure:\n\n"
        for folder in results:
            output += f"/{folder['path']} ({folder['doc_count']} documents)\n"

        return output

    def _list_documents(
        self,
        doc_type: Optional[str] = None,
        folder_path: Optional[str] = None,
    ) -> str:
        filters = []
        if doc_type:
            filters.append(f"d.doc_type = '{doc_type}'")
        if folder_path:
            filters.append(f"f.path CONTAINS '{folder_path}'")

        where_clause = ""
        if filters:
            where_clause = "WHERE " + " AND ".join(filters)

        query = f"""
        MATCH (d:Document {{dataroom_id: $dataroom_id}})
        MATCH (d)<-[:CONTAINS]-(f:Folder)
        {where_clause}
        RETURN d.filename AS filename,
               d.doc_type AS doc_type,
               d.doc_type_confidence AS confidence,
               f.path AS folder
        ORDER BY f.path, d.filename
        LIMIT 100
        """

        results = self.neo4j.execute_read(
            query, {"dataroom_id": self.dataroom_id}
        )

        if not results:
            return "No documents found matching the criteria."

        output = "Documents:\n\n"
        for doc in results:
            conf = f"{doc['confidence']:.0%}" if doc.get('confidence') else "N/A"
            output += f"- /{doc['folder']}/{doc['filename']}\n"
            output += f"  Type: {doc['doc_type']} (confidence: {conf})\n"

        return output

    def _get_classification_confidence(
        self, min_confidence: float = 0.5
    ) -> str:
        query = """
        MATCH (d:Document {dataroom_id: $dataroom_id})
        WHERE d.doc_type_confidence < $min_confidence OR d.doc_type = 'Unknown'
        RETURN d.filename AS filename,
               d.doc_type AS doc_type,
               d.doc_type_confidence AS confidence
        ORDER BY d.doc_type_confidence
        LIMIT 20
        """

        results = self.neo4j.execute_read(
            query,
            {"dataroom_id": self.dataroom_id, "min_confidence": min_confidence},
        )

        if not results:
            return f"All documents have classification confidence >= {min_confidence:.0%}"

        output = f"Documents with low classification confidence (< {min_confidence:.0%}):\n\n"
        for doc in results:
            conf = f"{doc['confidence']:.0%}" if doc.get('confidence') else "N/A"
            output += f"- {doc['filename']}: {doc['doc_type']} ({conf})\n"

        return output

    def _find_duplicate_entities(
        self, entity_type: Optional[str] = None
    ) -> str:
        # First get all entities
        type_filter = ""
        if entity_type:
            type_filter = f"AND e.entity_type = '{entity_type}'"

        query = f"""
        MATCH (e:Entity {{dataroom_id: $dataroom_id}})
        WHERE true {type_filter}
        RETURN e.id AS id,
               e.canonical_name AS name,
               e.entity_type AS type,
               e.aliases AS aliases
        """

        results = self.neo4j.execute_read(
            query, {"dataroom_id": self.dataroom_id}
        )

        if not results:
            return "No entities found."

        # Simple similarity check
        from ..extraction.canonicalizer import EntityCanonicalizer
        canonicalizer = EntityCanonicalizer(self.neo4j)

        candidates = []
        entities = list(results)
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                similarity = canonicalizer.compute_similarity(
                    e1["name"], e2["name"]
                )
                if similarity >= 0.7:
                    candidates.append({
                        "entity1": e1["name"],
                        "entity2": e2["name"],
                        "type": e1["type"],
                        "similarity": similarity,
                    })

        if not candidates:
            return "No potential duplicate entities found."

        output = "Potential duplicate entities:\n\n"
        for cand in sorted(candidates, key=lambda x: x["similarity"], reverse=True)[:10]:
            output += f"- '{cand['entity1']}' vs '{cand['entity2']}'\n"
            output += f"  Type: {cand['type']}, Similarity: {cand['similarity']:.0%}\n\n"

        return output

    def _propose_rename(
        self, current_name: str, new_name: str, rationale: str
    ) -> str:
        # Store the proposal (in real implementation, save to a queue)
        return f"""Rename Proposal Recorded:
- Current: {current_name}
- Proposed: {new_name}
- Rationale: {rationale}

This proposal has been saved for user review."""

    def _propose_reclassification(
        self,
        filename: str,
        current_type: str,
        proposed_type: str,
        rationale: str,
    ) -> str:
        return f"""Reclassification Proposal Recorded:
- Document: {filename}
- Current Type: {current_type}
- Proposed Type: {proposed_type}
- Rationale: {rationale}

This proposal has been saved for user review."""

    def _analyze_document(self, filename: str) -> str:
        common_tools = CommonTools(
            self.neo4j, self.retriever, self.dataroom_id
        )
        return common_tools.get_document_summary(filename)
