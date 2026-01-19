"""Enrichment Agent - Gap identification and information needs."""

import logging
from typing import Optional

from .base import BaseAgent, ToolResult, CommonTools
from ..database.neo4j_client import Neo4jClient, get_neo4j_client
from ..retrieval.graphrag import GraphRAGRetriever

logger = logging.getLogger(__name__)


# Standard PE due diligence checklist
PE_DILIGENCE_CHECKLIST = {
    "LPA": {
        "required": True,
        "description": "Limited Partnership Agreement",
        "key_sections": ["Capital Commitments", "Distributions", "Management Fee", "Carried Interest", "Key Man"],
    },
    "PPM": {
        "required": True,
        "description": "Private Placement Memorandum",
        "key_sections": ["Investment Strategy", "Risk Factors", "Track Record", "Team Bios"],
    },
    "Side letter": {
        "required": False,
        "description": "Side letters with MFN provisions",
        "key_sections": ["Fee Discounts", "Co-Investment Rights", "Reporting Requirements"],
    },
    "Subscription agreement": {
        "required": True,
        "description": "Subscription documents and investor qualifications",
        "key_sections": ["Commitment Amount", "Representations", "Tax Forms"],
    },
    "Financial statements": {
        "required": True,
        "description": "Audited financial statements",
        "key_sections": ["Balance Sheet", "Statement of Operations", "Partner Capital"],
    },
    "Fee schedule": {
        "required": True,
        "description": "Fee arrangements and calculations",
        "key_sections": ["Management Fee", "Carried Interest", "Organizational Expenses"],
    },
    "Track record": {
        "required": True,
        "description": "Historical performance data",
        "key_sections": ["IRR", "TVPI", "DPI", "Prior Fund Performance"],
    },
}


class EnrichmentAgent(BaseAgent):
    """Agent for identifying information gaps and generating prompts."""

    @property
    def system_prompt(self) -> str:
        return """You are an Enrichment Agent specialized in identifying information gaps in private equity data rooms.

Your responsibilities:
1. Analyze data room completeness against PE due diligence standards
2. Identify missing document types or information
3. Find entities mentioned but lacking detailed information
4. Generate prompts to help users gather missing information
5. Prioritize gaps based on diligence importance

Standard document types for PE funds:
- LPA (Limited Partnership Agreement) - Required
- PPM (Private Placement Memorandum) - Required
- Side letters - Important for large LPs
- Subscription agreements - Required
- Financial statements (audited) - Required
- Fee schedule - Required
- Track record/performance data - Required
- Quarterly/Annual reports - Expected for active funds
- Capital call notices - Expected for active funds

When identifying gaps:
- Compare against standard PE due diligence requirements
- Note missing sections within documents
- Flag entities that need more information
- Prioritize by investment decision impact

Always use your tools to analyze the current state before making recommendations."""

    def _define_tools(self) -> list[dict]:
        return [
            {
                "name": "get_entity_coverage",
                "description": "Analyze entity extraction coverage across documents",
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
                "name": "get_doc_type_inventory",
                "description": "Get inventory of document types present in the data room",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "check_diligence_completeness",
                "description": "Check completeness against standard PE due diligence requirements",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "identify_entity_gaps",
                "description": "Find entities that are mentioned but lack detailed information",
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
                "name": "search_for_topic",
                "description": "Search data room for a specific topic to check coverage",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic to search for",
                        },
                    },
                    "required": ["topic"],
                },
            },
            {
                "name": "create_information_request",
                "description": "Create a structured request for missing information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Category of information needed",
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Priority level",
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of what's needed",
                        },
                        "suggested_prompt": {
                            "type": "string",
                            "description": "Suggested prompt for obtaining the information",
                        },
                    },
                    "required": ["category", "priority", "description", "suggested_prompt"],
                },
            },
        ]

    async def _execute_tool(
        self, tool_name: str, tool_input: dict
    ) -> ToolResult:
        try:
            if tool_name == "get_entity_coverage":
                result = self._get_entity_coverage(
                    tool_input.get("entity_type")
                )
            elif tool_name == "get_doc_type_inventory":
                result = self._get_doc_type_inventory()
            elif tool_name == "check_diligence_completeness":
                result = self._check_diligence_completeness()
            elif tool_name == "identify_entity_gaps":
                result = self._identify_entity_gaps(
                    tool_input.get("entity_type")
                )
            elif tool_name == "search_for_topic":
                result = await self._search_for_topic(tool_input["topic"])
            elif tool_name == "create_information_request":
                result = self._create_information_request(
                    tool_input["category"],
                    tool_input["priority"],
                    tool_input["description"],
                    tool_input["suggested_prompt"],
                )
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

    def _get_entity_coverage(self, entity_type: Optional[str] = None) -> str:
        query = """
        MATCH (e:Entity {dataroom_id: $dataroom_id})
        WHERE CASE WHEN $entity_type IS NOT NULL THEN e.entity_type = $entity_type ELSE true END
        OPTIONAL MATCH (e)<-[:MENTIONS]-(c:Chunk)
        WITH e.entity_type AS type, count(DISTINCT e) AS entity_count,
             sum(e.mention_count) AS total_mentions
        RETURN type, entity_count, total_mentions
        ORDER BY total_mentions DESC
        """

        results = self.neo4j.execute_read(
            query, {"dataroom_id": self.dataroom_id, "entity_type": entity_type}
        )

        if not results:
            return "No entities found in this data room."

        output = "Entity Coverage Analysis:\n\n"
        for row in results:
            output += f"- {row['type']}: {row['entity_count']} entities, "
            output += f"{row['total_mentions']} total mentions\n"

        return output

    def _get_doc_type_inventory(self) -> str:
        query = """
        MATCH (d:Document {dataroom_id: $dataroom_id})
        WITH d.doc_type AS doc_type, count(*) AS count
        RETURN doc_type, count
        ORDER BY count DESC
        """

        results = self.neo4j.execute_read(
            query, {"dataroom_id": self.dataroom_id}
        )

        output = "Document Type Inventory:\n\n"
        found_types = set()

        for row in results:
            output += f"- {row['doc_type']}: {row['count']} document(s)\n"
            found_types.add(row['doc_type'])

        output += "\nMissing Document Types:\n"
        for doc_type, info in PE_DILIGENCE_CHECKLIST.items():
            if doc_type not in found_types:
                status = "REQUIRED" if info["required"] else "Recommended"
                output += f"- {doc_type} ({status}): {info['description']}\n"

        return output

    def _check_diligence_completeness(self) -> str:
        query = """
        MATCH (d:Document {dataroom_id: $dataroom_id})
        WITH d.doc_type AS doc_type, collect(d.filename) AS files
        RETURN doc_type, files
        """

        results = self.neo4j.execute_read(
            query, {"dataroom_id": self.dataroom_id}
        )

        found_types = {r["doc_type"]: r["files"] for r in results}

        output = "Due Diligence Completeness Check:\n\n"

        complete = []
        missing = []
        partial = []

        for doc_type, info in PE_DILIGENCE_CHECKLIST.items():
            if doc_type in found_types:
                complete.append((doc_type, len(found_types[doc_type])))
            elif info["required"]:
                missing.append((doc_type, info["description"]))
            else:
                partial.append((doc_type, info["description"]))

        output += "Complete:\n"
        for doc_type, count in complete:
            output += f"  [OK] {doc_type} ({count} document(s))\n"

        output += "\nMissing (Required):\n"
        for doc_type, desc in missing:
            output += f"  [MISSING] {doc_type}: {desc}\n"

        output += "\nMissing (Recommended):\n"
        for doc_type, desc in partial:
            output += f"  [RECOMMENDED] {doc_type}: {desc}\n"

        # Calculate completeness score
        required_count = sum(1 for _, info in PE_DILIGENCE_CHECKLIST.items() if info["required"])
        found_required = sum(1 for t, _ in complete if PE_DILIGENCE_CHECKLIST.get(t, {}).get("required", False))
        score = (found_required / required_count * 100) if required_count > 0 else 0

        output += f"\nCompleteness Score: {score:.0f}% of required documents present"

        return output

    def _identify_entity_gaps(self, entity_type: Optional[str] = None) -> str:
        # Find entities with few mentions or missing key attributes
        query = """
        MATCH (e:Entity {dataroom_id: $dataroom_id})
        WHERE CASE WHEN $entity_type IS NOT NULL THEN e.entity_type = $entity_type ELSE true END
        RETURN e.canonical_name AS name,
               e.entity_type AS type,
               e.mention_count AS mentions,
               e.description AS description
        ORDER BY e.mention_count ASC
        LIMIT 20
        """

        results = self.neo4j.execute_read(
            query, {"dataroom_id": self.dataroom_id, "entity_type": entity_type}
        )

        if not results:
            return "No entities found."

        output = "Entities Needing More Information:\n\n"

        for entity in results:
            issues = []
            if entity.get("mentions", 0) < 3:
                issues.append("few mentions")
            if not entity.get("description"):
                issues.append("no description")

            if issues:
                output += f"- {entity['name']} ({entity['type']})\n"
                output += f"  Issues: {', '.join(issues)}\n"
                output += f"  Current mentions: {entity.get('mentions', 0)}\n\n"

        return output

    async def _search_for_topic(self, topic: str) -> str:
        common_tools = CommonTools(
            self.neo4j, self.retriever, self.dataroom_id
        )
        return await common_tools.hybrid_search(topic, top_k=5)

    def _create_information_request(
        self,
        category: str,
        priority: str,
        description: str,
        suggested_prompt: str,
    ) -> str:
        return f"""Information Request Created:

Category: {category}
Priority: {priority.upper()}

Description:
{description}

Suggested Prompt for User:
"{suggested_prompt}"

This request has been logged for follow-up."""
