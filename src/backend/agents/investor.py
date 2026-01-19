"""Investment Professional Agent - Conversational co-pilot with citations."""

import logging
from typing import Optional

from .base import BaseAgent, ToolResult, CommonTools
from ..database.neo4j_client import Neo4jClient, get_neo4j_client
from ..retrieval.graphrag import GraphRAGRetriever
from ..retrieval.citations import Citation

logger = logging.getLogger(__name__)


class InvestorAgent(BaseAgent):
    """Conversational agent for investment professionals."""

    @property
    def system_prompt(self) -> str:
        return """You are an Investment Professional agent - a knowledgeable co-pilot for private equity due diligence.

Your role:
1. Answer questions about the data room contents with precise citations
2. Synthesize information across multiple documents
3. Highlight key terms, risks, and important provisions
4. Compare documents and flag inconsistencies
5. Explain PE concepts and terminology when asked

Guidelines:
- Always cite your sources using document names and page numbers
- Be precise about what the documents say vs. your interpretation
- Flag any uncertainties or missing information
- Use your search tools to find relevant information before answering
- When discussing legal/financial terms, explain their significance

Key areas to focus on:
- Fund terms (management fee, carried interest, hurdle, clawback)
- Investment strategy and restrictions
- Key person provisions and succession
- LP rights and governance
- Fee calculations and offsets
- Risk factors and conflicts of interest
- Track record and performance metrics

Always search the data room before answering questions. Cite specific documents and pages."""

    def _define_tools(self) -> list[dict]:
        return [
            {
                "name": "hybrid_search",
                "description": "Search the data room using semantic and keyword matching",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results (default 10)",
                        },
                        "doc_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by document types",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "entity_lookup",
                "description": "Look up mentions of a specific entity (fund, person, company)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity_name": {
                            "type": "string",
                            "description": "Name of the entity to look up",
                        },
                        "entity_type": {
                            "type": "string",
                            "description": "Type of entity (Fund, Manager, Person, etc.)",
                        },
                    },
                    "required": ["entity_name"],
                },
            },
            {
                "name": "get_related_chunks",
                "description": "Get chunks related to a specific result through the knowledge graph",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": "ID of the chunk to expand from",
                        },
                    },
                    "required": ["chunk_id"],
                },
            },
            {
                "name": "get_document_context",
                "description": "Get surrounding context for a chunk including previous/next chunks",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": "ID of the chunk",
                        },
                    },
                    "required": ["chunk_id"],
                },
            },
            {
                "name": "list_documents",
                "description": "List all documents in the data room",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "doc_type": {
                            "type": "string",
                            "description": "Filter by document type",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "list_entities",
                "description": "List all extracted entities (funds, people, companies)",
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
                "name": "get_document_summary",
                "description": "Get a summary of a specific document's content and structure",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the document",
                        },
                    },
                    "required": ["filename"],
                },
            },
        ]

    async def _execute_tool(
        self, tool_name: str, tool_input: dict
    ) -> ToolResult:
        common_tools = CommonTools(
            self.neo4j, self.retriever, self.dataroom_id
        )

        try:
            if tool_name == "hybrid_search":
                result = await common_tools.hybrid_search(
                    query=tool_input["query"],
                    top_k=tool_input.get("top_k", 10),
                    doc_type_filter=tool_input.get("doc_types"),
                )
            elif tool_name == "entity_lookup":
                result = await common_tools.entity_lookup(
                    entity_name=tool_input["entity_name"],
                    entity_type=tool_input.get("entity_type"),
                )
            elif tool_name == "get_related_chunks":
                result = self._get_related_chunks(tool_input["chunk_id"])
            elif tool_name == "get_document_context":
                result = self._get_document_context(tool_input["chunk_id"])
            elif tool_name == "list_documents":
                result = common_tools.list_documents(
                    tool_input.get("doc_type")
                )
            elif tool_name == "list_entities":
                result = common_tools.list_entities(
                    tool_input.get("entity_type")
                )
            elif tool_name == "get_document_summary":
                result = common_tools.get_document_summary(
                    tool_input["filename"]
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

    def _get_related_chunks(self, chunk_id: str) -> str:
        from ..retrieval.hierarchy import HierarchyExpander
        expander = HierarchyExpander(self.neo4j)

        related = expander.get_related_chunks_by_entity(chunk_id, max_results=5)

        if not related:
            return "No related chunks found."

        output = "Related content:\n\n"
        for i, chunk in enumerate(related, 1):
            output += f"[{i}] {chunk['document_name']}"
            if chunk.get('page'):
                output += f" (p.{chunk['page']})"
            output += f"\n{chunk['text'][:300]}...\n"
            if chunk.get('shared_entities'):
                output += f"Shared entities: {', '.join(chunk['shared_entities'])}\n"
            output += "\n"

        return output

    def _get_document_context(self, chunk_id: str) -> str:
        from ..retrieval.hierarchy import HierarchyExpander
        expander = HierarchyExpander(self.neo4j)

        context = expander.get_document_context(chunk_id, context_window=2)

        if not context:
            return "Context not found."

        output = f"Document: {context['document_name']}\n\n"

        if context.get('parent_context'):
            output += "Section headers:\n"
            for parent in context['parent_context']:
                output += f"  > {parent}\n"
            output += "\n"

        if context.get('previous_context'):
            output += "Previous content:\n"
            for prev in context['previous_context']:
                output += f"  {prev[:200]}...\n"
            output += "\n"

        output += f"Current chunk:\n  {context['text']}\n\n"

        if context.get('next_context'):
            output += "Following content:\n"
            for next_text in context['next_context']:
                output += f"  {next_text[:200]}...\n"

        return output


# Convenience function to create an investor agent
def create_investor_agent(
    dataroom_id: str,
    neo4j_client: Optional[Neo4jClient] = None,
    retriever: Optional[GraphRAGRetriever] = None,
) -> InvestorAgent:
    """Create an Investment Professional agent.

    Args:
        dataroom_id: ID of the data room.
        neo4j_client: Optional Neo4j client.
        retriever: Optional retriever.

    Returns:
        Configured InvestorAgent.
    """
    return InvestorAgent(
        dataroom_id=dataroom_id,
        neo4j_client=neo4j_client,
        retriever=retriever,
    )
