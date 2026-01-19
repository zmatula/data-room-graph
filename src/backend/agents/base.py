"""Base agent implementation using Claude SDK."""

import logging
import json
from typing import Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import anthropic

from ..config import get_settings
from ..database.neo4j_client import Neo4jClient, get_neo4j_client
from ..retrieval.graphrag import GraphRAGRetriever
from ..retrieval.citations import Citation

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from executing a tool."""

    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class AgentMessage:
    """A message in the agent conversation."""

    role: str  # "user" or "assistant"
    content: str
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Response from an agent run."""

    message: str
    citations: list[Citation]
    tool_calls_made: int = 0
    tokens_used: int = 0


class BaseAgent(ABC):
    """Base class for all agents using Claude SDK."""

    def __init__(
        self,
        dataroom_id: str,
        neo4j_client: Optional[Neo4jClient] = None,
        retriever: Optional[GraphRAGRetriever] = None,
    ):
        """Initialize the agent.

        Args:
            dataroom_id: ID of the data room to work with.
            neo4j_client: Neo4j client.
            retriever: GraphRAG retriever.
        """
        self.dataroom_id = dataroom_id
        self.neo4j = neo4j_client or get_neo4j_client()
        self.retriever = retriever or GraphRAGRetriever(self.neo4j)

        settings = get_settings()
        self.model = settings.claude_model
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        self._conversation: list[dict] = []
        self._tools = self._define_tools()

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for the agent."""
        pass

    @abstractmethod
    def _define_tools(self) -> list[dict]:
        """Define the tools available to this agent."""
        pass

    @abstractmethod
    async def _execute_tool(
        self, tool_name: str, tool_input: dict
    ) -> ToolResult:
        """Execute a tool and return the result."""
        pass

    async def run(
        self,
        user_message: str,
        max_turns: int = 10,
    ) -> AgentResponse:
        """Run the agent with a user message.

        Args:
            user_message: User's input message.
            max_turns: Maximum number of agent turns.

        Returns:
            AgentResponse with the final message and citations.
        """
        self._conversation.append({
            "role": "user",
            "content": user_message,
        })

        tool_calls_made = 0
        total_tokens = 0
        citations = []

        for turn in range(max_turns):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=self.system_prompt,
                tools=self._tools,
                messages=self._conversation,
            )

            total_tokens += response.usage.input_tokens + response.usage.output_tokens

            # Check if we're done
            if response.stop_reason == "end_turn":
                # Extract final text response
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text

                self._conversation.append({
                    "role": "assistant",
                    "content": response.content,
                })

                return AgentResponse(
                    message=final_text,
                    citations=citations,
                    tool_calls_made=tool_calls_made,
                    tokens_used=total_tokens,
                )

            # Process tool calls
            if response.stop_reason == "tool_use":
                # Add assistant message with tool use
                self._conversation.append({
                    "role": "assistant",
                    "content": response.content,
                })

                # Execute each tool
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls_made += 1
                        result = await self._execute_tool(
                            block.name, block.input
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result.content,
                            "is_error": result.is_error,
                        })

                        # Extract citations if present
                        if hasattr(result, 'citations'):
                            citations.extend(result.citations)

                # Add tool results to conversation
                self._conversation.append({
                    "role": "user",
                    "content": tool_results,
                })

        # Max turns reached
        return AgentResponse(
            message="I've reached the maximum number of steps. Please try a more specific request.",
            citations=citations,
            tool_calls_made=tool_calls_made,
            tokens_used=total_tokens,
        )

    def reset_conversation(self):
        """Reset the conversation history."""
        self._conversation = []


class CommonTools:
    """Common tool implementations shared across agents."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        retriever: GraphRAGRetriever,
        dataroom_id: str,
    ):
        """Initialize common tools.

        Args:
            neo4j_client: Neo4j client.
            retriever: GraphRAG retriever.
            dataroom_id: Data room ID.
        """
        self.neo4j = neo4j_client
        self.retriever = retriever
        self.dataroom_id = dataroom_id

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        doc_type_filter: Optional[list[str]] = None,
    ) -> str:
        """Perform hybrid search."""
        response = await self.retriever.retrieve(
            query=query,
            dataroom_id=self.dataroom_id,
            top_k=top_k,
            doc_type_filter=doc_type_filter,
        )

        if not response.results:
            return "No results found for this query."

        results_text = []
        for i, result in enumerate(response.results, 1):
            text = f"[{i}] {result.document_name}"
            if result.page:
                text += f" (p.{result.page})"
            text += f"\n{result.text[:500]}..."
            if result.entities:
                text += f"\nMentions: {', '.join(result.entities[:5])}"
            results_text.append(text)

        return "\n\n".join(results_text)

    async def entity_lookup(
        self,
        entity_name: str,
        entity_type: Optional[str] = None,
    ) -> str:
        """Look up an entity and its mentions."""
        results = await self.retriever.entity_search(
            entity_name=entity_name,
            dataroom_id=self.dataroom_id,
            entity_type=entity_type,
            top_k=10,
        )

        if not results:
            return f"No entity found matching '{entity_name}'."

        output = f"Found mentions of entities matching '{entity_name}':\n\n"
        for i, result in enumerate(results, 1):
            output += f"[{i}] {result.document_name}"
            if result.page:
                output += f" (p.{result.page})"
            output += f"\n{result.text[:300]}...\n\n"

        return output

    def list_documents(
        self,
        doc_type: Optional[str] = None,
    ) -> str:
        """List documents in the data room."""
        type_filter = ""
        if doc_type:
            type_filter = f"AND d.doc_type = '{doc_type}'"

        query = f"""
        MATCH (d:Document {{dataroom_id: $dataroom_id}})
        WHERE true {type_filter}
        RETURN d.filename AS filename,
               d.doc_type AS doc_type,
               d.chunk_count AS chunks,
               d.ingestion_status AS status
        ORDER BY d.filename
        LIMIT 50
        """

        results = self.neo4j.execute_read(
            query, {"dataroom_id": self.dataroom_id}
        )

        if not results:
            return "No documents found in this data room."

        output = "Documents:\n\n"
        for doc in results:
            output += f"- {doc['filename']} [{doc['doc_type']}] ({doc['chunks']} chunks)\n"

        return output

    def list_entities(
        self,
        entity_type: Optional[str] = None,
    ) -> str:
        """List entities in the data room."""
        type_filter = ""
        if entity_type:
            type_filter = f"AND e.entity_type = '{entity_type}'"

        query = f"""
        MATCH (e:Entity {{dataroom_id: $dataroom_id}})
        WHERE true {type_filter}
        RETURN e.canonical_name AS name,
               e.entity_type AS type,
               e.mention_count AS mentions
        ORDER BY e.mention_count DESC
        LIMIT 50
        """

        results = self.neo4j.execute_read(
            query, {"dataroom_id": self.dataroom_id}
        )

        if not results:
            return "No entities found in this data room."

        output = "Entities:\n\n"
        for entity in results:
            output += f"- {entity['name']} ({entity['type']}) - {entity['mentions']} mentions\n"

        return output

    def get_document_summary(self, filename: str) -> str:
        """Get a summary of a specific document."""
        query = """
        MATCH (d:Document {dataroom_id: $dataroom_id})
        WHERE d.filename CONTAINS $filename
        OPTIONAL MATCH (d)-[:HAS_ROOT|CONTAINS*]->(c:Chunk {element_type: 'Title'})
        OPTIONAL MATCH (d)-[:HAS_ROOT|CONTAINS*]->(:Chunk)-[:MENTIONS]->(e:Entity)
        RETURN d.filename AS filename,
               d.doc_type AS doc_type,
               d.chunk_count AS chunks,
               d.page_count AS pages,
               collect(DISTINCT c.text)[..10] AS sections,
               collect(DISTINCT e.name)[..10] AS entities
        LIMIT 1
        """

        results = self.neo4j.execute_read(
            query, {"dataroom_id": self.dataroom_id, "filename": filename}
        )

        if not results:
            return f"Document '{filename}' not found."

        doc = results[0]
        output = f"Document: {doc['filename']}\n"
        output += f"Type: {doc['doc_type']}\n"
        output += f"Chunks: {doc['chunks']}, Pages: {doc.get('pages', 'N/A')}\n\n"

        if doc.get('sections'):
            output += "Sections:\n"
            for section in doc['sections']:
                output += f"  - {section}\n"

        if doc.get('entities'):
            output += "\nKey Entities:\n"
            for entity in doc['entities']:
                output += f"  - {entity}\n"

        return output
