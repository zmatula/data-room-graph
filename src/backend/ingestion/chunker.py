"""Hierarchy builder from Unstructured.io elements."""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .unstructured import UnstructuredElement
from ..database.models import Chunk, Document

logger = logging.getLogger(__name__)


@dataclass
class HierarchyNode:
    """A node in the document hierarchy tree."""

    element: UnstructuredElement
    children: list["HierarchyNode"] = field(default_factory=list)
    parent: Optional["HierarchyNode"] = None
    depth: int = 0

    @property
    def text(self) -> str:
        """Get the node's text content."""
        return self.element.text

    @property
    def element_type(self) -> str:
        """Get the element type."""
        return self.element.type

    def add_child(self, node: "HierarchyNode"):
        """Add a child node."""
        node.parent = self
        node.depth = self.depth + 1
        self.children.append(node)

    def get_hierarchy_path(self) -> str:
        """Get the path from root to this node.

        Returns:
            String like "0/1/3" representing the path.
        """
        path_parts = []
        node = self
        while node.parent:
            siblings = node.parent.children
            index = siblings.index(node)
            path_parts.append(str(index))
            node = node.parent
        return "/".join(reversed(path_parts)) or "0"


class HierarchyBuilder:
    """Builds a hierarchical structure from Unstructured.io elements."""

    def __init__(self, max_chunk_tokens: int = 512):
        """Initialize the builder.

        Args:
            max_chunk_tokens: Maximum tokens per chunk for splitting.
        """
        self.max_chunk_tokens = max_chunk_tokens
        # Rough estimate: 1 token ≈ 4 characters
        self.max_chunk_chars = max_chunk_tokens * 4

    def build_hierarchy(
        self, elements: list[UnstructuredElement]
    ) -> list[HierarchyNode]:
        """Build a hierarchy tree from flat elements.

        Uses title elements as section headers and nests content underneath.

        Args:
            elements: List of elements from Unstructured.io.

        Returns:
            List of root-level HierarchyNode objects.
        """
        if not elements:
            return []

        roots: list[HierarchyNode] = []
        section_stack: list[HierarchyNode] = []

        for element in elements:
            node = HierarchyNode(element=element)

            if element.is_title:
                # Title elements create new sections
                depth = element.category_depth

                # Pop stack until we find parent at lower depth
                while section_stack and section_stack[-1].element.category_depth >= depth:
                    section_stack.pop()

                if section_stack:
                    # Nest under parent section
                    section_stack[-1].add_child(node)
                else:
                    # Top-level section
                    roots.append(node)

                section_stack.append(node)

            else:
                # Non-title elements attach to current section
                if section_stack:
                    section_stack[-1].add_child(node)
                else:
                    # No section context, add as root
                    roots.append(node)

        return roots

    def flatten_hierarchy(
        self,
        roots: list[HierarchyNode],
        include_parents: bool = True,
    ) -> list[HierarchyNode]:
        """Flatten hierarchy to a list while preserving parent references.

        Args:
            roots: List of root nodes.
            include_parents: Whether to include section/title nodes.

        Returns:
            Flat list of all nodes in document order.
        """
        result = []

        def traverse(node: HierarchyNode):
            if include_parents or not node.element.is_title:
                result.append(node)
            for child in node.children:
                traverse(child)

        for root in roots:
            traverse(root)

        return result

    def create_chunks(
        self,
        elements: list[UnstructuredElement],
        dataroom_id: str,
        document_id: str,
    ) -> list[Chunk]:
        """Create Chunk objects from elements with hierarchy info.

        Args:
            elements: List of elements from Unstructured.io.
            dataroom_id: ID of the containing data room.
            document_id: ID of the source document.

        Returns:
            List of Chunk objects ready for Neo4j.
        """
        # Build hierarchy
        roots = self.build_hierarchy(elements)
        nodes = self.flatten_hierarchy(roots)

        chunks = []
        chunk_id_map = {}  # element_id -> chunk_id

        for i, node in enumerate(nodes):
            # Determine parent chunk ID
            parent_chunk_id = None
            if node.parent:
                parent_element_id = node.parent.element.element_id
                parent_chunk_id = chunk_id_map.get(parent_element_id)

            # Create chunk
            chunk = Chunk.create(
                dataroom_id=dataroom_id,
                document_id=document_id,
                text=node.text,
                hierarchy_path=node.get_hierarchy_path(),
                order=i,
                element_type=node.element_type,
                hierarchy_level=node.depth,
                page_start=node.element.page_number,
                page_end=node.element.page_number,
                offset_start=node.element.offset_start,
                offset_end=node.element.offset_end,
                coordinates=node.element.coordinates,
                parent_chunk_id=parent_chunk_id,
            )

            chunk_id_map[node.element.element_id] = chunk.id
            chunks.append(chunk)

        # Split oversized chunks
        chunks = self._split_large_chunks(chunks, dataroom_id, document_id)

        logger.info(
            f"Created {len(chunks)} chunks from {len(elements)} elements"
        )
        return chunks

    def _split_large_chunks(
        self,
        chunks: list[Chunk],
        dataroom_id: str,
        document_id: str,
    ) -> list[Chunk]:
        """Split chunks that exceed the maximum size.

        Args:
            chunks: List of chunks to process.
            dataroom_id: Data room ID.
            document_id: Document ID.

        Returns:
            List of chunks with large ones split.
        """
        result = []

        for i, chunk in enumerate(chunks):
            # Validate chunk has required attributes
            if not hasattr(chunk, 'hierarchy_path'):
                logger.error(
                    f"Chunk {i} missing 'hierarchy_path': "
                    f"type={type(chunk).__module__}.{type(chunk).__name__}, "
                    f"attrs={[a for a in dir(chunk) if not a.startswith('_')]}"
                )
                continue
            if not isinstance(chunk, Chunk):
                logger.error(
                    f"Chunk {i} is not a Chunk model instance: "
                    f"type={type(chunk).__module__}.{type(chunk).__name__}"
                )
            if len(chunk.text) <= self.max_chunk_chars:
                result.append(chunk)
                continue

            # Add the original chunk as a container node (it will have CONTAINS relationships to sub-chunks)
            # Clear the text to avoid duplication - it becomes a structural parent node
            container_chunk = Chunk(
                id=chunk.id,
                dataroom_id=dataroom_id,
                document_id=document_id,
                text="",  # Empty text as this is now a container
                hierarchy_path=chunk.hierarchy_path,
                element_type=chunk.element_type,
                hierarchy_level=chunk.hierarchy_level,
                sequence_order=chunk.sequence_order,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                offset_start=chunk.offset_start,
                offset_end=chunk.offset_end,
                parent_chunk_id=chunk.parent_chunk_id,
            )
            result.append(container_chunk)

            # Split on paragraph boundaries
            paragraphs = chunk.text.split("\n\n")
            current_text = ""
            sub_order = 0
            current_offset = chunk.offset_start

            for para in paragraphs:
                if len(current_text) + len(para) + 2 > self.max_chunk_chars:
                    if current_text:
                        # Create sub-chunk
                        sub_text = current_text.strip()
                        sub_chunk = Chunk.create(
                            dataroom_id=dataroom_id,
                            document_id=document_id,
                            text=sub_text,
                            hierarchy_path=f"{chunk.hierarchy_path}/{sub_order}",
                            order=chunk.sequence_order * 1000 + sub_order,
                            element_type=chunk.element_type,
                            hierarchy_level=chunk.hierarchy_level + 1,
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                            offset_start=current_offset,
                            offset_end=current_offset + len(sub_text),
                            parent_chunk_id=chunk.id,
                        )
                        result.append(sub_chunk)
                        current_offset += len(current_text)
                        sub_order += 1
                        current_text = ""

                current_text += para + "\n\n"

            # Add remaining text
            if current_text.strip():
                sub_text = current_text.strip()
                sub_chunk = Chunk.create(
                    dataroom_id=dataroom_id,
                    document_id=document_id,
                    text=sub_text,
                    hierarchy_path=f"{chunk.hierarchy_path}/{sub_order}",
                    order=chunk.sequence_order * 1000 + sub_order,
                    element_type=chunk.element_type,
                    hierarchy_level=chunk.hierarchy_level + 1,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    offset_start=current_offset,
                    offset_end=current_offset + len(sub_text),
                    parent_chunk_id=chunk.id,
                )
                result.append(sub_chunk)

        return result

    def get_section_context(
        self, chunk: Chunk, all_chunks: list[Chunk]
    ) -> str:
        """Get the section context for a chunk (parent titles).

        Args:
            chunk: The chunk to get context for.
            all_chunks: All chunks in the document.

        Returns:
            Concatenated parent section titles.
        """
        context_parts = []
        chunk_map = {c.id: c for c in all_chunks}

        current_id = chunk.parent_chunk_id
        while current_id:
            parent = chunk_map.get(current_id)
            if parent and parent.element_type == "Title":
                context_parts.append(parent.text)
            current_id = parent.parent_chunk_id if parent else None

        return " > ".join(reversed(context_parts))
