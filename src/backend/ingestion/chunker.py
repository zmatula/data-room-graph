"""Section and chunk builder from Unstructured.io elements."""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .unstructured import UnstructuredElement
from ..database.models import Section, Chunk

logger = logging.getLogger(__name__)


@dataclass
class SectionNode:
    """A node representing a section in the document hierarchy."""

    element: UnstructuredElement
    children: list["SectionNode"] = field(default_factory=list)
    parent: Optional["SectionNode"] = None
    depth: int = 0
    page_numbers: set[int] = field(default_factory=set)
    content_elements: list[UnstructuredElement] = field(default_factory=list)

    @property
    def title(self) -> str:
        """Get the section title."""
        return self.element.text if self.element else "[Document Content]"

    def add_child(self, node: "SectionNode"):
        """Add a child section."""
        node.parent = self
        node.depth = self.depth + 1
        self.children.append(node)

    def add_content(self, element: UnstructuredElement):
        """Add a content element to this section."""
        self.content_elements.append(element)
        if element.page_number is not None:
            self.page_numbers.add(element.page_number)

    def get_hierarchy_path(self) -> str:
        """Get the path from root to this section.

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


class SectionChunkBuilder:
    """Builds Section and Chunk nodes from Unstructured.io elements."""

    def __init__(self, max_chunk_tokens: int = 512):
        """Initialize the builder.

        Args:
            max_chunk_tokens: Maximum tokens per chunk for splitting.
        """
        self.max_chunk_tokens = max_chunk_tokens
        # Rough estimate: 1 token ~= 4 characters
        self.max_chunk_chars = max_chunk_tokens * 4

    def build_sections(
        self,
        elements: list[UnstructuredElement],
        dataroom_id: str,
        document_id: str,
        page_id_map: dict[int, str],
    ) -> tuple[list[Section], list[SectionNode], dict[str, set[str]]]:
        """Build Section nodes from Title elements.

        Args:
            elements: List of elements from Unstructured.io.
            dataroom_id: ID of the containing data room.
            document_id: ID of the source document.
            page_id_map: Mapping from page number to page ID.

        Returns:
            Tuple of (list of Section models, list of SectionNodes, dict mapping section_id to set of page_ids).
        """
        if not elements:
            return [], [], {}

        # Build section tree
        root_sections: list[SectionNode] = []
        section_stack: list[SectionNode] = []
        has_titles = any(el.is_title for el in elements)

        # If no titles found, create a synthetic root section
        if not has_titles:
            synthetic_element = UnstructuredElement({
                "element_id": f"{document_id}_synthetic_root",
                "type": "Title",
                "text": "[Document Content]",
                "metadata": {},
            })
            root_section = SectionNode(element=synthetic_element)
            root_sections.append(root_section)
            section_stack.append(root_section)

            # Add all content to this synthetic section
            for element in elements:
                root_section.add_content(element)

        else:
            # Process elements and build section tree
            for element in elements:
                if element.is_title:
                    # Title elements create new sections
                    node = SectionNode(element=element)
                    depth = element.category_depth

                    # Add page number if present
                    if element.page_number is not None:
                        node.page_numbers.add(element.page_number)

                    # Pop stack until we find parent at lower depth
                    while section_stack and section_stack[-1].element.category_depth >= depth:
                        section_stack.pop()

                    if section_stack:
                        # Nest under parent section
                        section_stack[-1].add_child(node)
                    else:
                        # Top-level section
                        root_sections.append(node)

                    section_stack.append(node)

                else:
                    # Non-title elements attach to current section
                    if section_stack:
                        section_stack[-1].add_content(element)
                    else:
                        # No section context - create synthetic section if needed
                        if not root_sections:
                            synthetic_element = UnstructuredElement({
                                "element_id": f"{document_id}_synthetic_preamble",
                                "type": "Title",
                                "text": "[Preamble]",
                                "metadata": {},
                            })
                            preamble = SectionNode(element=synthetic_element)
                            root_sections.insert(0, preamble)
                            section_stack.append(preamble)
                        root_sections[0].add_content(element)

        # Flatten section tree and create Section models
        sections: list[Section] = []
        section_nodes: list[SectionNode] = []
        section_page_map: dict[str, set[str]] = {}

        def process_section(node: SectionNode, order: int) -> int:
            hierarchy_path = node.get_hierarchy_path()

            # Determine parent section ID
            parent_section_id = None
            if node.parent:
                # Find parent in sections list
                for s in sections:
                    if s.title == node.parent.title and s.hierarchy_path == node.parent.get_hierarchy_path():
                        parent_section_id = s.id
                        break

            # Determine primary page_id (first/minimum page number)
            primary_page_id = None
            if node.page_numbers:
                first_page_num = min(node.page_numbers)
                primary_page_id = page_id_map.get(first_page_num)

            section = Section.create(
                dataroom_id=dataroom_id,
                document_id=document_id,
                title=node.title,
                hierarchy_path=hierarchy_path,
                order=order,
                hierarchy_level=node.depth,
                parent_section_id=parent_section_id,
                page_id=primary_page_id,
            )

            # Map section to its pages
            page_ids = set()
            for page_num in node.page_numbers:
                if page_num in page_id_map:
                    page_ids.add(page_id_map[page_num])
            section_page_map[section.id] = page_ids

            sections.append(section)
            section_nodes.append(node)

            # Store section_id in node for chunk building
            node._section_id = section.id

            # Process children
            next_order = order + 1
            for child in node.children:
                next_order = process_section(child, next_order)

            return next_order

        order = 0
        for root in root_sections:
            order = process_section(root, order)

        logger.info(f"Built {len(sections)} sections from {len(elements)} elements")
        return sections, section_nodes, section_page_map

    def build_chunks(
        self,
        section_nodes: list[SectionNode],
        dataroom_id: str,
        document_id: str,
        page_id_map: dict[int, str],
    ) -> list[Chunk]:
        """Build Chunk nodes from section content.

        Args:
            section_nodes: List of section nodes with content.
            dataroom_id: Data room ID.
            document_id: Document ID.
            page_id_map: Mapping from page number to page ID.

        Returns:
            List of Chunk objects.
        """
        chunks: list[Chunk] = []
        global_order = 0

        def process_section_chunks(node: SectionNode):
            nonlocal global_order

            section_id = getattr(node, "_section_id", None)

            for element in node.content_elements:
                # Skip empty elements
                if not element.text.strip():
                    continue

                # Get page ID for this element
                page_id = None
                if element.page_number is not None:
                    page_id = page_id_map.get(element.page_number)

                chunk = Chunk.create(
                    dataroom_id=dataroom_id,
                    document_id=document_id,
                    text=element.text,
                    order=global_order,
                    section_id=section_id,
                    page_id=page_id,
                    element_type=element.type,
                    offset_start=element.offset_start,
                    offset_end=element.offset_end,
                    coordinates=element.coordinates,
                )
                chunks.append(chunk)
                global_order += 1

            # Process child sections
            for child in node.children:
                process_section_chunks(child)

        for node in section_nodes:
            # Only process root-level nodes (children will be processed recursively)
            if node.parent is None:
                process_section_chunks(node)

        # Split oversized chunks
        chunks = self._split_large_chunks(chunks, dataroom_id, document_id)

        logger.info(f"Built {len(chunks)} chunks from section content")
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

        for chunk in chunks:
            if len(chunk.text) <= self.max_chunk_chars:
                result.append(chunk)
                continue

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
                            order=chunk.sequence_order * 1000 + sub_order,
                            section_id=chunk.section_id,
                            page_id=chunk.page_id,
                            element_type=chunk.element_type,
                            offset_start=current_offset,
                            offset_end=current_offset + len(sub_text),
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
                    order=chunk.sequence_order * 1000 + sub_order,
                    section_id=chunk.section_id,
                    page_id=chunk.page_id,
                    element_type=chunk.element_type,
                    offset_start=current_offset,
                    offset_end=current_offset + len(sub_text),
                )
                result.append(sub_chunk)

        return result

    def get_section_content_preview(
        self,
        section_node: SectionNode,
        max_chars: int = 500,
    ) -> str:
        """Get a preview of the section content for description generation.

        Args:
            section_node: The section node.
            max_chars: Maximum characters to include.

        Returns:
            Content preview string.
        """
        text_parts = []
        total_chars = 0

        for element in section_node.content_elements:
            if total_chars >= max_chars:
                break
            remaining = max_chars - total_chars
            text_parts.append(element.text[:remaining])
            total_chars += len(element.text)

        return "\n".join(text_parts)


# Backwards compatibility alias
HierarchyBuilder = SectionChunkBuilder
