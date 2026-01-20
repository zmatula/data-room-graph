"""Document pre-processing adapter for Neo4j GraphRAG.

This module provides the preprocessing layer that bridges the existing
Unstructured.io parsing with neo4j-graphrag's pipeline requirements.

The preprocessing layer:
1. Runs Unstructured.io parsing to extract document elements
2. Runs document classification
3. Extracts section hierarchy from Title elements
4. Produces output format compatible with neo4j-graphrag's TextSplitter
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .unstructured import UnstructuredClient, UnstructuredElement
from .classifier import DocumentClassifier, DocumentType

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk, preserved through the pipeline."""

    index: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    section_path: Optional[str] = None  # Hierarchy path like "0/1/2"
    element_type: str = "NarrativeText"
    offset_start: int = 0
    offset_end: int = 0
    coordinates: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for neo4j properties."""
        return {
            "index": self.index,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "section_path": self.section_path,
            "element_type": self.element_type,
            "offset_start": self.offset_start,
            "offset_end": self.offset_end,
        }


@dataclass
class PreprocessedDocument:
    """Result of document preprocessing, ready for neo4j-graphrag."""

    file_path: Path
    full_text: str  # Concatenated text for neo4j-graphrag
    chunk_texts: list[str]  # Individual chunk texts
    metadata_map: dict[int, ChunkMetadata]  # Chunk index -> metadata
    doc_type: DocumentType
    doc_type_confidence: float
    page_count: int
    element_count: int
    raw_elements: list[UnstructuredElement] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        """Number of chunks."""
        return len(self.chunk_texts)


@dataclass
class SectionInfo:
    """Information about a document section."""

    title: str
    depth: int
    path: str
    page_number: Optional[int]


class DocumentPreprocessor:
    """Pre-processes documents for neo4j-graphrag pipeline.

    This class handles the first phase of ingestion:
    1. Parse documents with Unstructured.io
    2. Classify document type
    3. Extract section hierarchy
    4. Build metadata map for chunks
    """

    def __init__(
        self,
        unstructured_client: Optional[UnstructuredClient] = None,
        classifier: Optional[DocumentClassifier] = None,
        chunk_separator: str = "\n\n",
    ):
        """Initialize the preprocessor.

        Args:
            unstructured_client: Client for Unstructured.io API.
            classifier: Document classifier.
            chunk_separator: Separator used between chunks in full text.
        """
        self.unstructured = unstructured_client or UnstructuredClient()
        self.classifier = classifier or DocumentClassifier()
        self.chunk_separator = chunk_separator

    async def preprocess(
        self,
        file_path: Path,
        skip_classification: bool = False,
    ) -> PreprocessedDocument:
        """Preprocess a document for neo4j-graphrag.

        Args:
            file_path: Path to the document file.
            skip_classification: If True, skip document classification.

        Returns:
            PreprocessedDocument ready for neo4j-graphrag pipeline.
        """
        logger.info(f"Preprocessing document: {file_path.name}")

        # Parse with Unstructured.io
        elements = await self.unstructured.parse_document(file_path)

        if not elements:
            logger.warning(f"No elements extracted from {file_path.name}")
            return PreprocessedDocument(
                file_path=file_path,
                full_text="",
                chunk_texts=[],
                metadata_map={},
                doc_type=DocumentType.UNKNOWN,
                doc_type_confidence=0.0,
                page_count=0,
                element_count=0,
                raw_elements=[],
            )

        # Build section hierarchy
        section_stack: list[SectionInfo] = []
        current_section: Optional[SectionInfo] = None

        # Process elements and build chunks with metadata
        chunk_texts: list[str] = []
        metadata_map: dict[int, ChunkMetadata] = {}
        chunk_index = 0
        page_numbers: set[int] = set()

        for element in elements:
            # Track page numbers
            if element.page_number is not None:
                page_numbers.add(element.page_number)

            # Handle section hierarchy
            if element.is_title:
                section_info = self._process_title_element(
                    element, section_stack
                )
                current_section = section_info
                continue  # Titles are not chunks themselves

            # Skip empty elements
            if not element.text.strip():
                continue

            # Create chunk metadata
            metadata = ChunkMetadata(
                index=chunk_index,
                page_number=element.page_number,
                section_title=current_section.title if current_section else None,
                section_path=current_section.path if current_section else None,
                element_type=element.type,
                offset_start=element.offset_start,
                offset_end=element.offset_end,
                coordinates=element.coordinates,
            )

            chunk_texts.append(element.text)
            metadata_map[chunk_index] = metadata
            chunk_index += 1

        # Build full text for neo4j-graphrag
        full_text = self.chunk_separator.join(chunk_texts)

        # Classify document
        doc_type = DocumentType.UNKNOWN
        confidence = 0.0
        if not skip_classification and full_text:
            doc_type, confidence = self.classifier.classify(full_text, file_path.name)

        result = PreprocessedDocument(
            file_path=file_path,
            full_text=full_text,
            chunk_texts=chunk_texts,
            metadata_map=metadata_map,
            doc_type=doc_type,
            doc_type_confidence=confidence,
            page_count=len(page_numbers),
            element_count=len(elements),
            raw_elements=elements,
        )

        logger.info(
            f"Preprocessed {file_path.name}: "
            f"{result.chunk_count} chunks, "
            f"{result.page_count} pages, "
            f"type={doc_type.value}"
        )

        return result

    def _process_title_element(
        self,
        element: UnstructuredElement,
        section_stack: list[SectionInfo],
    ) -> SectionInfo:
        """Process a title element and update section hierarchy.

        Args:
            element: Title element from Unstructured.io.
            section_stack: Current section stack.

        Returns:
            SectionInfo for the new section.
        """
        depth = element.category_depth

        # Pop sections at same or higher depth
        while section_stack and section_stack[-1].depth >= depth:
            section_stack.pop()

        # Build hierarchy path
        if section_stack:
            parent_path = section_stack[-1].path
            sibling_count = sum(
                1 for s in section_stack
                if s.path.startswith(parent_path) and s.depth == depth
            )
            path = f"{parent_path}/{sibling_count}"
        else:
            path = str(len([s for s in section_stack if s.depth == 0]))

        section_info = SectionInfo(
            title=element.text,
            depth=depth,
            path=path,
            page_number=element.page_number,
        )

        section_stack.append(section_info)
        return section_info

    async def preprocess_batch(
        self,
        file_paths: list[Path],
        max_concurrent: int = 5,
    ) -> dict[Path, PreprocessedDocument]:
        """Preprocess multiple documents.

        Args:
            file_paths: List of file paths to preprocess.
            max_concurrent: Maximum concurrent operations.

        Returns:
            Dictionary mapping file paths to preprocessed results.
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}

        async def preprocess_with_semaphore(path: Path):
            async with semaphore:
                try:
                    return path, await self.preprocess(path)
                except Exception as e:
                    logger.error(f"Failed to preprocess {path}: {e}")
                    return path, None

        tasks = [preprocess_with_semaphore(path) for path in file_paths]
        completed = await asyncio.gather(*tasks)

        for path, result in completed:
            if result is not None:
                results[path] = result

        return results

    def shutdown(self):
        """Shutdown resources."""
        self.unstructured.shutdown()


def extract_metadata_for_graphrag(
    preprocessed: PreprocessedDocument,
) -> dict[int, dict]:
    """Extract metadata dict suitable for neo4j-graphrag TextChunk.

    This function converts PreprocessedDocument metadata into the format
    expected by neo4j-graphrag's TextChunk metadata field.

    Args:
        preprocessed: Preprocessed document result.

    Returns:
        Dictionary mapping chunk index to metadata dict.
    """
    metadata_dict = {}

    for idx, chunk_meta in preprocessed.metadata_map.items():
        metadata_dict[idx] = {
            "page_number": chunk_meta.page_number,
            "section_title": chunk_meta.section_title,
            "section_path": chunk_meta.section_path,
            "element_type": chunk_meta.element_type,
            "document_type": preprocessed.doc_type.value,
            "filename": preprocessed.file_path.name,
        }

    return metadata_dict
