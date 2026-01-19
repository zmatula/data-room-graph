# Data Room Graph - Ingestion Package
from .unstructured import UnstructuredClient, UnstructuredElement
from .chunker import SectionChunkBuilder, SectionNode, HierarchyBuilder
from .page_builder import PageBuilder
from .narrative_generator import NarrativeGenerator
from .classifier import DocumentClassifier
from .embedder import Embedder, get_embedder
from .pipeline import IngestionPipeline, IngestionProgress, IngestionResult

# Backwards compatibility alias
HierarchyNode = SectionNode

__all__ = [
    "UnstructuredClient",
    "UnstructuredElement",
    "SectionChunkBuilder",
    "SectionNode",
    "HierarchyBuilder",  # Backwards compatibility
    "HierarchyNode",  # Backwards compatibility
    "PageBuilder",
    "NarrativeGenerator",
    "DocumentClassifier",
    "Embedder",
    "get_embedder",
    "IngestionPipeline",
    "IngestionProgress",
    "IngestionResult",
]
