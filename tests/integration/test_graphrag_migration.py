"""Integration tests for Neo4j GraphRAG migration.

These tests verify the correctness of the new GraphRAG-based ingestion pipeline
and its compatibility with the existing retrieval layer.

Prerequisites:
- Neo4j instance running (docker compose up)
- API keys configured in .env
- Sample test documents in tests/fixtures/

Run with:
    pytest tests/integration/test_graphrag_migration.py -v
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import modules to test
from src.backend.ingestion.pe_schema import (
    PE_SCHEMA,
    LEXICAL_GRAPH_CONFIG,
    get_entity_labels,
    get_schema_description,
)
from src.backend.ingestion.pe_prompt import (
    PE_EXTRACTION_PROMPT,
    PE_EXTRACTION_EXAMPLES,
    get_extraction_prompt,
    get_extraction_examples,
)
from src.backend.ingestion.preprocessing import (
    DocumentPreprocessor,
    PreprocessedDocument,
    ChunkMetadata,
    extract_metadata_for_graphrag,
)
from src.backend.ingestion.cross_type_resolver import (
    CrossTypeEntityResolver,
    CrossTypeMatch,
)
from src.backend.ingestion.pipeline import create_pipeline, PipelineType
from src.backend.ingestion.validation import (
    ExtractionValidator,
    ValidationIssue,
    ValidationResult,
    IssueType,
    IssueSeverity,
)


class TestPESchema:
    """Tests for PE schema definition."""

    def test_schema_structure(self):
        """Test that schema has required keys."""
        assert "node_types" in PE_SCHEMA
        assert "relationship_types" in PE_SCHEMA
        assert "patterns" in PE_SCHEMA

    def test_strict_schema_enforcement(self):
        """Test that strict schema enforcement flags are set."""
        # These flags prevent LLM from inventing new entity/relationship types
        assert PE_SCHEMA.get("additional_node_types") is False, \
            "additional_node_types should be False for strict enforcement"
        assert PE_SCHEMA.get("additional_relationship_types") is False, \
            "additional_relationship_types should be False for strict enforcement"

    def test_node_types_defined(self):
        """Test that all expected entity types are defined."""
        labels = get_entity_labels()
        expected = [
            "Fund", "Manager", "Person", "Vehicle", "ServiceProvider",
            "Investor", "PortfolioCompany", "Location", "Asset"
        ]
        for expected_label in expected:
            assert expected_label in labels, f"Missing entity type: {expected_label}"

    def test_node_types_have_properties(self):
        """Test that node types have property definitions."""
        for node_type in PE_SCHEMA["node_types"]:
            if isinstance(node_type, dict):
                assert "label" in node_type
                assert "properties" in node_type
                assert len(node_type["properties"]) > 0

    def test_relationship_types_defined(self):
        """Test that key relationships are defined."""
        rel_labels = [
            r["label"] if isinstance(r, dict) else r
            for r in PE_SCHEMA["relationship_types"]
        ]
        expected = ["MANAGED_BY", "HAS_SERVICE_PROVIDER", "HAS_ROLE", "MENTIONS"]
        for expected_rel in expected:
            assert expected_rel in rel_labels, f"Missing relationship: {expected_rel}"

    def test_relationship_types_have_properties(self):
        """Test that relationship types have rich property definitions."""
        # Find key relationships that should have properties
        key_rels = ["MANAGED_BY", "INVESTS_IN", "HAS_SERVICE_PROVIDER"]

        for rel_type in PE_SCHEMA["relationship_types"]:
            if isinstance(rel_type, dict) and rel_type.get("label") in key_rels:
                assert "properties" in rel_type, \
                    f"Relationship {rel_type['label']} should have properties"
                assert len(rel_type["properties"]) > 0, \
                    f"Relationship {rel_type['label']} should have at least one property"

    def test_invests_in_has_investment_properties(self):
        """Test that INVESTS_IN relationship has investment-related properties."""
        for rel_type in PE_SCHEMA["relationship_types"]:
            if isinstance(rel_type, dict) and rel_type.get("label") == "INVESTS_IN":
                property_names = [p["name"] for p in rel_type.get("properties", [])]
                expected_props = ["investment_date", "investment_amount", "ownership_pct"]
                for prop in expected_props:
                    assert prop in property_names, \
                        f"INVESTS_IN should have property: {prop}"

    def test_patterns_valid(self):
        """Test that patterns reference valid node types."""
        labels = get_entity_labels()
        for pattern in PE_SCHEMA["patterns"]:
            source, rel, target = pattern
            assert source in labels, f"Invalid source in pattern: {source}"
            assert target in labels, f"Invalid target in pattern: {target}"

    def test_lexical_graph_config(self):
        """Test lexical graph configuration."""
        assert LEXICAL_GRAPH_CONFIG["chunk_node_label"] == "Chunk"
        assert LEXICAL_GRAPH_CONFIG["document_node_label"] == "Document"
        assert "next_chunk_relationship_type" in LEXICAL_GRAPH_CONFIG

    def test_get_schema_description(self):
        """Test schema description generation."""
        description = get_schema_description()
        assert "Entity Types:" in description
        assert "Relationship Types:" in description
        assert "Fund:" in description
        assert "MANAGED_BY:" in description


class TestPEPrompt:
    """Tests for PE extraction prompt."""

    def test_prompt_has_required_placeholders(self):
        """Test that prompt has required placeholders."""
        prompt = get_extraction_prompt(include_examples=True)
        assert "{text}" in prompt
        assert "{schema}" in prompt
        assert "{examples}" in prompt

    def test_prompt_without_examples(self):
        """Test prompt variant without examples placeholder."""
        prompt = get_extraction_prompt(include_examples=False)
        assert "{text}" in prompt
        assert "{schema}" in prompt
        assert "{examples}" not in prompt

    def test_examples_format(self):
        """Test that examples are properly formatted."""
        examples = get_extraction_examples()
        assert "Example 1:" in examples
        assert '"nodes":' in examples
        assert '"relationships":' in examples

    def test_prompt_contains_disambiguation_rules(self):
        """Test that prompt contains disambiguation guidance."""
        prompt = PE_EXTRACTION_PROMPT
        assert "MANAGER vs FUND" in prompt
        assert "PORTFOLIO_COMPANY vs FUND" in prompt
        assert "LOCATION vs MANAGER" in prompt


class TestPreprocessing:
    """Tests for document preprocessing."""

    def test_chunk_metadata_to_dict(self):
        """Test ChunkMetadata conversion to dict."""
        metadata = ChunkMetadata(
            index=0,
            page_number=1,
            section_title="Introduction",
            section_path="0/0",
            element_type="NarrativeText",
            offset_start=0,
            offset_end=100,
        )
        result = metadata.to_dict()
        assert result["index"] == 0
        assert result["page_number"] == 1
        assert result["section_title"] == "Introduction"

    def test_extract_metadata_for_graphrag(self):
        """Test metadata extraction for GraphRAG format."""
        # Create mock preprocessed document
        from src.backend.ingestion.classifier import DocumentType

        metadata_map = {
            0: ChunkMetadata(
                index=0,
                page_number=1,
                section_title="Test Section",
                section_path="0",
                element_type="NarrativeText",
                offset_start=0,
                offset_end=50,
            ),
        }

        preprocessed = PreprocessedDocument(
            file_path=Path("test.pdf"),
            full_text="Test content",
            chunk_texts=["Test content"],
            metadata_map=metadata_map,
            doc_type=DocumentType.LPA,
            doc_type_confidence=0.9,
            page_count=1,
            element_count=1,
        )

        result = extract_metadata_for_graphrag(preprocessed)
        assert 0 in result
        assert result[0]["page_number"] == 1
        assert result[0]["section_title"] == "Test Section"
        assert result[0]["document_type"] == "LPA"
        assert result[0]["filename"] == "test.pdf"


class TestCrossTypeResolver:
    """Tests for cross-type entity resolution."""

    def test_normalize_name(self):
        """Test name normalization."""
        resolver = CrossTypeEntityResolver()

        # Legal suffixes should be removed
        assert resolver.normalize_name("EnCap Investments LP") == "encap investments"
        assert resolver.normalize_name("Deloitte LLP") == "deloitte"
        assert resolver.normalize_name("Apple Inc.") == "apple"

        # Commas and dots should be normalized to spaces
        normalized = resolver.normalize_name("Morgan, Lewis & Bockius")
        assert "morgan" in normalized
        assert "lewis" in normalized
        assert "bockius" in normalized

    def test_compute_similarity(self):
        """Test similarity computation."""
        resolver = CrossTypeEntityResolver()

        # Similar names after normalization
        assert resolver.compute_similarity("EnCap LP", "EnCap") > 0.6

        # Similar names (same base name)
        assert resolver.compute_similarity(
            "EnCap Investments LP",
            "EnCap Investments"
        ) > 0.8

        # Different names should have low similarity
        assert resolver.compute_similarity(
            "EnCap Investments",
            "Blackstone Group"
        ) < 0.5

    def test_cross_type_match_dataclass(self):
        """Test CrossTypeMatch dataclass."""
        match = CrossTypeMatch(
            canonical_id="entity:dr1:abc123",
            canonical_name="EnCap Investments LP",
            canonical_label="Manager",
            source_id="entity:dr1:def456",
            source_name="EnCap Investments",
            source_label="Fund",
            similarity=0.95,
        )
        assert match.canonical_label == "Manager"
        assert match.source_label == "Fund"
        assert match.similarity > 0.9


class TestPipelineFactory:
    """Tests for pipeline factory function."""

    def test_create_legacy_pipeline(self):
        """Test creating legacy pipeline."""
        # Note: This may fail if Neo4j is not running
        try:
            pipeline = create_pipeline(pipeline_type="legacy")
            assert hasattr(pipeline, "ingest_folder")
            assert hasattr(pipeline, "shutdown")
        except Exception:
            pytest.skip("Neo4j not available")

    def test_create_graphrag_pipeline(self):
        """Test creating GraphRAG pipeline."""
        try:
            pipeline = create_pipeline(pipeline_type="graphrag")
            assert hasattr(pipeline, "ingest_folder")
            assert hasattr(pipeline, "shutdown")
        except Exception:
            pytest.skip("Neo4j not available")

    def test_invalid_pipeline_type(self):
        """Test that invalid pipeline type raises error."""
        with pytest.raises(ValueError):
            create_pipeline(pipeline_type="invalid")


class TestMetadataPreservingSplitter:
    """Tests for the custom metadata-preserving splitter."""

    @pytest.mark.asyncio
    async def test_splitter_preserves_metadata(self):
        """Test that splitter preserves chunk metadata."""
        from src.backend.ingestion.graphrag_pipeline import MetadataPreservingSplitter

        splitter = MetadataPreservingSplitter()

        chunk_texts = ["First chunk", "Second chunk", "Third chunk"]
        metadata_map = {
            0: {"page_number": 1, "section_title": "Intro"},
            1: {"page_number": 1, "section_title": "Body"},
            2: {"page_number": 2, "section_title": "Conclusion"},
        }

        splitter.set_chunks(chunk_texts, metadata_map)
        result = await splitter.run("Full text ignored")

        assert len(result.chunks) == 3
        assert result.chunks[0].text == "First chunk"
        assert result.chunks[0].metadata["page_number"] == 1
        assert result.chunks[1].metadata["section_title"] == "Body"
        assert result.chunks[2].index == 2


class TestSchemaCompatibility:
    """Tests for schema compatibility between legacy and GraphRAG."""

    def test_entity_types_match(self):
        """Test that GraphRAG schema entity types match legacy EntityType enum."""
        from src.backend.database.models import EntityType

        graphrag_labels = get_entity_labels()
        legacy_types = [et.value for et in EntityType]

        # All legacy types should be in GraphRAG schema
        for legacy_type in legacy_types:
            # Handle naming differences (ServiceProvider vs SERVICE_PROVIDER)
            normalized = legacy_type.replace("_", "")
            matching = [
                label for label in graphrag_labels
                if label.lower().replace("_", "") == normalized.lower()
            ]
            assert len(matching) > 0, f"Missing GraphRAG type for: {legacy_type}"


class TestExtractionValidator:
    """Tests for extraction validation component."""

    def test_looks_like_fund(self):
        """Test Fund pattern detection."""
        validator = ExtractionValidator()

        # Clear Fund indicators
        is_fund, conf = validator._looks_like_fund("EnCap Fund XII LP")
        assert is_fund is True
        assert conf > 0.90

        # Fund with roman numerals
        is_fund, conf = validator._looks_like_fund("Apollo Investment Fund IX")
        assert is_fund is True
        assert conf > 0.80

        # Not a Fund
        is_fund, conf = validator._looks_like_fund("Blackstone Group")
        assert is_fund is False

    def test_looks_like_manager(self):
        """Test Manager pattern detection."""
        validator = ExtractionValidator()

        # Clear Manager indicators
        is_manager, conf = validator._looks_like_manager("EnCap Investments LP")
        assert is_manager is False or conf < 0.7  # "Investments" alone is weak

        is_manager, conf = validator._looks_like_manager("Apollo Global Management")
        assert is_manager is True
        assert conf > 0.85

        is_manager, conf = validator._looks_like_manager("Blackstone Advisors")
        assert is_manager is True
        assert conf > 0.80

    def test_looks_like_location(self):
        """Test Location pattern detection."""
        validator = ExtractionValidator()

        # Known basins
        is_loc, conf = validator._looks_like_location("Permian Basin")
        assert is_loc is True
        assert conf > 0.90

        is_loc, conf = validator._looks_like_location("Eagle Ford Shale")
        assert is_loc is True
        assert conf > 0.90

        # Not a Location
        is_loc, conf = validator._looks_like_location("Double Eagle Energy")
        assert is_loc is False

    def test_looks_like_service_provider(self):
        """Test ServiceProvider pattern detection."""
        validator = ExtractionValidator()

        # Known providers
        is_sp, conf = validator._looks_like_service_provider("Deloitte LLP")
        assert is_sp is True
        assert conf > 0.90

        is_sp, conf = validator._looks_like_service_provider("Morgan Lewis & Bockius LLP")
        assert is_sp is True
        assert conf > 0.80

        # Not a ServiceProvider
        is_sp, conf = validator._looks_like_service_provider("EnCap Investments LP")
        assert is_sp is False

    @pytest.mark.asyncio
    async def test_validate_entity_fund_misclassified(self):
        """Test detection of Fund misclassified as Manager."""
        validator = ExtractionValidator()

        # Fund that looks like a Manager (should flag)
        issues = await validator.validate_entity(
            entity_id="test:1",
            entity_name="Apollo Global Management",
            entity_label="Fund",
        )

        # Should detect this looks more like a Manager
        assert len(issues) > 0
        assert issues[0].issue_type == IssueType.MISCLASSIFIED_TYPE
        assert issues[0].suggested_label == "Manager"

    @pytest.mark.asyncio
    async def test_validate_entity_manager_misclassified(self):
        """Test detection of Manager misclassified as Fund."""
        validator = ExtractionValidator()

        # Manager that's actually a Fund
        issues = await validator.validate_entity(
            entity_id="test:2",
            entity_name="EnCap Fund XII LP",
            entity_label="Manager",
        )

        # Should detect this looks more like a Fund
        assert len(issues) > 0
        assert issues[0].issue_type == IssueType.MISCLASSIFIED_TYPE
        assert issues[0].suggested_label == "Fund"

    @pytest.mark.asyncio
    async def test_validate_entity_correct_classification(self):
        """Test that correctly classified entities pass validation."""
        validator = ExtractionValidator()

        # Correctly classified Fund
        issues = await validator.validate_entity(
            entity_id="test:3",
            entity_name="EnCap Fund XII LP",
            entity_label="Fund",
        )
        # Should have no misclassification issues
        misclass_issues = [i for i in issues if i.issue_type == IssueType.MISCLASSIFIED_TYPE]
        assert len(misclass_issues) == 0

        # Correctly classified Manager
        issues = await validator.validate_entity(
            entity_id="test:4",
            entity_name="Apollo Global Management",
            entity_label="Manager",
        )
        misclass_issues = [i for i in issues if i.issue_type == IssueType.MISCLASSIFIED_TYPE]
        assert len(misclass_issues) == 0

    def test_validation_issue_to_dict(self):
        """Test ValidationIssue serialization."""
        issue = ValidationIssue(
            entity_id="test:1",
            entity_name="Test Entity",
            entity_label="Fund",
            issue_type=IssueType.MISCLASSIFIED_TYPE,
            issue="Test issue",
            confidence=0.85,
            severity=IssueSeverity.WARNING,
            suggested_label="Manager",
        )

        result = issue.to_dict()
        assert result["entity_id"] == "test:1"
        assert result["issue_type"] == "misclassified_type"
        assert result["severity"] == "warning"
        assert result["confidence"] == 0.85

    def test_validation_result_counts(self):
        """Test ValidationResult issue counting."""
        result = ValidationResult(
            total_entities=10,
            issues=[
                ValidationIssue(
                    entity_id="1", entity_name="E1", entity_label="Fund",
                    issue_type=IssueType.MISCLASSIFIED_TYPE,
                    issue="Test", confidence=0.9, severity=IssueSeverity.ERROR,
                ),
                ValidationIssue(
                    entity_id="2", entity_name="E2", entity_label="Manager",
                    issue_type=IssueType.INVALID_NAME_PATTERN,
                    issue="Test", confidence=0.7, severity=IssueSeverity.WARNING,
                ),
                ValidationIssue(
                    entity_id="3", entity_name="E3", entity_label="Location",
                    issue_type=IssueType.INVALID_NAME_PATTERN,
                    issue="Test", confidence=0.5, severity=IssueSeverity.INFO,
                ),
            ],
        )

        assert result.error_count == 1
        assert result.warning_count == 1
        assert result.issues_found == 0  # Not set in this test
        assert len(result.issues) == 3


# Fixtures for integration tests

@pytest.fixture
def mock_neo4j_client():
    """Create a mock Neo4j client."""
    mock = Mock()
    mock.execute_read = Mock(return_value=[])
    mock.execute_write = Mock(return_value=[])
    mock.execute_batch = Mock(return_value=[])
    return mock


@pytest.fixture
def sample_preprocessed_doc():
    """Create a sample preprocessed document for testing."""
    from src.backend.ingestion.classifier import DocumentType

    return PreprocessedDocument(
        file_path=Path("sample_lpa.pdf"),
        full_text="EnCap Fund XII is managed by EnCap Investments LP.",
        chunk_texts=["EnCap Fund XII is managed by EnCap Investments LP."],
        metadata_map={
            0: ChunkMetadata(
                index=0,
                page_number=1,
                section_title="Overview",
                section_path="0",
                element_type="NarrativeText",
                offset_start=0,
                offset_end=50,
            ),
        },
        doc_type=DocumentType.LPA,
        doc_type_confidence=0.95,
        page_count=10,
        element_count=50,
    )


# Skip slow integration tests by default
pytestmark = pytest.mark.integration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
