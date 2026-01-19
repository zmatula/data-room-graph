"""Pydantic models for Neo4j nodes and relationships."""

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Document classification types for PE data rooms."""

    LPA = "LPA"
    SIDE_LETTER = "Side letter"
    PPM = "PPM"
    SUBSCRIPTION_AGREEMENT = "Subscription agreement"
    CAPITAL_CALL_NOTICE = "Capital call notice"
    QUARTERLY_REPORT = "Quarterly report"
    ANNUAL_REPORT = "Annual report"
    FINANCIAL_STATEMENTS = "Financial statements"
    FEE_SCHEDULE = "Fee schedule"
    TRACK_RECORD = "Track record/marketing deck"
    OTHER = "Other"
    UNKNOWN = "Unknown"


class EntityType(str, Enum):
    """Entity types extracted from documents."""

    FUND = "Fund"
    MANAGER = "Manager"
    PERSON = "Person"
    VEHICLE = "Vehicle"
    SERVICE_PROVIDER = "ServiceProvider"
    INVESTOR = "Investor"


def generate_id(node_type: str, data_room_id: str, unique_content: str) -> str:
    """Generate a deterministic ID for a node.

    Args:
        node_type: Type of node (e.g., 'chunk', 'document').
        data_room_id: ID of the containing data room.
        unique_content: Content to hash for uniqueness.

    Returns:
        Formatted ID string: {node_type}:{data_room_id}:{hash}
    """
    content_hash = hashlib.sha256(unique_content.encode()).hexdigest()[:12]
    return f"{node_type}:{data_room_id}:{content_hash}"


class BaseNode(BaseModel):
    """Base model for all Neo4j nodes."""

    id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_neo4j_properties(self) -> dict:
        """Convert model to Neo4j-compatible properties dict."""
        data = self.model_dump(exclude_none=True)
        # Convert non-primitive types to Neo4j-compatible values
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Enum):
                data[key] = value.value
            elif isinstance(value, dict):
                # Neo4j doesn't support nested maps, serialize to JSON
                data[key] = json.dumps(value)
        return data


class DataRoom(BaseNode):
    """A data room containing documents for a PE fund."""

    name: str
    description: Optional[str] = None
    source_path: Optional[str] = None
    document_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0
    ingestion_status: str = "pending"  # pending, in_progress, completed, failed

    @classmethod
    def create(cls, name: str, **kwargs) -> "DataRoom":
        """Create a new DataRoom with auto-generated ID."""
        dr_id = f"dataroom:{hashlib.sha256(name.encode()).hexdigest()[:12]}"
        return cls(id=dr_id, name=name, **kwargs)


class Folder(BaseNode):
    """A folder within a data room."""

    dataroom_id: str
    name: str
    path: str
    parent_folder_id: Optional[str] = None
    document_count: int = 0
    narrative: Optional[str] = None  # Claude-generated narrative describing folder contents

    @classmethod
    def create(cls, dataroom_id: str, path: str, name: str, **kwargs) -> "Folder":
        """Create a new Folder with auto-generated ID."""
        folder_id = generate_id("folder", dataroom_id, path)
        return cls(id=folder_id, dataroom_id=dataroom_id, name=name, path=path, **kwargs)


class Document(BaseNode):
    """A document within a folder."""

    dataroom_id: str
    folder_id: str
    filename: str
    full_path: str
    file_size: int = 0
    file_type: str = ""
    doc_type: DocumentType = DocumentType.UNKNOWN
    doc_type_confidence: float = 0.0
    page_count: Optional[int] = None
    chunk_count: int = 0
    content_hash: Optional[str] = None
    ingestion_status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None

    @classmethod
    def create(
        cls, dataroom_id: str, folder_id: str, full_path: str, filename: str, **kwargs
    ) -> "Document":
        """Create a new Document with auto-generated ID."""
        content_hash = kwargs.get("content_hash", full_path)
        doc_id = generate_id("document", dataroom_id, f"{full_path}:{content_hash}")
        return cls(
            id=doc_id,
            dataroom_id=dataroom_id,
            folder_id=folder_id,
            full_path=full_path,
            filename=filename,
            **kwargs,
        )


class Page(BaseNode):
    """A page within a document."""

    dataroom_id: str
    document_id: str
    page_number: int

    @classmethod
    def create(cls, dataroom_id: str, document_id: str, page_number: int, **kwargs) -> "Page":
        """Create a new Page with auto-generated ID."""
        unique_content = f"{document_id}:page:{page_number}"
        page_id = generate_id("page", dataroom_id, unique_content)
        return cls(
            id=page_id,
            dataroom_id=dataroom_id,
            document_id=document_id,
            page_number=page_number,
            **kwargs,
        )


class Section(BaseNode):
    """A section within a document, derived from Title elements."""

    dataroom_id: str
    document_id: str
    title: str
    description: Optional[str] = None  # Claude-generated description for top-level sections
    sequence_order: int = 0  # Order within the document
    hierarchy_level: int = 0  # 0 = top-level, higher = nested
    hierarchy_path: str = "0"  # Path from root like "0/1/3"
    parent_section_id: Optional[str] = None
    page_id: Optional[str] = None  # Primary page where section starts

    @classmethod
    def create(
        cls,
        dataroom_id: str,
        document_id: str,
        title: str,
        hierarchy_path: str,
        order: int,
        **kwargs,
    ) -> "Section":
        """Create a new Section with auto-generated ID."""
        unique_content = f"{document_id}:section:{hierarchy_path}:{order}:{title[:50]}"
        section_id = generate_id("section", dataroom_id, unique_content)
        return cls(
            id=section_id,
            dataroom_id=dataroom_id,
            document_id=document_id,
            title=title,
            hierarchy_path=hierarchy_path,
            sequence_order=order,
            **kwargs,
        )


class Chunk(BaseNode):
    """A chunk of text from a document."""

    dataroom_id: str
    document_id: str
    section_id: Optional[str] = None  # Parent section
    page_id: Optional[str] = None  # Primary page where chunk appears
    text: str
    element_type: str = "NarrativeText"  # Title, NarrativeText, ListItem, Table, etc.
    sequence_order: int = 0  # Order within section
    offset_start: int = 0
    offset_end: int = 0
    coordinates: Optional[dict] = None  # {x, y, width, height}
    embedding: Optional[list[float]] = None
    token_count: Optional[int] = None

    @classmethod
    def create(
        cls,
        dataroom_id: str,
        document_id: str,
        text: str,
        order: int,
        **kwargs,
    ) -> "Chunk":
        """Create a new Chunk with auto-generated ID."""
        section_id = kwargs.get("section_id", "")
        unique_content = f"{document_id}:{section_id}:{order}:{text[:100]}"
        chunk_id = generate_id("chunk", dataroom_id, unique_content)
        return cls(
            id=chunk_id,
            dataroom_id=dataroom_id,
            document_id=document_id,
            text=text,
            sequence_order=order,
            **kwargs,
        )


class Entity(BaseNode):
    """Base class for extracted entities."""

    dataroom_id: str
    entity_type: EntityType
    name: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    embedding: Optional[list[float]] = None
    mention_count: int = 0
    confidence: float = 1.0

    @classmethod
    def create(
        cls, dataroom_id: str, entity_type: EntityType, canonical_name: str, **kwargs
    ) -> "Entity":
        """Create a new Entity with auto-generated ID."""
        entity_id = generate_id(
            entity_type.value.lower(), dataroom_id, canonical_name.lower()
        )
        name = kwargs.pop("name", canonical_name)
        return cls(
            id=entity_id,
            dataroom_id=dataroom_id,
            entity_type=entity_type,
            name=name,
            canonical_name=canonical_name,
            **kwargs,
        )


class Fund(Entity):
    """A private equity fund entity."""

    vintage_year: Optional[int] = None
    strategy: Optional[str] = None
    target_size: Optional[float] = None
    currency: str = "USD"

    def __init__(self, **data):
        data["entity_type"] = EntityType.FUND
        super().__init__(**data)


class Manager(Entity):
    """A fund manager/GP entity."""

    aum: Optional[float] = None
    headquarters: Optional[str] = None
    founded_year: Optional[int] = None

    def __init__(self, **data):
        data["entity_type"] = EntityType.MANAGER
        super().__init__(**data)


class Person(Entity):
    """A person entity."""

    title: Optional[str] = None
    role: Optional[str] = None
    organization: Optional[str] = None

    def __init__(self, **data):
        data["entity_type"] = EntityType.PERSON
        super().__init__(**data)


class Vehicle(Entity):
    """An investment vehicle entity (SPV, Feeder, Blocker, etc.)."""

    vehicle_type: Optional[str] = None  # SPV, Feeder, Blocker, Aggregator
    jurisdiction: Optional[str] = None
    parent_fund_id: Optional[str] = None

    def __init__(self, **data):
        data["entity_type"] = EntityType.VEHICLE
        super().__init__(**data)


class ServiceProvider(Entity):
    """A service provider entity."""

    provider_type: Optional[str] = None  # Auditor, Administrator, Counsel, Custodian
    services: list[str] = Field(default_factory=list)

    def __init__(self, **data):
        data["entity_type"] = EntityType.SERVICE_PROVIDER
        super().__init__(**data)


class Investor(Entity):
    """An investor/LP entity."""

    investor_type: Optional[str] = None  # Pension, Endowment, Family Office, etc.
    commitment_amount: Optional[float] = None
    currency: str = "USD"

    def __init__(self, **data):
        data["entity_type"] = EntityType.INVESTOR
        super().__init__(**data)


class Evidence(BaseModel):
    """Evidence linking an entity mention to its source."""

    chunk_id: str
    document_id: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    offset_start: int
    offset_end: int
    coordinates: Optional[dict] = None
    snippet: str
    confidence: float = 1.0

    def to_neo4j_properties(self) -> dict:
        """Convert to Neo4j relationship properties."""
        return self.model_dump(exclude_none=True)
