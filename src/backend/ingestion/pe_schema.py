"""Private Equity entity schema for Neo4j GraphRAG.

This module defines the PE-domain entity schema as a dict-based configuration
compatible with neo4j-graphrag's SimpleKGPipeline and LLMEntityRelationExtractor.

Schema format follows neo4j-graphrag conventions:
- node_types: List of entity types with optional descriptions and properties
- relationship_types: List of relationship types with optional properties
- patterns: List of valid (source, rel, target) tuples
"""

from typing import TypedDict, Optional, Union


class PropertyDef(TypedDict, total=False):
    """Schema property definition."""
    name: str
    type: str  # STRING, INTEGER, FLOAT, BOOLEAN, LIST
    description: Optional[str]


class NodeTypeDef(TypedDict, total=False):
    """Schema node type definition."""
    label: str
    description: Optional[str]
    properties: list[PropertyDef]


class RelationshipTypeDef(TypedDict, total=False):
    """Schema relationship type definition."""
    label: str
    description: Optional[str]
    properties: list[PropertyDef]


# PE Domain Entity Schema
# IMPORTANT: additional_node_types and additional_relationship_types are set to False
# to enforce strict schema adherence and prevent extraction of off-schema entities
PE_SCHEMA: dict = {
    # Strict schema enforcement - prevents LLM from inventing new types
    "additional_node_types": False,
    "additional_relationship_types": False,
    "node_types": [
        {
            "label": "Fund",
            "description": "Investment fund/vehicle - the actual fund entity (NOT the management company). Look for roman numerals (I, II, XII), 'Fund' in name, 'LP' suffix.",
            "properties": [
                {"name": "name", "type": "STRING", "description": "Fund name as it appears"},
                {"name": "vintage_year", "type": "INTEGER", "description": "Year fund was raised"},
                {"name": "strategy", "type": "STRING", "description": "Investment strategy (buyout, growth, venture, etc.)"},
                {"name": "target_size", "type": "FLOAT", "description": "Target fund size in millions"},
                {"name": "currency", "type": "STRING", "description": "Currency (default USD)"},
            ]
        },
        {
            "label": "Manager",
            "description": "Fund manager, general partner, or management company that manages funds. Look for 'Management', 'GP', 'Advisors', 'Capital' (when referring to firm), 'Partners' (company).",
            "properties": [
                {"name": "name", "type": "STRING", "description": "Manager/GP name"},
                {"name": "aum", "type": "FLOAT", "description": "Assets under management in millions"},
                {"name": "headquarters", "type": "STRING", "description": "Headquarters location"},
                {"name": "founded_year", "type": "INTEGER", "description": "Year founded"},
            ]
        },
        {
            "label": "Person",
            "description": "An individual person with a name. Extract title/role if mentioned.",
            "properties": [
                {"name": "name", "type": "STRING", "description": "Full name"},
                {"name": "title", "type": "STRING", "description": "Job title"},
                {"name": "role", "type": "STRING", "description": "Role in organization"},
                {"name": "organization", "type": "STRING", "description": "Associated organization"},
            ]
        },
        {
            "label": "Vehicle",
            "description": "Legal investment structures and SPEs: SPVs, feeders, blockers, aggregators. NOT the main fund - subsidiary structures.",
            "properties": [
                {"name": "name", "type": "STRING", "description": "Vehicle name"},
                {"name": "vehicle_type", "type": "STRING", "description": "Type: SPV, Feeder, Blocker, Aggregator"},
                {"name": "jurisdiction", "type": "STRING", "description": "Legal jurisdiction"},
            ]
        },
        {
            "label": "ServiceProvider",
            "description": "Third-party service providers: auditors, administrators, legal counsel, custodians, placement agents.",
            "properties": [
                {"name": "name", "type": "STRING", "description": "Provider name"},
                {"name": "provider_type", "type": "STRING", "description": "Type: Auditor, Administrator, Counsel, Custodian, Placement Agent"},
                {"name": "services", "type": "LIST", "description": "List of services provided"},
            ]
        },
        {
            "label": "Investor",
            "description": "Limited partners or investors in the fund: pension funds, endowments, family offices, sovereign wealth funds.",
            "properties": [
                {"name": "name", "type": "STRING", "description": "Investor name"},
                {"name": "investor_type", "type": "STRING", "description": "Type: Pension, Endowment, Family Office, SWF, Insurance, etc."},
                {"name": "commitment_amount", "type": "FLOAT", "description": "Commitment amount in millions"},
            ]
        },
        {
            "label": "PortfolioCompany",
            "description": "Companies that funds invest in or have invested in. Look for context: 'investments', 'acquisitions', 'portfolio holdings', 'backed by', exits.",
            "properties": [
                {"name": "name", "type": "STRING", "description": "Company name"},
                {"name": "industry", "type": "STRING", "description": "Industry sector"},
                {"name": "ownership_pct", "type": "FLOAT", "description": "Ownership percentage"},
                {"name": "investment_date", "type": "STRING", "description": "Date of investment"},
                {"name": "exit_date", "type": "STRING", "description": "Date of exit if applicable"},
            ]
        },
        {
            "label": "Location",
            "description": "Geographic areas relevant to investments: oil/gas basins, shale formations, regions, countries, states. NOT a company - geographic/geological areas.",
            "properties": [
                {"name": "name", "type": "STRING", "description": "Location name"},
                {"name": "location_type", "type": "STRING", "description": "Type: basin, formation, region, country, state"},
                {"name": "parent_location", "type": "STRING", "description": "Parent geographic area"},
            ]
        },
        {
            "label": "Asset",
            "description": "Specific named physical assets or infrastructure: wells, properties, facilities, gathering systems, pipelines. Only if specifically named.",
            "properties": [
                {"name": "name", "type": "STRING", "description": "Asset name/identifier"},
                {"name": "asset_type", "type": "STRING", "description": "Type: well, property, facility, gathering system, pipeline"},
                {"name": "location", "type": "STRING", "description": "Location of asset"},
            ]
        },
    ],
    "relationship_types": [
        {
            "label": "MANAGED_BY",
            "description": "Fund managed by Manager/GP",
            "properties": [
                {"name": "role", "type": "STRING", "description": "Role type (GP, Sub-Advisor, etc.)"},
                {"name": "since_year", "type": "INTEGER", "description": "Year management relationship started"},
                {"name": "ownership_pct", "type": "FLOAT", "description": "Ownership percentage of GP stake"},
            ]
        },
        {
            "label": "HAS_SERVICE_PROVIDER",
            "description": "Fund uses ServiceProvider for services",
            "properties": [
                {"name": "service_type", "type": "STRING", "description": "Type of service provided"},
                {"name": "fee_amount", "type": "FLOAT", "description": "Fee amount in millions"},
                {"name": "fee_type", "type": "STRING", "description": "Fee type (fixed, percentage, etc.)"},
                {"name": "contract_start", "type": "STRING", "description": "Contract start date"},
            ]
        },
        {
            "label": "HAS_ROLE",
            "description": "Person has role at Manager organization",
            "properties": [
                {"name": "title", "type": "STRING", "description": "Job title"},
                {"name": "department", "type": "STRING", "description": "Department or team"},
                {"name": "start_date", "type": "STRING", "description": "Role start date"},
                {"name": "is_key_person", "type": "BOOLEAN", "description": "Whether this is a key person"},
            ]
        },
        {
            "label": "INVESTS_IN",
            "description": "Fund invests in PortfolioCompany",
            "properties": [
                {"name": "investment_date", "type": "STRING", "description": "Date of investment"},
                {"name": "investment_amount", "type": "FLOAT", "description": "Investment amount in millions"},
                {"name": "ownership_pct", "type": "FLOAT", "description": "Ownership percentage acquired"},
                {"name": "investment_type", "type": "STRING", "description": "Type: growth, buyout, recap, etc."},
                {"name": "exit_date", "type": "STRING", "description": "Date of exit if applicable"},
                {"name": "exit_multiple", "type": "FLOAT", "description": "Exit multiple (MOIC) if exited"},
                {"name": "exit_type", "type": "STRING", "description": "Exit type: IPO, sale, recap, etc."},
            ]
        },
        {
            "label": "INVESTED_BY",
            "description": "Fund has Investor as LP",
            "properties": [
                {"name": "commitment_amount", "type": "FLOAT", "description": "Commitment amount in millions"},
                {"name": "commitment_date", "type": "STRING", "description": "Date of commitment"},
                {"name": "investor_class", "type": "STRING", "description": "Investor class/tier"},
                {"name": "is_anchor", "type": "BOOLEAN", "description": "Whether anchor investor"},
            ]
        },
        {
            "label": "AFFILIATED_WITH",
            "description": "Vehicle affiliated with Fund",
            "properties": [
                {"name": "affiliation_type", "type": "STRING", "description": "Type: feeder, blocker, SPV, etc."},
                {"name": "jurisdiction", "type": "STRING", "description": "Legal jurisdiction"},
                {"name": "purpose", "type": "STRING", "description": "Purpose of vehicle"},
            ]
        },
        {
            "label": "LOCATED_IN",
            "description": "Asset or PortfolioCompany located in Location",
            "properties": [
                {"name": "location_type", "type": "STRING", "description": "Type: primary, operations, reserves, etc."},
                {"name": "area_acres", "type": "FLOAT", "description": "Acreage if applicable"},
            ]
        },
        {
            "label": "MENTIONS",
            "description": "Chunk mentions entity with evidence",
            "properties": [
                {"name": "confidence", "type": "FLOAT", "description": "Extraction confidence score"},
                {"name": "snippet", "type": "STRING", "description": "Text snippet containing mention"},
                {"name": "offset_start", "type": "INTEGER", "description": "Start offset in chunk"},
                {"name": "offset_end", "type": "INTEGER", "description": "End offset in chunk"},
            ]
        },
    ],
    "patterns": [
        # Fund relationships
        ("Fund", "MANAGED_BY", "Manager"),
        ("Fund", "HAS_SERVICE_PROVIDER", "ServiceProvider"),
        ("Fund", "INVESTS_IN", "PortfolioCompany"),
        ("Fund", "INVESTED_BY", "Investor"),
        # Vehicle relationships
        ("Vehicle", "AFFILIATED_WITH", "Fund"),
        # Person relationships
        ("Person", "HAS_ROLE", "Manager"),
        ("Person", "HAS_ROLE", "Fund"),
        # Location relationships
        ("PortfolioCompany", "LOCATED_IN", "Location"),
        ("Asset", "LOCATED_IN", "Location"),
    ]
}


# Lexical graph configuration to match existing schema
LEXICAL_GRAPH_CONFIG: dict = {
    "chunk_node_label": "Chunk",
    "document_node_label": "Document",
    "next_chunk_relationship_type": "NEXT",
    "chunk_to_document_relationship_type": "FROM_DOCUMENT",
    "node_to_chunk_relationship_type": "MENTIONED_IN",
}


def get_entity_labels() -> list[str]:
    """Get all entity type labels from the schema.

    Returns:
        List of entity label strings.
    """
    return [
        node_type["label"] if isinstance(node_type, dict) else node_type
        for node_type in PE_SCHEMA["node_types"]
    ]


def get_schema_description() -> str:
    """Get a formatted description of the schema for prompts.

    Returns:
        Formatted schema description string.
    """
    lines = ["Entity Types:"]

    for node_type in PE_SCHEMA["node_types"]:
        if isinstance(node_type, dict):
            label = node_type["label"]
            desc = node_type.get("description", "")
            lines.append(f"- {label}: {desc}")
        else:
            lines.append(f"- {node_type}")

    lines.append("\nRelationship Types:")
    for rel_type in PE_SCHEMA["relationship_types"]:
        if isinstance(rel_type, dict):
            label = rel_type["label"]
            desc = rel_type.get("description", "")
            lines.append(f"- {label}: {desc}")
        else:
            lines.append(f"- {rel_type}")

    return "\n".join(lines)
