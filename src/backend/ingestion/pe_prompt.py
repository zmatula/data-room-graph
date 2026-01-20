"""PE-domain extraction prompt for Neo4j GraphRAG.

This module provides the extraction prompt template adapted for neo4j-graphrag's
LLMEntityRelationExtractor. The prompt uses {text}, {schema}, and {examples}
placeholders as expected by the library.

The prompt is designed to extract PE-domain entities with high accuracy,
including disambiguation rules to prevent common misclassifications.
"""

# Main extraction prompt for PE documents
# Uses {text}, {schema}, {examples} placeholders as required by neo4j-graphrag
PE_EXTRACTION_PROMPT = """You are an expert at extracting structured entities and relationships from private equity fund documents, specifically energy and natural resources focused PE funds.

Your task is to extract entities and their relationships from the provided text according to the schema below.

=== ENTITY TYPE DEFINITIONS ===

FUND - The investment fund vehicle itself
  Indicators: "Fund" in name, roman numerals (I, II, XII), vintage years, "LP" suffix
  Examples: "EnCap Fund XII", "Apollo Investment Fund IX LP", "Blackstone Energy Partners III"
  NOT the management company that runs the fund

MANAGER - The fund manager, general partner, or management company
  Indicators: "Management", "GP", "Advisors", "Capital" (when referring to the firm), "Partners" (the company)
  Examples: "EnCap Investments LP", "Apollo Global Management", "Blackstone Group"
  This is the firm that manages funds, NOT the fund itself

PORTFOLIO_COMPANY - Companies that funds invest in or have invested in
  Context: investments, acquisitions, portfolio holdings, "backed by", exits
  Examples: "Double Eagle Energy", "Grayson Mill Energy", "Centennial Resource Development"
  Often appear in tables listing "Portfolio Companies" or "Investments"

LOCATION - Geographic areas relevant to investments
  Types: Oil/gas basins, shale formations, regions, countries, states
  Examples: "Permian Basin", "Bakken Formation", "Eagle Ford Shale", "Gulf Coast", "Midland Basin"
  NOT a company - these are geographic/geological areas

ASSET - Specific named physical assets or infrastructure
  Types: Wells, properties, facilities, gathering systems, pipelines
  Examples: "Well #123", "Eagle Ford Gathering System", "Delaware Basin Midstream"
  Only extract if specifically named with identifiers or clear asset designation

PERSON - An individual with a name
  Examples: "John Smith", "Jane Doe, Partner", "David Miller, CEO"
  Extract title/role if mentioned

VEHICLE - Legal investment structures and special purpose entities
  Types: SPVs, feeders, blockers, aggregator vehicles
  Examples: "Acme Feeder LP", "XYZ Blocker LLC", "Delaware Aggregator"
  NOT the main fund - these are subsidiary investment structures

SERVICE_PROVIDER - Third-party service providers to the fund
  Types: Auditors, administrators, legal counsel, custodians, placement agents
  Examples: "Deloitte LLP", "Morgan Lewis", "State Street", "Citco Fund Services"

INVESTOR - Limited partners or investors in the fund
  Types: Pension funds, endowments, family offices, sovereign wealth funds
  Examples: "CalPERS", "Harvard Endowment", "Texas Teachers Retirement"

=== DISAMBIGUATION RULES ===

1. MANAGER vs FUND:
   - "EnCap Investments LP" = MANAGER (manages funds)
   - "EnCap Fund XII" = FUND (the investment vehicle)
   - If "X Capital" manages funds, it's MANAGER. If "X Fund" is the vehicle, it's FUND.

2. PORTFOLIO_COMPANY vs FUND:
   - Companies appearing under "Portfolio Companies", "Investments", or "Holdings" = PORTFOLIO_COMPANY
   - Names like "Grayson Mill", "Double Eagle" without "Fund" = likely PORTFOLIO_COMPANY
   - Check context: is this something the fund invests IN, or is it the fund itself?

3. LOCATION vs MANAGER/COMPANY:
   - "Bakken", "Permian", "Eagle Ford" = LOCATION (geographic basins)
   - These are geographic formations, not companies
   - Common basins: Permian, Bakken, Eagle Ford, Marcellus, Utica, DJ Basin, Haynesville

4. When uncertain, use lower confidence scores and check the surrounding context carefully.

=== SCHEMA ===
{schema}

=== EXAMPLES ===
{examples}

=== TEXT TO ANALYZE ===
{text}

Extract all entities and relationships from the text above. For each entity:
- Determine the correct type based on the definitions and disambiguation rules
- Extract relevant properties as defined in the schema
- Identify relationships between entities

Return your response as valid JSON with the following structure:
{{
  "nodes": [
    {{
      "id": "unique_id",
      "label": "EntityType",
      "properties": {{
        "name": "Entity Name",
        ...other properties
      }}
    }}
  ],
  "relationships": [
    {{
      "start_node_id": "source_id",
      "end_node_id": "target_id",
      "type": "RELATIONSHIP_TYPE",
      "properties": {{}}
    }}
  ]
}}

If no entities are found, return: {{"nodes": [], "relationships": []}}
"""

# Simplified prompt for when examples are not available
PE_EXTRACTION_PROMPT_NO_EXAMPLES = """You are an expert at extracting structured entities and relationships from private equity fund documents, specifically energy and natural resources focused PE funds.

Your task is to extract entities and their relationships from the provided text according to the schema below.

=== ENTITY TYPE DEFINITIONS ===

FUND - The investment fund vehicle itself
  Indicators: "Fund" in name, roman numerals (I, II, XII), vintage years, "LP" suffix
  NOT the management company that runs the fund

MANAGER - The fund manager, general partner, or management company
  Indicators: "Management", "GP", "Advisors", "Capital" (firm), "Partners" (company)
  This is the firm that manages funds, NOT the fund itself

PORTFOLIO_COMPANY - Companies that funds invest in
  Context: investments, acquisitions, portfolio holdings, "backed by", exits

LOCATION - Geographic areas: basins, formations, regions, countries
  NOT a company - geographic/geological areas

ASSET - Specific named physical assets: wells, facilities, pipelines
  Only extract if specifically named

PERSON - Individual with a name, extract title/role if mentioned

VEHICLE - SPVs, feeders, blockers, aggregators
  NOT the main fund - subsidiary structures

SERVICE_PROVIDER - Auditors, administrators, counsel, custodians

INVESTOR - LPs: pension funds, endowments, family offices, SWFs

=== DISAMBIGUATION RULES ===

1. MANAGER vs FUND: "X Investments" = MANAGER, "X Fund XII" = FUND
2. PORTFOLIO_COMPANY vs FUND: Check context - invested IN = PORTFOLIO_COMPANY
3. LOCATION vs COMPANY: Basins (Permian, Bakken, Eagle Ford) = LOCATION

=== SCHEMA ===
{schema}

=== TEXT TO ANALYZE ===
{text}

Extract all entities and relationships. Return valid JSON:
{{
  "nodes": [{{"id": "unique_id", "label": "EntityType", "properties": {{"name": "Entity Name"}}}}],
  "relationships": [{{"start_node_id": "id1", "end_node_id": "id2", "type": "REL_TYPE"}}]
}}

If no entities found: {{"nodes": [], "relationships": []}}
"""

# Few-shot examples for better extraction quality
PE_EXTRACTION_EXAMPLES = """
Example 1:
Text: "EnCap Investments L.P. ('EnCap') is a leading provider of venture capital to the independent sector of the oil and gas industry. EnCap Fund XII, L.P. was formed in 2019 with a target size of $6 billion."

Extracted:
{
  "nodes": [
    {"id": "manager_1", "label": "Manager", "properties": {"name": "EnCap Investments L.P."}},
    {"id": "fund_1", "label": "Fund", "properties": {"name": "EnCap Fund XII, L.P.", "vintage_year": 2019, "target_size": 6000}}
  ],
  "relationships": [
    {"start_node_id": "fund_1", "end_node_id": "manager_1", "type": "MANAGED_BY"}
  ]
}

Example 2:
Text: "The Fund's investments include Double Eagle Energy Holdings, a Permian Basin-focused E&P company, and Grayson Mill Energy, which operates in the Bakken Formation."

Extracted:
{
  "nodes": [
    {"id": "portco_1", "label": "PortfolioCompany", "properties": {"name": "Double Eagle Energy Holdings", "industry": "E&P"}},
    {"id": "portco_2", "label": "PortfolioCompany", "properties": {"name": "Grayson Mill Energy"}},
    {"id": "loc_1", "label": "Location", "properties": {"name": "Permian Basin", "location_type": "basin"}},
    {"id": "loc_2", "label": "Location", "properties": {"name": "Bakken Formation", "location_type": "formation"}}
  ],
  "relationships": [
    {"start_node_id": "portco_1", "end_node_id": "loc_1", "type": "LOCATED_IN"},
    {"start_node_id": "portco_2", "end_node_id": "loc_2", "type": "LOCATED_IN"}
  ]
}

Example 3:
Text: "Deloitte LLP serves as the Fund's independent auditor. Morgan, Lewis & Bockius LLP provides legal counsel."

Extracted:
{
  "nodes": [
    {"id": "sp_1", "label": "ServiceProvider", "properties": {"name": "Deloitte LLP", "provider_type": "Auditor"}},
    {"id": "sp_2", "label": "ServiceProvider", "properties": {"name": "Morgan, Lewis & Bockius LLP", "provider_type": "Counsel"}}
  ],
  "relationships": []
}
"""


def get_extraction_prompt(include_examples: bool = True) -> str:
    """Get the appropriate extraction prompt.

    Args:
        include_examples: Whether to use the prompt with examples placeholder.

    Returns:
        Extraction prompt string.
    """
    if include_examples:
        return PE_EXTRACTION_PROMPT
    return PE_EXTRACTION_PROMPT_NO_EXAMPLES


def get_extraction_examples() -> str:
    """Get the few-shot examples for extraction.

    Returns:
        Examples string.
    """
    return PE_EXTRACTION_EXAMPLES
