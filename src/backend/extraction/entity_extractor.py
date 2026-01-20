"""Entity extraction using Claude LLM."""

import logging
import json
import re
from typing import Any, Optional
from dataclasses import dataclass, field

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..database.models import (
    EntityType, Entity, Fund, Manager, Person, Vehicle, ServiceProvider, Investor,
    PortfolioCompany, Location, Asset, generate_id, generate_entity_id
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """An entity extracted from text."""

    name: str
    entity_type: EntityType
    confidence: float
    context: str  # Surrounding text for evidence
    offset_start: int
    offset_end: int
    attributes: dict[str, Any] = field(default_factory=dict)


# Extraction prompt template with PE-domain-specific definitions
EXTRACTION_PROMPT = """You are an expert at extracting structured entities from private equity fund documents, specifically energy and natural resources focused PE funds.

Extract all entities from the following text. For each entity, provide:
- name: The entity's name as it appears in the text
- type: One of the entity types defined below
- confidence: Your confidence in this extraction (0.0-1.0)
- context: The sentence or phrase containing the entity
- attributes: Any relevant attributes mentioned

{context_section}

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

=== TEXT TO ANALYZE ===
```
{text}
```

Respond with a JSON array of extracted entities:
```json
[
  {{
    "name": "Entity Name",
    "type": "EntityType",
    "confidence": 0.95,
    "context": "The surrounding sentence...",
    "attributes": {{"key": "value"}}
  }}
]
```

If no entities are found, return an empty array: []
"""


class EntityExtractor:
    """Extracts entities from text using Claude."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize the extractor.

        Args:
            api_key: Anthropic API key. Defaults to settings.
            model: Claude model name. Defaults to settings.
        """
        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.claude_model

        if not self.api_key:
            logger.warning(
                "No Anthropic API key configured. "
                "Set ANTHROPIC_API_KEY in .env"
            )

        self._client: Optional[anthropic.Anthropic] = None

    @property
    def client(self) -> anthropic.Anthropic:
        """Get the Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def extract_entities(
        self,
        text: str,
        chunk_offset: int = 0,
        doc_type: Optional[str] = None,
        section_title: Optional[str] = None,
        element_type: Optional[str] = None,
    ) -> list[ExtractedEntity]:
        """Extract entities from text.

        Args:
            text: Text to extract entities from.
            chunk_offset: Offset of this chunk in the document.
            doc_type: Type of document (e.g., "LPA", "Quarterly report").
            section_title: Title of the section containing this text.
            element_type: Element type (e.g., "Table", "NarrativeText").

        Returns:
            List of extracted entities.
        """
        if not text.strip():
            return []

        # Truncate if too long
        max_chars = 10000
        if len(text) > max_chars:
            text = text[:max_chars]

        # Build context section for the prompt
        context_parts = []
        if doc_type:
            context_parts.append(f"Document Type: {doc_type}")
        if section_title:
            context_parts.append(f"Section: {section_title}")
        if element_type:
            context_parts.append(f"Element Type: {element_type}")

        if context_parts:
            context_section = "=== DOCUMENT CONTEXT ===\n" + "\n".join(context_parts) + "\n"
        else:
            context_section = ""

        prompt = EXTRACTION_PROMPT.format(text=text, context_section=context_section)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )

            # Validate response has content
            if not response.content or len(response.content) == 0:
                logger.warning("Entity extraction returned empty response")
                return []

            content = response.content[0].text
            if not content:
                logger.warning("Entity extraction returned empty text content")
                return []

            entities = self._parse_response(content, text, chunk_offset)
            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    def _parse_response(
        self,
        response: str,
        original_text: str,
        chunk_offset: int,
    ) -> list[ExtractedEntity]:
        """Parse the LLM response into ExtractedEntity objects.

        Args:
            response: Raw LLM response.
            original_text: Original text for offset calculation.
            chunk_offset: Offset of chunk in document.

        Returns:
            List of extracted entities.
        """
        # Extract JSON from response
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            return []

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse entity extraction response as JSON")
            return []

        entities = []
        for item in data:
            try:
                # Validate entity type
                type_str = item.get("type", "").strip()
                entity_type = self._map_entity_type(type_str)
                if not entity_type:
                    continue

                name = item.get("name", "").strip()
                if not name:
                    continue

                # Find entity position in text
                offset_start, offset_end = self._find_entity_offset(
                    name, original_text, chunk_offset
                )

                entity = ExtractedEntity(
                    name=name,
                    entity_type=entity_type,
                    confidence=float(item.get("confidence", 0.8)),
                    context=item.get("context", ""),
                    offset_start=offset_start,
                    offset_end=offset_end,
                    attributes=item.get("attributes", {}),
                )
                entities.append(entity)

            except Exception as e:
                logger.warning(f"Failed to parse entity: {e}")
                continue

        return entities

    def _map_entity_type(self, type_str: str) -> Optional[EntityType]:
        """Map string to EntityType enum.

        Args:
            type_str: Entity type as string.

        Returns:
            EntityType or None if invalid.
        """
        type_mapping = {
            # Fund types
            "fund": EntityType.FUND,
            # Manager types
            "manager": EntityType.MANAGER,
            "gp": EntityType.MANAGER,
            "general partner": EntityType.MANAGER,
            "management company": EntityType.MANAGER,
            # Person types
            "person": EntityType.PERSON,
            "individual": EntityType.PERSON,
            # Vehicle types
            "vehicle": EntityType.VEHICLE,
            "spv": EntityType.VEHICLE,
            "feeder": EntityType.VEHICLE,
            "blocker": EntityType.VEHICLE,
            # Service provider types
            "serviceprovider": EntityType.SERVICE_PROVIDER,
            "service provider": EntityType.SERVICE_PROVIDER,
            "service_provider": EntityType.SERVICE_PROVIDER,
            "auditor": EntityType.SERVICE_PROVIDER,
            "administrator": EntityType.SERVICE_PROVIDER,
            "counsel": EntityType.SERVICE_PROVIDER,
            # Investor types
            "investor": EntityType.INVESTOR,
            "lp": EntityType.INVESTOR,
            "limited partner": EntityType.INVESTOR,
            # Portfolio company types
            "portfolio_company": EntityType.PORTFOLIO_COMPANY,
            "portfoliocompany": EntityType.PORTFOLIO_COMPANY,
            "portfolio company": EntityType.PORTFOLIO_COMPANY,
            "company": EntityType.PORTFOLIO_COMPANY,
            "portco": EntityType.PORTFOLIO_COMPANY,
            # Location types
            "location": EntityType.LOCATION,
            "geography": EntityType.LOCATION,
            "basin": EntityType.LOCATION,
            "region": EntityType.LOCATION,
            "formation": EntityType.LOCATION,
            # Asset types
            "asset": EntityType.ASSET,
            "property": EntityType.ASSET,
            "well": EntityType.ASSET,
            "facility": EntityType.ASSET,
            "infrastructure": EntityType.ASSET,
        }
        return type_mapping.get(type_str.lower().strip())

    def _find_entity_offset(
        self,
        name: str,
        text: str,
        chunk_offset: int,
    ) -> tuple[int, int]:
        """Find the offset of an entity name in text.

        Args:
            name: Entity name to find.
            text: Text to search in.
            chunk_offset: Base offset of the chunk.

        Returns:
            Tuple of (start_offset, end_offset).
        """
        # Try exact match first
        idx = text.find(name)
        if idx >= 0:
            return chunk_offset + idx, chunk_offset + idx + len(name)

        # Try case-insensitive match
        idx = text.lower().find(name.lower())
        if idx >= 0:
            return chunk_offset + idx, chunk_offset + idx + len(name)

        # Fallback to chunk boundaries
        return chunk_offset, chunk_offset + len(name)

    def extract_entities_batch(
        self,
        texts: list[tuple[str, int]],  # (text, offset) pairs
    ) -> list[list[ExtractedEntity]]:
        """Extract entities from multiple texts.

        Args:
            texts: List of (text, offset) tuples.

        Returns:
            List of entity lists, one per input text.
        """
        results = []
        for text, offset in texts:
            entities = self.extract_entities(text, offset)
            results.append(entities)
        return results

    def create_entity_node(
        self,
        extracted: ExtractedEntity,
        dataroom_id: str,
    ) -> Entity:
        """Create an Entity model from an extracted entity.

        Args:
            extracted: Extracted entity data.
            dataroom_id: Data room ID.

        Returns:
            Entity model instance.
        """
        # Generate type-independent ID based on canonical name
        # This prevents duplicate entities when type classification changes
        entity_id = generate_entity_id(dataroom_id, extracted.name)

        base_kwargs = {
            "id": entity_id,
            "dataroom_id": dataroom_id,
            "name": extracted.name,
            "canonical_name": extracted.name,  # Will be updated by canonicalizer
            "confidence": extracted.confidence,
        }

        # Create specific entity type
        if extracted.entity_type == EntityType.FUND:
            return Fund(
                **base_kwargs,
                vintage_year=extracted.attributes.get("vintage_year"),
                strategy=extracted.attributes.get("strategy"),
            )
        elif extracted.entity_type == EntityType.MANAGER:
            return Manager(
                **base_kwargs,
                aum=extracted.attributes.get("aum"),
                headquarters=extracted.attributes.get("headquarters"),
            )
        elif extracted.entity_type == EntityType.PERSON:
            return Person(
                **base_kwargs,
                title=extracted.attributes.get("title"),
                role=extracted.attributes.get("role"),
                organization=extracted.attributes.get("organization"),
            )
        elif extracted.entity_type == EntityType.VEHICLE:
            return Vehicle(
                **base_kwargs,
                vehicle_type=extracted.attributes.get("vehicle_type"),
                jurisdiction=extracted.attributes.get("jurisdiction"),
            )
        elif extracted.entity_type == EntityType.SERVICE_PROVIDER:
            # Ensure services is a list (LLM sometimes returns a string)
            services = extracted.attributes.get("services", [])
            if isinstance(services, str):
                services = [services] if services else []
            return ServiceProvider(
                **base_kwargs,
                provider_type=extracted.attributes.get("provider_type"),
                services=services,
            )
        elif extracted.entity_type == EntityType.INVESTOR:
            return Investor(
                **base_kwargs,
                investor_type=extracted.attributes.get("investor_type"),
                commitment_amount=extracted.attributes.get("commitment_amount"),
            )
        elif extracted.entity_type == EntityType.PORTFOLIO_COMPANY:
            return PortfolioCompany(
                **base_kwargs,
                industry=extracted.attributes.get("industry"),
                ownership_pct=extracted.attributes.get("ownership_pct"),
                investment_date=extracted.attributes.get("investment_date"),
                exit_date=extracted.attributes.get("exit_date"),
            )
        elif extracted.entity_type == EntityType.LOCATION:
            return Location(
                **base_kwargs,
                location_type=extracted.attributes.get("location_type"),
                parent_location=extracted.attributes.get("parent_location"),
            )
        elif extracted.entity_type == EntityType.ASSET:
            return Asset(
                **base_kwargs,
                asset_type=extracted.attributes.get("asset_type"),
                location_id=extracted.attributes.get("location_id"),
            )
        else:
            return Entity.create(
                dataroom_id=dataroom_id,
                entity_type=extracted.entity_type,
                canonical_name=extracted.name,
                confidence=extracted.confidence,
            )
