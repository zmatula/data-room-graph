"""Extraction validation component for Neo4j GraphRAG.

This module provides post-extraction validation that applies PE domain heuristics
to verify entity types and suggest corrections for likely misclassifications.

The validator runs AFTER entity extraction to:
1. Verify entity types match domain patterns
2. Flag potential misclassifications
3. Suggest corrections with confidence scores
4. Apply automatic fixes for high-confidence issues
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from ..database.neo4j_client import Neo4jClient, get_neo4j_client
from .pe_schema import get_entity_labels

logger = logging.getLogger(__name__)


class IssueType(str, Enum):
    """Types of validation issues."""

    MISCLASSIFIED_TYPE = "misclassified_type"
    INVALID_NAME_PATTERN = "invalid_name_pattern"
    MISSING_REQUIRED_PROPERTY = "missing_required_property"
    SUSPICIOUS_CONFIDENCE = "suspicious_confidence"
    POTENTIAL_DUPLICATE = "potential_duplicate"


class IssueSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"      # High confidence this is wrong
    WARNING = "warning"  # Likely wrong, needs review
    INFO = "info"        # Possible issue, low confidence


@dataclass
class ValidationIssue:
    """A validation issue found during extraction validation."""

    entity_id: str
    entity_name: str
    entity_label: str
    issue_type: IssueType
    issue: str
    confidence: float  # Confidence that this is actually an issue (0-1)
    severity: IssueSeverity = IssueSeverity.WARNING
    suggested_label: Optional[str] = None
    suggested_fix: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "entity_label": self.entity_label,
            "issue_type": self.issue_type.value,
            "issue": self.issue,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "suggested_label": self.suggested_label,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class ValidationResult:
    """Result of validation run."""

    total_entities: int = 0
    issues_found: int = 0
    issues_fixed: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)


class ExtractionValidator:
    """Validates extracted entities against PE domain heuristics.

    This validator applies domain-specific rules to detect misclassified
    entities and suggest corrections.
    """

    # PE Domain validation patterns
    FUND_INDICATORS = [
        r"\bFund\b",           # "Fund" in name
        r"\b[IVXLCDM]+\b",     # Roman numerals
        r"\b(19|20)\d{2}\b",   # Year numbers (1900s-2000s)
        r"\bLP\b",             # LP suffix
        r"\bL\.P\.\b",
    ]

    MANAGER_INDICATORS = [
        r"\bManagement\b",
        r"\bAdvisors?\b",
        r"\bCapital\b",
        r"\bPartners\b",
        r"\bGP\b",
        r"\bG\.P\.\b",
        r"\bGroup\b",
        r"\bHoldings\b",
    ]

    LOCATION_INDICATORS = [
        r"\bBasin\b",
        r"\bFormation\b",
        r"\bShale\b",
        r"\bPlay\b",
        r"\bField\b",
        # Known basins
        r"\bPermian\b",
        r"\bBakken\b",
        r"\bEagle Ford\b",
        r"\bMarcellus\b",
        r"\bUtica\b",
        r"\bHaynesville\b",
        r"\bDJ Basin\b",
        r"\bDelaware Basin\b",
        r"\bMidland Basin\b",
        r"\bAnadarko\b",
    ]

    SERVICE_PROVIDER_PATTERNS = [
        r"\bLLP\b",
        r"\bL\.L\.P\.\b",
        r"\b& Co\.?\b",
        # Known providers
        r"Deloitte",
        r"Ernst & Young",
        r"EY",
        r"KPMG",
        r"PwC",
        r"PricewaterhouseCoopers",
        r"Morgan Lewis",
        r"Kirkland",
        r"Simpson Thacher",
        r"Latham",
        r"Citco",
        r"State Street",
    ]

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        auto_fix_threshold: float = 0.90,
    ):
        """Initialize the validator.

        Args:
            neo4j_client: Neo4j client for database operations.
            auto_fix_threshold: Confidence threshold for automatic fixes.
        """
        self.neo4j = neo4j_client or get_neo4j_client()
        self.auto_fix_threshold = auto_fix_threshold

        # Compile patterns
        self.fund_pattern = re.compile(
            "|".join(self.FUND_INDICATORS), re.IGNORECASE
        )
        self.manager_pattern = re.compile(
            "|".join(self.MANAGER_INDICATORS), re.IGNORECASE
        )
        self.location_pattern = re.compile(
            "|".join(self.LOCATION_INDICATORS), re.IGNORECASE
        )
        self.service_provider_pattern = re.compile(
            "|".join(self.SERVICE_PROVIDER_PATTERNS), re.IGNORECASE
        )

    def _looks_like_fund(self, name: str) -> tuple[bool, float]:
        """Check if name looks like a Fund.

        Args:
            name: Entity name to check.

        Returns:
            Tuple of (is_fund_like, confidence).
        """
        matches = self.fund_pattern.findall(name)
        has_fund = bool(re.search(r"\bFund\b", name, re.IGNORECASE))
        has_numeral = bool(re.search(r"\b[IVXLCDM]+\b|\b\d+\b", name))

        if has_fund and has_numeral:
            return True, 0.95
        elif has_fund:
            return True, 0.75
        elif len(matches) >= 2:
            return True, 0.70

        return False, 0.0

    def _looks_like_manager(self, name: str) -> tuple[bool, float]:
        """Check if name looks like a Manager.

        Args:
            name: Entity name to check.

        Returns:
            Tuple of (is_manager_like, confidence).
        """
        matches = self.manager_pattern.findall(name)

        # Strong indicators
        if re.search(r"\bManagement\b", name, re.IGNORECASE):
            return True, 0.90
        if re.search(r"\bAdvisors?\b", name, re.IGNORECASE):
            return True, 0.85
        if re.search(r"\bGP\b|\bG\.P\.\b", name, re.IGNORECASE):
            return True, 0.85

        # Weak indicators (need multiple)
        if len(matches) >= 2:
            return True, 0.70
        elif len(matches) == 1:
            return True, 0.50

        return False, 0.0

    def _looks_like_location(self, name: str) -> tuple[bool, float]:
        """Check if name looks like a Location.

        Args:
            name: Entity name to check.

        Returns:
            Tuple of (is_location_like, confidence).
        """
        matches = self.location_pattern.findall(name)

        # Strong indicators
        if re.search(r"\bBasin\b|\bFormation\b|\bShale\b", name, re.IGNORECASE):
            return True, 0.95

        # Known basins
        known_basins = [
            "Permian", "Bakken", "Eagle Ford", "Marcellus",
            "Utica", "Haynesville", "DJ Basin", "Delaware Basin",
            "Midland Basin", "Anadarko"
        ]
        for basin in known_basins:
            if basin.lower() in name.lower():
                return True, 0.90

        return False, 0.0

    def _looks_like_service_provider(self, name: str) -> tuple[bool, float]:
        """Check if name looks like a ServiceProvider.

        Args:
            name: Entity name to check.

        Returns:
            Tuple of (is_service_provider_like, confidence).
        """
        matches = self.service_provider_pattern.findall(name)

        # Known providers
        known_providers = [
            "Deloitte", "Ernst", "KPMG", "PwC", "PricewaterhouseCoopers",
            "Morgan Lewis", "Kirkland", "Simpson Thacher", "Latham",
            "Citco", "State Street"
        ]
        for provider in known_providers:
            if provider.lower() in name.lower():
                return True, 0.95

        # LLP is strong indicator of law firm
        if re.search(r"\bLLP\b|\bL\.L\.P\.\b", name):
            return True, 0.85

        return False, 0.0

    async def validate_entity(
        self,
        entity_id: str,
        entity_name: str,
        entity_label: str,
    ) -> list[ValidationIssue]:
        """Validate a single entity.

        Args:
            entity_id: Entity ID.
            entity_name: Entity name.
            entity_label: Current entity label.

        Returns:
            List of validation issues found.
        """
        issues = []

        # Check Fund classification
        if entity_label == "Fund":
            looks_like_fund, fund_conf = self._looks_like_fund(entity_name)
            looks_like_manager, manager_conf = self._looks_like_manager(entity_name)

            if not looks_like_fund and looks_like_manager and manager_conf > 0.70:
                issues.append(ValidationIssue(
                    entity_id=entity_id,
                    entity_name=entity_name,
                    entity_label=entity_label,
                    issue_type=IssueType.MISCLASSIFIED_TYPE,
                    issue=f"Fund '{entity_name}' has Manager indicators but no Fund indicators",
                    confidence=manager_conf,
                    severity=IssueSeverity.WARNING if manager_conf < 0.85 else IssueSeverity.ERROR,
                    suggested_label="Manager",
                    suggested_fix=f"Change label from Fund to Manager",
                ))

        # Check Manager classification
        elif entity_label == "Manager":
            looks_like_fund, fund_conf = self._looks_like_fund(entity_name)
            looks_like_manager, manager_conf = self._looks_like_manager(entity_name)

            if looks_like_fund and fund_conf > 0.80 and not looks_like_manager:
                issues.append(ValidationIssue(
                    entity_id=entity_id,
                    entity_name=entity_name,
                    entity_label=entity_label,
                    issue_type=IssueType.MISCLASSIFIED_TYPE,
                    issue=f"Manager '{entity_name}' contains 'Fund' + numerals - likely a Fund",
                    confidence=fund_conf,
                    severity=IssueSeverity.ERROR,
                    suggested_label="Fund",
                    suggested_fix=f"Change label from Manager to Fund",
                ))

        # Check PortfolioCompany classification
        elif entity_label == "PortfolioCompany":
            looks_like_location, location_conf = self._looks_like_location(entity_name)

            if looks_like_location and location_conf > 0.80:
                issues.append(ValidationIssue(
                    entity_id=entity_id,
                    entity_name=entity_name,
                    entity_label=entity_label,
                    issue_type=IssueType.MISCLASSIFIED_TYPE,
                    issue=f"PortfolioCompany '{entity_name}' matches Location patterns",
                    confidence=location_conf,
                    severity=IssueSeverity.WARNING,
                    suggested_label="Location",
                    suggested_fix=f"Change label from PortfolioCompany to Location",
                ))

        # Check Location classification
        elif entity_label == "Location":
            looks_like_location, location_conf = self._looks_like_location(entity_name)

            if not looks_like_location:
                issues.append(ValidationIssue(
                    entity_id=entity_id,
                    entity_name=entity_name,
                    entity_label=entity_label,
                    issue_type=IssueType.INVALID_NAME_PATTERN,
                    issue=f"Location '{entity_name}' doesn't match typical location patterns",
                    confidence=0.60,
                    severity=IssueSeverity.INFO,
                ))

        # Check ServiceProvider classification
        elif entity_label == "ServiceProvider":
            looks_like_sp, sp_conf = self._looks_like_service_provider(entity_name)
            looks_like_manager, manager_conf = self._looks_like_manager(entity_name)

            if not looks_like_sp and looks_like_manager and manager_conf > 0.70:
                issues.append(ValidationIssue(
                    entity_id=entity_id,
                    entity_name=entity_name,
                    entity_label=entity_label,
                    issue_type=IssueType.MISCLASSIFIED_TYPE,
                    issue=f"ServiceProvider '{entity_name}' matches Manager patterns",
                    confidence=manager_conf,
                    severity=IssueSeverity.WARNING,
                    suggested_label="Manager",
                    suggested_fix=f"Change label from ServiceProvider to Manager",
                ))

        return issues

    async def validate_dataroom(
        self,
        dataroom_id: str,
        auto_fix: bool = False,
    ) -> ValidationResult:
        """Validate all entities in a data room.

        Args:
            dataroom_id: Data room ID.
            auto_fix: Whether to automatically fix high-confidence issues.

        Returns:
            ValidationResult with all issues found.
        """
        logger.info(f"Validating entities for data room {dataroom_id}")

        result = ValidationResult()
        entity_labels = get_entity_labels()

        # Query all entities
        all_entities = []
        for label in entity_labels:
            query = f"""
            MATCH (e:{label})
            WHERE e.dataroom_id = $dataroom_id
            RETURN e.id AS id, e.name AS name, '{label}' AS label
            """

            entities = self.neo4j.execute_read(
                query, {"dataroom_id": dataroom_id}
            )

            if entities:
                all_entities.extend(entities)

        result.total_entities = len(all_entities)
        logger.info(f"Found {result.total_entities} entities to validate")

        # Validate each entity
        for entity in all_entities:
            issues = await self.validate_entity(
                entity_id=entity["id"],
                entity_name=entity["name"] or "",
                entity_label=entity["label"],
            )

            for issue in issues:
                result.issues.append(issue)
                result.issues_found += 1

                # Auto-fix high-confidence issues
                if auto_fix and issue.confidence >= self.auto_fix_threshold:
                    if issue.issue_type == IssueType.MISCLASSIFIED_TYPE and issue.suggested_label:
                        success = await self._apply_label_fix(issue)
                        if success:
                            result.issues_fixed += 1

        logger.info(
            f"Validation complete: {result.issues_found} issues found, "
            f"{result.issues_fixed} fixed, {result.error_count} errors, "
            f"{result.warning_count} warnings"
        )

        return result

    async def _apply_label_fix(self, issue: ValidationIssue) -> bool:
        """Apply a label fix to an entity.

        Args:
            issue: Validation issue with suggested fix.

        Returns:
            True if fix was applied successfully.
        """
        if not issue.suggested_label:
            return False

        logger.info(
            f"Applying fix: changing {issue.entity_name} from "
            f"{issue.entity_label} to {issue.suggested_label}"
        )

        try:
            # Remove old label and add new one
            query = f"""
            MATCH (e:{issue.entity_label} {{id: $entity_id}})
            REMOVE e:{issue.entity_label}
            SET e:{issue.suggested_label}
            SET e.entity_type = $new_type
            SET e.validation_fixed = true
            SET e.original_type = $original_type
            """

            self.neo4j.execute_write(
                query,
                {
                    "entity_id": issue.entity_id,
                    "new_type": issue.suggested_label,
                    "original_type": issue.entity_label,
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to apply fix for {issue.entity_id}: {e}")
            return False

    async def get_validation_report(
        self,
        dataroom_id: str,
    ) -> dict:
        """Generate a validation report for a data room.

        Args:
            dataroom_id: Data room ID.

        Returns:
            Report dictionary with statistics and issues.
        """
        result = await self.validate_dataroom(dataroom_id, auto_fix=False)

        # Group issues by type
        issues_by_type = {}
        for issue in result.issues:
            issue_type = issue.issue_type.value
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue.to_dict())

        # Group issues by severity
        issues_by_severity = {
            "error": [i.to_dict() for i in result.issues if i.severity == IssueSeverity.ERROR],
            "warning": [i.to_dict() for i in result.issues if i.severity == IssueSeverity.WARNING],
            "info": [i.to_dict() for i in result.issues if i.severity == IssueSeverity.INFO],
        }

        return {
            "dataroom_id": dataroom_id,
            "total_entities": result.total_entities,
            "issues_found": result.issues_found,
            "error_count": result.error_count,
            "warning_count": result.warning_count,
            "issues_by_type": issues_by_type,
            "issues_by_severity": issues_by_severity,
        }


async def validate_extraction(
    dataroom_id: str,
    auto_fix: bool = False,
    fix_threshold: float = 0.90,
) -> ValidationResult:
    """Convenience function to run extraction validation.

    Args:
        dataroom_id: Data room ID.
        auto_fix: Whether to automatically fix high-confidence issues.
        fix_threshold: Confidence threshold for automatic fixes.

    Returns:
        ValidationResult with all issues found.
    """
    validator = ExtractionValidator(auto_fix_threshold=fix_threshold)
    return await validator.validate_dataroom(dataroom_id, auto_fix=auto_fix)
