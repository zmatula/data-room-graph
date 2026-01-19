"""Document type classification for PE data rooms."""

import logging
import re
from typing import Optional
from pathlib import Path

from ..database.models import DocumentType

logger = logging.getLogger(__name__)


# Classification patterns
CLASSIFICATION_PATTERNS: dict[DocumentType, list[tuple[str, float]]] = {
    DocumentType.LPA: [
        (r"limited\s+partnership\s+agreement", 0.95),
        (r"\bLPA\b", 0.7),
        (r"partnership\s+agreement", 0.6),
        (r"general\s+partner.*limited\s+partner", 0.8),
        (r"capital\s+commitment.*partnership", 0.7),
    ],
    DocumentType.SIDE_LETTER: [
        (r"side\s+letter", 0.95),
        (r"letter\s+agreement", 0.6),
        (r"most\s+favored\s+nation", 0.8),
        (r"MFN\s+provision", 0.8),
        (r"special\s+terms.*investor", 0.7),
    ],
    DocumentType.PPM: [
        (r"private\s+placement\s+memorandum", 0.95),
        (r"\bPPM\b", 0.7),
        (r"confidential\s+offering", 0.8),
        (r"offering\s+memorandum", 0.85),
        (r"investment\s+memorandum", 0.6),
    ],
    DocumentType.SUBSCRIPTION_AGREEMENT: [
        (r"subscription\s+agreement", 0.95),
        (r"subscription\s+booklet", 0.9),
        (r"investor\s+questionnaire", 0.7),
        (r"accredited\s+investor\s+certification", 0.8),
        (r"kyc.*aml", 0.5),
    ],
    DocumentType.CAPITAL_CALL_NOTICE: [
        (r"capital\s+call\s+notice", 0.95),
        (r"drawdown\s+notice", 0.9),
        (r"capital\s+contribution.*due", 0.8),
        (r"funding\s+notice", 0.7),
        (r"call\s+percentage.*commitment", 0.8),
    ],
    DocumentType.QUARTERLY_REPORT: [
        (r"quarterly\s+report", 0.9),
        (r"Q[1-4]\s+\d{4}\s+report", 0.85),
        (r"quarterly\s+update", 0.7),
        (r"quarter\s+ended", 0.6),
    ],
    DocumentType.ANNUAL_REPORT: [
        (r"annual\s+report", 0.9),
        (r"year\s+ended.*report", 0.8),
        (r"fiscal\s+year\s+\d{4}", 0.6),
        (r"annual\s+review", 0.7),
    ],
    DocumentType.FINANCIAL_STATEMENTS: [
        (r"financial\s+statements", 0.9),
        (r"audited\s+financials", 0.9),
        (r"balance\s+sheet", 0.6),
        (r"statement\s+of\s+operations", 0.7),
        (r"auditor.*report", 0.6),
        (r"notes\s+to.*financial\s+statements", 0.8),
    ],
    DocumentType.FEE_SCHEDULE: [
        (r"fee\s+schedule", 0.95),
        (r"management\s+fee.*schedule", 0.9),
        (r"carried\s+interest.*schedule", 0.8),
        (r"fee\s+letter", 0.7),
        (r"fee\s+arrangement", 0.7),
    ],
    DocumentType.TRACK_RECORD: [
        (r"track\s+record", 0.9),
        (r"performance\s+track\s+record", 0.95),
        (r"marketing\s+deck", 0.7),
        (r"investor\s+presentation", 0.7),
        (r"fund\s+performance", 0.6),
        (r"IRR.*performance", 0.7),
        (r"historical\s+returns", 0.8),
    ],
}

# Filename patterns
FILENAME_PATTERNS: dict[DocumentType, list[tuple[str, float]]] = {
    DocumentType.LPA: [
        (r"lpa", 0.8),
        (r"limited.*partnership.*agreement", 0.9),
    ],
    DocumentType.SIDE_LETTER: [
        (r"side.*letter", 0.9),
    ],
    DocumentType.PPM: [
        (r"ppm", 0.8),
        (r"offering.*memo", 0.7),
    ],
    DocumentType.SUBSCRIPTION_AGREEMENT: [
        (r"subscription", 0.7),
        (r"sub.*agreement", 0.7),
    ],
    DocumentType.CAPITAL_CALL_NOTICE: [
        (r"capital.*call", 0.9),
        (r"drawdown", 0.8),
    ],
    DocumentType.QUARTERLY_REPORT: [
        (r"q[1-4].*\d{4}", 0.8),
        (r"quarterly", 0.7),
    ],
    DocumentType.ANNUAL_REPORT: [
        (r"annual.*report", 0.8),
    ],
    DocumentType.FINANCIAL_STATEMENTS: [
        (r"financials?", 0.6),
        (r"audited", 0.7),
    ],
    DocumentType.FEE_SCHEDULE: [
        (r"fee", 0.5),
    ],
    DocumentType.TRACK_RECORD: [
        (r"track.*record", 0.9),
        (r"deck", 0.5),
        (r"presentation", 0.5),
    ],
}


class DocumentClassifier:
    """Classifies documents based on content and filename patterns."""

    def __init__(self):
        """Initialize the classifier."""
        # Compile content patterns
        self.content_patterns: dict[DocumentType, list[tuple[re.Pattern, float]]] = {}
        for doc_type, patterns in CLASSIFICATION_PATTERNS.items():
            self.content_patterns[doc_type] = [
                (re.compile(pattern, re.IGNORECASE), weight)
                for pattern, weight in patterns
            ]

        # Compile filename patterns
        self.filename_patterns: dict[DocumentType, list[tuple[re.Pattern, float]]] = {}
        for doc_type, patterns in FILENAME_PATTERNS.items():
            self.filename_patterns[doc_type] = [
                (re.compile(pattern, re.IGNORECASE), weight)
                for pattern, weight in patterns
            ]

    def classify(
        self,
        text: str,
        filename: Optional[str] = None,
        sample_size: int = 5000,
    ) -> tuple[DocumentType, float]:
        """Classify a document based on content and filename.

        Args:
            text: Document text content.
            filename: Optional filename for additional hints.
            sample_size: Number of characters to sample from text.

        Returns:
            Tuple of (DocumentType, confidence score).
        """
        scores: dict[DocumentType, float] = {dt: 0.0 for dt in DocumentType}

        # Sample text from beginning
        sample = text[:sample_size].lower()

        # Score based on content patterns
        for doc_type, patterns in self.content_patterns.items():
            for pattern, weight in patterns:
                matches = pattern.findall(sample)
                if matches:
                    # More matches = higher score, but diminishing returns
                    match_score = weight * min(len(matches), 3) / 3
                    scores[doc_type] = max(scores[doc_type], match_score)

        # Score based on filename
        if filename:
            filename_lower = filename.lower()
            for doc_type, patterns in self.filename_patterns.items():
                for pattern, weight in patterns:
                    if pattern.search(filename_lower):
                        # Filename is a strong hint, boost the score
                        scores[doc_type] += weight * 0.3

        # Find best match
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type], 1.0)

        # If confidence is too low, mark as unknown
        if confidence < 0.3:
            return DocumentType.UNKNOWN, confidence

        # If multiple types have similar scores, reduce confidence
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] < 0.2:
            confidence *= 0.8

        logger.debug(
            f"Classified '{filename or 'document'}' as {best_type.value} "
            f"with confidence {confidence:.2f}"
        )
        return best_type, confidence

    def classify_from_file(self, file_path: Path) -> tuple[DocumentType, float]:
        """Classify a document from a file path.

        Args:
            file_path: Path to the document file.

        Returns:
            Tuple of (DocumentType, confidence score).
        """
        # For now, just use filename-based classification
        # Full classification requires reading and parsing the document
        filename = file_path.name

        scores: dict[DocumentType, float] = {dt: 0.0 for dt in DocumentType}

        filename_lower = filename.lower()
        for doc_type, patterns in self.filename_patterns.items():
            for pattern, weight in patterns:
                if pattern.search(filename_lower):
                    scores[doc_type] = max(scores[doc_type], weight)

        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type], 1.0)

        if confidence < 0.3:
            return DocumentType.UNKNOWN, 0.0

        return best_type, confidence

    def get_all_scores(
        self, text: str, filename: Optional[str] = None
    ) -> dict[DocumentType, float]:
        """Get classification scores for all document types.

        Args:
            text: Document text content.
            filename: Optional filename.

        Returns:
            Dictionary mapping DocumentType to score.
        """
        scores: dict[DocumentType, float] = {dt: 0.0 for dt in DocumentType}
        sample = text[:5000].lower()

        for doc_type, patterns in self.content_patterns.items():
            for pattern, weight in patterns:
                if pattern.search(sample):
                    scores[doc_type] = max(scores[doc_type], weight)

        if filename:
            filename_lower = filename.lower()
            for doc_type, patterns in self.filename_patterns.items():
                for pattern, weight in patterns:
                    if pattern.search(filename_lower):
                        scores[doc_type] += weight * 0.3

        return {k: min(v, 1.0) for k, v in scores.items()}
