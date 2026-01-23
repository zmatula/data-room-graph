"""VLM-based entity extraction from PDF images.

This module provides a supplementary extraction pass that uses a Vision Language Model
(GPT-4o) to directly analyze PDF page images and extract entity names that may be
missed by standard OCR, such as text in stylized logos or graphics.

This is fully LLM-driven with no deterministic rules.
"""

import base64
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

from openai import OpenAI

from ..config import get_settings

logger = logging.getLogger(__name__)


VLM_EXTRACTION_PROMPT = """Analyze this image from a private equity fund document to extract company and fund names.

FOCUS: Look for company LOGOS - these appear as graphical/stylized text, often with images or icons.

CRITICAL EXTRACTION RULES:
1. Extract ONLY the text that appears DIRECTLY IN the logo graphic itself
2. Do NOT add words from surrounding text to the logo name
3. Portfolio company logos typically have format: "[Company Name] [Roman Numeral]"
   - Examples: "Double Eagle IV", "Ridge Runner II", "BlackSwan II", "Pegasus III", "Paloma Resources VII"
   - The roman numeral IS part of the name - include it
4. If a logo shows "RIDGE RUNNER" and separately "II", combine them as "Ridge Runner II"
5. Do NOT include generic words like "Resources", "Energy", "Partners" unless they are clearly part of the logo

WHAT TO LOOK FOR:
- Company logos (graphical/stylized text, often with icons)
- Fund names containing "Fund" (e.g., "EnCap Fund XII")
- Portfolio company names with roman numerals

WHAT TO IGNORE:
- Generic descriptive text near logos (like "Portfolio Company", "Investment")
- Section headers and titles
- Footer/header text

Return a JSON array:
[
  {"name": "Exact Logo Text", "type": "company|fund", "source": "logo in image"}
]

If no logos/entities found, return: []"""


@dataclass
class VLMExtractedEntity:
    """An entity extracted from an image via VLM."""
    name: str
    entity_type: str  # company, fund, person
    source: str  # description of where it was found
    page_number: int


@dataclass
class VLMExtractionResult:
    """Result of VLM extraction pass."""
    pages_analyzed: int = 0
    entities_found: int = 0
    entities: list[VLMExtractedEntity] = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = []


class VLMEntityExtractor:
    """Extract entities from PDF images using Vision Language Model."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_pages: int = 20,
    ):
        """Initialize the VLM extractor.

        Args:
            model: OpenAI vision model to use.
            max_pages: Maximum number of pages to analyze.
        """
        settings = get_settings()
        self.model = model
        self.max_pages = max_pages
        self.client = OpenAI(api_key=settings.openai_api_key)

    def _pdf_to_images(self, pdf_path: Path) -> list[tuple[int, bytes]]:
        """Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            List of (page_number, image_bytes) tuples.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF not installed. Run: pip install pymupdf")
            return []

        images = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(min(len(doc), self.max_pages)):
                page = doc.load_page(page_num)
                # Render at 2x resolution for better OCR
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                images.append((page_num + 1, img_bytes))
            doc.close()
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")

        return images

    def _extract_from_image(
        self,
        image_bytes: bytes,
        page_number: int,
    ) -> list[VLMExtractedEntity]:
        """Extract entities from a single image using VLM.

        Args:
            image_bytes: PNG image bytes.
            page_number: Page number for tracking.

        Returns:
            List of extracted entities.
        """
        import json
        import re

        # Encode image to base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VLM_EXTRACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()

            # Handle empty responses
            if not content:
                return []

            # Handle markdown code blocks
            if "```" in content:
                code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                if code_match:
                    content = code_match.group(1).strip()

            # Handle prose responses
            if content.lower().startswith("no ") or content.lower() == "none":
                return []

            # Try to find JSON array
            array_match = re.search(r'\[[\s\S]*\]', content)
            if array_match:
                content = array_match.group(0)

            if content.strip() == "[]":
                return []

            entities_data = json.loads(content)

            if not isinstance(entities_data, list):
                return []

            entities = []
            for ent in entities_data:
                if isinstance(ent, dict) and "name" in ent:
                    entities.append(VLMExtractedEntity(
                        name=ent["name"],
                        entity_type=ent.get("type", "company"),
                        source=ent.get("source", "VLM extraction"),
                        page_number=page_number,
                    ))

            return entities

        except json.JSONDecodeError as e:
            logger.debug(f"No valid JSON in VLM response: {e}")
            return []
        except Exception as e:
            logger.error(f"VLM extraction failed for page {page_number}: {e}")
            return []

    def extract_from_pdf(self, pdf_path: Path) -> VLMExtractionResult:
        """Extract entities from a PDF using VLM on each page.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            VLMExtractionResult with extracted entities.
        """
        logger.info(f"Starting VLM extraction for {pdf_path}")

        result = VLMExtractionResult()

        # Convert PDF to images
        images = self._pdf_to_images(pdf_path)
        if not images:
            logger.warning(f"No images extracted from {pdf_path}")
            return result

        # Extract from each page
        seen_names = set()
        for page_num, img_bytes in images:
            result.pages_analyzed += 1
            entities = self._extract_from_image(img_bytes, page_num)

            for ent in entities:
                # Deduplicate by name (case-insensitive)
                name_lower = ent.name.lower()
                if name_lower not in seen_names:
                    seen_names.add(name_lower)
                    result.entities.append(ent)
                    result.entities_found += 1
                    logger.info(f"VLM found entity: {ent.name} ({ent.entity_type}) on page {page_num}")

        logger.info(
            f"VLM extraction complete: {result.pages_analyzed} pages, "
            f"{result.entities_found} entities found"
        )

        return result
