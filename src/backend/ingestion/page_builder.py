"""Page node builder from Unstructured.io elements."""

import logging
from typing import Optional

from .unstructured import UnstructuredElement
from ..database.models import Page

logger = logging.getLogger(__name__)


class PageBuilder:
    """Builds Page nodes from Unstructured.io elements."""

    def extract_pages(
        self,
        elements: list[UnstructuredElement],
        dataroom_id: str,
        document_id: str,
    ) -> tuple[list[Page], dict[int, str]]:
        """Extract Page nodes from elements.

        Creates one Page node for each unique page number found in the elements.

        Args:
            elements: List of elements from Unstructured.io.
            dataroom_id: ID of the containing data room.
            document_id: ID of the source document.

        Returns:
            Tuple of (list of Page objects, dict mapping page_number to page_id).
        """
        # Collect unique page numbers
        page_numbers: set[int] = set()
        for element in elements:
            if element.page_number is not None:
                page_numbers.add(element.page_number)

        # Create Page objects
        pages: list[Page] = []
        page_id_map: dict[int, str] = {}

        for page_num in sorted(page_numbers):
            page = Page.create(
                dataroom_id=dataroom_id,
                document_id=document_id,
                page_number=page_num,
            )
            pages.append(page)
            page_id_map[page_num] = page.id

        logger.info(f"Extracted {len(pages)} pages from {len(elements)} elements")
        return pages, page_id_map

    def get_page_id_for_element(
        self,
        element: UnstructuredElement,
        page_id_map: dict[int, str],
    ) -> Optional[str]:
        """Get the Page ID for an element.

        Args:
            element: The Unstructured element.
            page_id_map: Mapping from page number to page ID.

        Returns:
            Page ID if element has a page number, None otherwise.
        """
        if element.page_number is not None:
            return page_id_map.get(element.page_number)
        return None
