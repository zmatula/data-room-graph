"""Claude-powered narrative and description generator."""

import logging
from typing import Optional
from pathlib import Path

import anthropic

from ..config import get_settings

logger = logging.getLogger(__name__)


class NarrativeGenerator:
    """Generates folder narratives and section descriptions using Claude."""

    def __init__(self):
        """Initialize the generator with Anthropic client."""
        settings = get_settings()
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model

    def generate_folder_narrative(
        self,
        folder_path: str,
        file_tree: str,
        dataroom_name: str,
    ) -> str:
        """Generate a narrative description for a folder.

        Args:
            folder_path: Relative path of the folder within the data room.
            file_tree: String representation of the full data room file tree.
            dataroom_name: Name of the data room.

        Returns:
            A 2-3 sentence narrative describing the folder's contents and purpose.
        """
        prompt = f"""You are analyzing a Private Equity fund data room. Generate a concise 2-3 sentence narrative describing this folder's contents and purpose within the data room context.

Data Room: {dataroom_name}

Full File Tree:
{file_tree}

Target Folder: {folder_path}

Generate a professional, informative narrative that:
1. Describes what types of documents are in this folder
2. Explains the folder's role in the data room structure
3. Notes any important subcategories if present

Respond with ONLY the narrative, no preamble or explanation."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )

            narrative = response.content[0].text.strip()
            logger.debug(f"Generated narrative for folder {folder_path}: {narrative[:50]}...")
            return narrative

        except Exception as e:
            logger.error(f"Failed to generate folder narrative: {e}")
            return ""

    def generate_section_description(
        self,
        title: str,
        content_preview: str,
        doc_type: str,
        doc_filename: str,
    ) -> str:
        """Generate a description for a top-level section.

        Args:
            title: The section title (from Title element).
            content_preview: First ~500 chars of the section content.
            doc_type: Document type classification.
            doc_filename: Name of the source document.

        Returns:
            A 1-2 sentence description of the section's content.
        """
        prompt = f"""You are analyzing a section from a Private Equity fund document. Generate a concise 1-2 sentence description of what this section contains.

Document: {doc_filename}
Document Type: {doc_type}
Section Title: {title}

Content Preview:
{content_preview[:500]}

Generate a professional description that:
1. Summarizes the key content or purpose of this section
2. Uses domain-appropriate terminology

Respond with ONLY the description, no preamble or explanation."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )

            description = response.content[0].text.strip()
            logger.debug(f"Generated description for section '{title}': {description[:50]}...")
            return description

        except Exception as e:
            logger.error(f"Failed to generate section description: {e}")
            return ""

    async def generate_folder_narratives_batch(
        self,
        folders: list[dict],
        file_tree: str,
        dataroom_name: str,
    ) -> dict[str, str]:
        """Generate narratives for multiple folders.

        Args:
            folders: List of folder dicts with 'id' and 'path' keys.
            file_tree: String representation of the full data room file tree.
            dataroom_name: Name of the data room.

        Returns:
            Dict mapping folder_id to narrative.
        """
        narratives = {}
        for folder in folders:
            narrative = self.generate_folder_narrative(
                folder["path"],
                file_tree,
                dataroom_name,
            )
            narratives[folder["id"]] = narrative
        return narratives

    def build_file_tree(self, root_path: Path, files: list[Path]) -> str:
        """Build a string representation of the file tree.

        Args:
            root_path: Root folder path.
            files: List of file paths.

        Returns:
            String representation of the folder structure.
        """
        # Get unique folder paths and organize into a tree structure
        tree_lines = []
        folders = set()

        for file_path in sorted(files):
            rel_path = file_path.relative_to(root_path)
            folders.add(str(rel_path.parent))

        # Add root
        tree_lines.append(f"{root_path.name}/")

        # Build folder structure
        seen_folders = set()
        for file_path in sorted(files):
            rel_path = file_path.relative_to(root_path)

            # Add any new parent folders
            parts = rel_path.parts[:-1]  # Exclude filename
            for i in range(len(parts)):
                folder = "/".join(parts[: i + 1])
                if folder not in seen_folders:
                    indent = "  " * (i + 1)
                    tree_lines.append(f"{indent}{parts[i]}/")
                    seen_folders.add(folder)

            # Add file
            indent = "  " * len(parts) + "  "
            tree_lines.append(f"{indent}{rel_path.name}")

        return "\n".join(tree_lines[:100])  # Limit for API context
