"""
Structure-aware document chunker for the RAG pipeline.

Features:
- Preserves document structure and headings
- Tracks parent/child heading hierarchy
- Keeps fenced code blocks together
- Keeps lists together where possible
- Keeps paragraphs together
- Splits oversized content on sentence boundaries
- Adds heading context to chunks
- Avoids tiny meaningless chunks
- Adds controlled overlap between split chunks
- Generates useful metadata for semantic/hybrid retrieval
- Supports Markdown, RST, Python, JSON, YAML and plain text

Output chunk format:

{
    "content": "...",
    "path": "...",
    "category": "...",
    "section": "...",
    "parent_section": "...",
    "section_path": [...],
    "chunk_index": 0,
    "chunk_type": "text",
    "language": "python"
}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Configuration
# ============================================================

DEFAULT_MAX_CHARS = 1800
DEFAULT_MIN_CHARS = 250
DEFAULT_OVERLAP_CHARS = 250


# ============================================================
# Internal data structures
# ============================================================

@dataclass
class Block:
    """
    A logical document block.

    Examples:
        - paragraph
        - heading
        - code
        - list
        - directive
    """

    content: str
    block_type: str

    level: int = 0

    heading: str = ""
    parent_heading: str = ""

    section_path: list[str] = field(default_factory=list)


# ============================================================
# Main Chunker
# ============================================================

class DocumentChunker:
    """
    Structure-aware document chunker.

    The chunker first converts a document into logical blocks and
    then combines compatible blocks into semantically meaningful
    chunks.

    This is intentionally more structure-aware than a simple
    character/word based splitter.
    """

    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_CHARS,
        min_chars: int = DEFAULT_MIN_CHARS,
        overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    ):
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")

        if min_chars < 0:
            raise ValueError("min_chars cannot be negative")

        if overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative")

        if min_chars >= max_chars:
            raise ValueError(
                "min_chars must be smaller than max_chars"
            )

        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_chars = min(
            overlap_chars,
            max_chars // 3,
        )

    # ========================================================
    # Public API
    # ========================================================

    def chunk_document(
        self,
        content: str,
        path: str = "",
        category: str = "",
    ) -> list[dict]:
        """
        Chunk a single document.

        Parameters
        ----------
        content:
            Raw document content.

        path:
            Repository/file path.

        category:
            Existing document category, e.g.
            "documentation" or "source".

        Returns
        -------
        list[dict]
            Structured chunks suitable for embedding/retrieval.
        """

        if not content or not content.strip():
            return []

        language = self._detect_language(path)

        blocks = self._parse_blocks(
            content=content,
            language=language,
        )

        if not blocks:
            return []

        chunks = self._build_chunks(
            blocks=blocks,
            path=path,
            category=category,
            language=language,
        )

        return chunks

    def chunk_documents(
        self,
        documents: list[dict],
    ) -> list[dict]:
        """
        Chunk multiple documents.

        Expected document format:

        {
            "content": "...",
            "path": "...",
            "category": "documentation"
        }
        """

        all_chunks = []

        for document in documents:
            chunks = self.chunk_document(
                content=document.get("content", ""),
                path=document.get("path", ""),
                category=document.get("category", ""),
            )

            all_chunks.extend(chunks)

        return all_chunks

    # ========================================================
    # Block parsing
    # ========================================================

    def _parse_blocks(
        self,
        content: str,
        language: str,
    ) -> list[Block]:
        """
        Convert raw text into logical blocks.
        """

        lines = content.replace("\r\n", "\n").replace(
            "\r",
            "\n",
        ).split("\n")

        blocks: list[Block] = []

        current_lines: list[str] = []

        current_code: list[str] = []

        in_code = False
        code_fence = None

        heading_stack: list[tuple[int, str]] = []

        i = 0

        while i < len(lines):
            line = lines[i]

            # ------------------------------------------------
            # Fenced code block
            # ------------------------------------------------

            fence_match = re.match(
                r"^\s*(```+|~~~+)(.*)$",
                line,
            )

            if fence_match:
                fence = fence_match.group(1)

                if not in_code:
                    self._flush_paragraph(
                        blocks,
                        current_lines,
                        heading_stack,
                    )
                    current_lines = []

                    in_code = True
                    code_fence = fence
                    current_code = [line]

                else:
                    current_code.append(line)

                    if line.strip().startswith(code_fence):
                        blocks.append(
                            Block(
                                content="\n".join(
                                    current_code
                                ).strip(),
                                block_type="code",
                                section_path=[
                                    x[1]
                                    for x in heading_stack
                                ],
                            )
                        )

                        current_code = []
                        in_code = False
                        code_fence = None

                i += 1
                continue

            # ------------------------------------------------
            # Inside fenced code
            # ------------------------------------------------

            if in_code:
                current_code.append(line)
                i += 1
                continue

            # ------------------------------------------------
            # Markdown ATX heading
            # ------------------------------------------------

            markdown_heading = re.match(
                r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$",
                line,
            )

            if markdown_heading:
                self._flush_paragraph(
                    blocks,
                    current_lines,
                    heading_stack,
                )
                current_lines = []

                level = len(markdown_heading.group(1))
                title = markdown_heading.group(2).strip()

                heading_stack = self._update_heading_stack(
                    heading_stack,
                    level,
                    title,
                )

                blocks.append(
                    Block(
                        content=title,
                        block_type="heading",
                        level=level,
                        heading=title,
                        section_path=[
                            x[1]
                            for x in heading_stack
                        ],
                    )
                )

                i += 1
                continue

            # ------------------------------------------------
            # RST underline heading
            #
            # Example:
            #
            # Linear Models
            # =============
            # ------------------------------------------------

            if (
                line.strip()
                and i + 1 < len(lines)
                and self._is_rst_underline(lines[i + 1])
            ):
                self._flush_paragraph(
                    blocks,
                    current_lines,
                    heading_stack,
                )
                current_lines = []

                title = line.strip()

                level = self._rst_heading_level(
                    lines[i + 1]
                )

                heading_stack = self._update_heading_stack(
                    heading_stack,
                    level,
                    title,
                )

                blocks.append(
                    Block(
                        content=title,
                        block_type="heading",
                        level=level,
                        heading=title,
                        section_path=[
                            x[1]
                            for x in heading_stack
                        ],
                    )
                )

                i += 2
                continue

            # ------------------------------------------------
            # Blank line
            # ------------------------------------------------

            if not line.strip():
                self._flush_paragraph(
                    blocks,
                    current_lines,
                    heading_stack,
                )
                current_lines = []

                i += 1
                continue

            # ------------------------------------------------
            # Lists
            # ------------------------------------------------

            if self._is_list_item(line):
                self._flush_paragraph(
                    blocks,
                    current_lines,
                    heading_stack,
                )
                current_lines = []

                list_lines = [line]

                i += 1

                while i < len(lines):
                    next_line = lines[i]

                    if not next_line.strip():
                        break

                    if self._is_list_item(next_line):
                        list_lines.append(next_line)
                        i += 1
                        continue

                    # Continuation line of list item
                    if (
                        next_line.startswith(" ")
                        or next_line.startswith("\t")
                    ):
                        list_lines.append(next_line)
                        i += 1
                        continue

                    break

                blocks.append(
                    Block(
                        content="\n".join(
                            list_lines
                        ).strip(),
                        block_type="list",
                        section_path=[
                            x[1]
                            for x in heading_stack
                        ],
                    )
                )

                continue

            # ------------------------------------------------
            # RST directives
            #
            # Example:
            #
            # .. note::
            #    Some information
            # ------------------------------------------------

            if re.match(
                r"^\s*\.\.\s+\S+::",
                line,
            ):
                self._flush_paragraph(
                    blocks,
                    current_lines,
                    heading_stack,
                )
                current_lines = []

                directive_lines = [line]

                i += 1

                while i < len(lines):
                    next_line = lines[i]

                    if (
                        next_line.startswith(" ")
                        or next_line.startswith("\t")
                        or not next_line.strip()
                    ):
                        directive_lines.append(next_line)
                        i += 1
                    else:
                        break

                blocks.append(
                    Block(
                        content="\n".join(
                            directive_lines
                        ).strip(),
                        block_type="directive",
                        section_path=[
                            x[1]
                            for x in heading_stack
                        ],
                    )
                )

                continue

            # ------------------------------------------------
            # Normal text
            # ------------------------------------------------

            current_lines.append(line)

            i += 1

        # ----------------------------------------------------
        # Flush remaining code
        # ----------------------------------------------------

        if current_code:
            blocks.append(
                Block(
                    content="\n".join(
                        current_code
                    ).strip(),
                    block_type="code",
                    section_path=[
                        x[1]
                        for x in heading_stack
                    ],
                )
            )

        # ----------------------------------------------------
        # Flush remaining paragraph
        # ----------------------------------------------------

        self._flush_paragraph(
            blocks,
            current_lines,
            heading_stack,
        )

        return blocks

    # ========================================================
    # Paragraph handling
    # ========================================================

    def _flush_paragraph(
        self,
        blocks: list[Block],
        lines: list[str],
        heading_stack: list[tuple[int, str]],
    ) -> None:
        """
        Convert accumulated text lines into a paragraph block.
        """

        if not lines:
            return

        text = "\n".join(lines).strip()

        if not text:
            return

        blocks.append(
            Block(
                content=text,
                block_type="text",
                section_path=[
                    x[1]
                    for x in heading_stack
                ],
            )
        )

    # ========================================================
    # Chunk construction
    # ========================================================

    def _build_chunks(
        self,
        blocks: list[Block],
        path: str,
        category: str,
        language: str,
    ) -> list[dict]:
        """
        Combine logical blocks into final chunks.
        """

        chunks: list[dict] = []

        current_blocks: list[Block] = []
        current_length = 0

        current_section_path: list[str] = []

        def flush_current():
            nonlocal current_blocks
            nonlocal current_length

            if not current_blocks:
                return

            chunks.extend(
                self._materialize_blocks(
                    current_blocks,
                    path,
                    category,
                    language,
                )
            )

            current_blocks = []
            current_length = 0

        for block in blocks:

            # ------------------------------------------------
            # Heading
            # ------------------------------------------------

            if block.block_type == "heading":
                flush_current()

                current_section_path = (
                    block.section_path
                )

                continue

            # ------------------------------------------------
            # Determine whether block fits
            # ------------------------------------------------

            block_length = len(block.content)

            separator_length = 2 if current_blocks else 0

            would_exceed = (
                current_length
                + separator_length
                + block_length
                > self.max_chars
            )

            # ------------------------------------------------
            # Code blocks should remain intact whenever possible
            # ------------------------------------------------

            if block.block_type == "code":

                if current_blocks:
                    flush_current()

                if block_length <= self.max_chars:
                    current_blocks = [block]
                    current_length = block_length
                    flush_current()

                else:
                    split_blocks = self._split_large_code_block(
                        block
                    )

                    for split_block in split_blocks:
                        chunks.extend(
                            self._materialize_blocks(
                                [split_block],
                                path,
                                category,
                                language,
                            )
                        )

                continue

            # ------------------------------------------------
            # Large block
            # ------------------------------------------------

            if block_length > self.max_chars:

                flush_current()

                split_blocks = self._split_large_block(
                    block
                )

                chunks.extend(
                    self._materialize_blocks(
                        split_blocks,
                        path,
                        category,
                        language,
                    )
                )

                continue

            # ------------------------------------------------
            # Current chunk would become too large
            # ------------------------------------------------

            if would_exceed:

                # If current chunk is still too small,
                # attempt to keep the current block with it.
                if (
                    current_length < self.min_chars
                    and current_blocks
                ):
                    current_blocks.append(block)
                    current_length += (
                        separator_length
                        + block_length
                    )

                else:
                    previous_blocks = current_blocks.copy()

                    flush_current()

                    # Add controlled overlap from the previous
                    # chunk where useful.
                    overlap_blocks = (
                        self._select_overlap_blocks(
                            previous_blocks
                        )
                    )

                    if overlap_blocks:
                        overlap_length = sum(
                            len(x.content)
                            for x in overlap_blocks
                        )

                        if (
                            overlap_length
                            + block_length
                            <= self.max_chars
                        ):
                            current_blocks.extend(
                                overlap_blocks
                            )

                            current_length = (
                                overlap_length
                                + 2
                                * max(
                                    0,
                                    len(overlap_blocks) - 1,
                                )
                            )

                    current_blocks.append(block)

                    current_length += (
                        block_length
                        + (
                            2
                            if len(current_blocks) > 1
                            else 0
                        )
                    )

            else:
                current_blocks.append(block)

                current_length += (
                    block_length
                    + (
                        2
                        if len(current_blocks) > 1
                        else 0
                    )
                )

        flush_current()

        # ----------------------------------------------------
        # Clean tiny chunks by merging with neighbors
        # ----------------------------------------------------

        chunks = self._merge_tiny_chunks(
            chunks,
            path,
            category,
            language,
        )

        # ----------------------------------------------------
        # Add indices
        # ----------------------------------------------------

        for index, chunk in enumerate(chunks):
            chunk["chunk_index"] = index

        return chunks

    # ========================================================
    # Materialization
    # ========================================================

    def _materialize_blocks(
        self,
        blocks: list[Block],
        path: str,
        category: str,
        language: str,
    ) -> list[dict]:
        """
        Convert logical blocks into retriever-ready chunks.
        """

        if not blocks:
            return []

        results = []

        content_parts = []

        section_path = []

        block_types = []

        for block in blocks:

            if block.section_path:
                section_path = block.section_path

            content_parts.append(
                block.content.strip()
            )

            block_types.append(
                block.block_type
            )

        body = "\n\n".join(
            x for x in content_parts if x
        ).strip()

        if not body:
            return []

        # ----------------------------------------------------
        # Determine section metadata
        # ----------------------------------------------------

        section = (
            section_path[-1]
            if section_path
            else ""
        )

        parent_section = (
            section_path[-2]
            if len(section_path) >= 2
            else ""
        )

        # ----------------------------------------------------
        # Context enrichment
        #
        # This is important for embeddings.
        #
        # Instead of embedding only:
        #
        #     "It minimizes the residual sum..."
        #
        # we embed:
        #
        #     Section: Linear Models
        #     Subsection: Ordinary Least Squares
        #
        #     It minimizes...
        # ----------------------------------------------------

        context_lines = []

        if section_path:
            context_lines.append(
                "Section: "
                + " > ".join(section_path)
            )

        enriched_content = body

        if context_lines:
            enriched_content = (
                "\n".join(context_lines)
                + "\n\n"
                + body
            )

        # ----------------------------------------------------
        # Determine chunk type
        # ----------------------------------------------------

        unique_types = set(block_types)

        if unique_types == {"code"}:
            chunk_type = "code"

        elif "code" in unique_types:
            chunk_type = "mixed"

        elif "list" in unique_types:
            chunk_type = "list"

        elif "directive" in unique_types:
            chunk_type = "directive"

        else:
            chunk_type = "text"

        results.append(
            {
                "content": enriched_content,
                "raw_content": body,
                "path": path,
                "category": category,
                "section": section,
                "parent_section": parent_section,
                "section_path": section_path.copy(),
                "chunk_type": chunk_type,
                "language": language,
                "char_count": len(body),
            }
        )

        return results

    # ========================================================
    # Large block splitting
    # ========================================================

    def _split_large_block(
        self,
        block: Block,
    ) -> list[Block]:
        """
        Split an oversized text/list/directive block.

        Priority:

            paragraphs
                ↓
            sentences
                ↓
            words
        """

        text = block.content.strip()

        # First split paragraphs.
        paragraphs = [
            x.strip()
            for x in re.split(
                r"\n\s*\n",
                text,
            )
            if x.strip()
        ]

        if len(paragraphs) > 1:
            return self._pack_text_parts(
                paragraphs,
                block,
            )

        # Otherwise split into sentences.
        sentences = self._split_sentences(text)

        if len(sentences) > 1:
            return self._pack_text_parts(
                sentences,
                block,
            )

        # Last resort: word-based split.
        words = text.split()

        parts = []

        current = []

        current_length = 0

        for word in words:

            additional = (
                len(word)
                if not current
                else len(word) + 1
            )

            if (
                current
                and current_length + additional
                > self.max_chars
            ):
                parts.append(
                    " ".join(current)
                )

                current = [word]
                current_length = len(word)

            else:
                current.append(word)
                current_length += additional

        if current:
            parts.append(
                " ".join(current)
            )

        return [
            Block(
                content=part,
                block_type=block.block_type,
                level=block.level,
                heading=block.heading,
                parent_heading=block.parent_heading,
                section_path=block.section_path.copy(),
            )
            for part in parts
        ]

    def _split_large_code_block(
        self,
        block: Block,
    ) -> list[Block]:
        """
        Split very large code blocks.

        We prefer splitting by lines rather than characters so
        individual source lines remain intact.
        """

        lines = block.content.splitlines()

        parts = []

        current = []

        current_length = 0

        for line in lines:

            line_length = len(line)

            additional = (
                line_length
                if not current
                else line_length + 1
            )

            if (
                current
                and current_length + additional
                > self.max_chars
            ):
                parts.append(
                    "\n".join(current)
                )

                current = [line]
                current_length = line_length

            else:
                current.append(line)
                current_length += additional

        if current:
            parts.append(
                "\n".join(current)
            )

        return [
            Block(
                content=part,
                block_type="code",
                level=block.level,
                heading=block.heading,
                parent_heading=block.parent_heading,
                section_path=block.section_path.copy(),
            )
            for part in parts
        ]

    def _pack_text_parts(
        self,
        parts: list[str],
        source_block: Block,
    ) -> list[Block]:
        """
        Pack paragraphs/sentences into chunks without exceeding
        max_chars.
        """

        results = []

        current: list[str] = []
        current_length = 0

        for part in parts:

            part_length = len(part)

            additional = (
                part_length
                if not current
                else part_length + 2
            )

            if (
                current
                and current_length + additional
                > self.max_chars
            ):
                results.append(
                    Block(
                        content="\n\n".join(current),
                        block_type=source_block.block_type,
                        level=source_block.level,
                        heading=source_block.heading,
                        parent_heading=source_block.parent_heading,
                        section_path=source_block.section_path.copy(),
                    )
                )

                # Controlled sentence/paragraph overlap.
                overlap = self._get_text_overlap(
                    current
                )

                current = []

                if overlap:
                    current.append(overlap)
                    current_length = len(overlap)
                else:
                    current_length = 0

            current.append(part)

            current_length += (
                part_length
                if len(current) == 1
                else part_length + 2
            )

        if current:
            results.append(
                Block(
                    content="\n\n".join(current),
                    block_type=source_block.block_type,
                    level=source_block.level,
                    heading=source_block.heading,
                    parent_heading=source_block.parent_heading,
                    section_path=source_block.section_path.copy(),
                )
            )

        return results

    # ========================================================
    # Overlap
    # ========================================================

    def _get_text_overlap(
        self,
        parts: list[str],
    ) -> str:
        """
        Return a small amount of text from the end of the
        previous chunk.
        """

        if not parts:
            return ""

        overlap_parts = []

        length = 0

        for part in reversed(parts):

            if (
                length + len(part)
                > self.overlap_chars
            ):
                break

            overlap_parts.insert(
                0,
                part,
            )

            length += len(part) + 2

        return "\n\n".join(
            overlap_parts
        )

    def _select_overlap_blocks(
        self,
        blocks: list[Block],
    ) -> list[Block]:
        """
        Select whole logical blocks for overlap.
        """

        selected = []

        length = 0

        for block in reversed(blocks):

            block_length = len(
                block.content
            )

            if (
                selected
                and length + block_length
                > self.overlap_chars
            ):
                break

            if block_length > self.overlap_chars:
                break

            selected.insert(
                0,
                block,
            )

            length += block_length + 2

        return selected

    # ========================================================
    # Tiny chunk cleanup
    # ========================================================

    def _merge_tiny_chunks(
        self,
        chunks: list[dict],
        path: str,
        category: str,
        language: str,
    ) -> list[dict]:
        """
        Merge very small chunks with neighboring chunks when
        doing so does not exceed max_chars.
        """

        if len(chunks) <= 1:
            return chunks

        result = []

        i = 0

        while i < len(chunks):

            chunk = chunks[i]

            if (
                chunk["char_count"]
                >= self.min_chars
                or len(result) == 0
            ):
                result.append(chunk)
                i += 1
                continue

            previous = result[-1]

            combined_raw = (
                previous["raw_content"]
                + "\n\n"
                + chunk["raw_content"]
            )

            if len(combined_raw) <= self.max_chars:

                previous["raw_content"] = (
                    combined_raw
                )

                previous["content"] = (
                    previous["content"]
                    + "\n\n"
                    + chunk["raw_content"]
                )

                previous["char_count"] = (
                    len(combined_raw)
                )

                i += 1
                continue

            # Try merging into next chunk.
            if i + 1 < len(chunks):

                next_chunk = chunks[i + 1]

                combined_raw = (
                    chunk["raw_content"]
                    + "\n\n"
                    + next_chunk["raw_content"]
                )

                if len(combined_raw) <= self.max_chars:

                    next_chunk["raw_content"] = (
                        combined_raw
                    )

                    next_chunk["content"] = (
                        next_chunk["content"]
                        + "\n\n"
                        + chunk["raw_content"]
                    )

                    next_chunk["char_count"] = (
                        len(combined_raw)
                    )

                    i += 1
                    continue

            result.append(chunk)
            i += 1

        return result

    # ========================================================
    # Heading utilities
    # ========================================================

    @staticmethod
    def _update_heading_stack(
        stack: list[tuple[int, str]],
        level: int,
        title: str,
    ) -> list[tuple[int, str]]:
        """
        Update heading hierarchy.

        Example:

            # Machine Learning
            ## Supervised Learning
            ### Classification

        becomes:

            ["Machine Learning"]
            ["Machine Learning", "Supervised Learning"]
            [
                "Machine Learning",
                "Supervised Learning",
                "Classification"
            ]
        """

        new_stack = [
            item
            for item in stack
            if item[0] < level
        ]

        new_stack.append(
            (level, title)
        )

        return new_stack

    @staticmethod
    def _is_rst_underline(
        line: str,
    ) -> bool:
        """
        Detect RST heading underline.
        """

        stripped = line.strip()

        if not stripped:
            return False

        if len(stripped) < 3:
            return False

        return bool(
            re.fullmatch(
                r"[=\-~^\"'#*+`:.]+",
                stripped,
            )
        )

    @staticmethod
    def _rst_heading_level(
        underline: str,
    ) -> int:
        """
        Map RST underline styles to heading levels.
        """

        char = underline.strip()[0]

        mapping = {
            "=": 1,
            "-": 2,
            "~": 3,
            "^": 4,
            '"': 5,
            "'": 5,
            "#": 5,
            "*": 5,
            "+": 5,
            "`": 5,
            ":": 5,
            ".": 5,
        }

        return mapping.get(char, 5)

    # ========================================================
    # List detection
    # ========================================================

    @staticmethod
    def _is_list_item(
        line: str,
    ) -> bool:
        """
        Detect common Markdown/RST list syntax.
        """

        stripped = line.lstrip()

        patterns = [
            r"^[-*+]\s+",
            r"^\d+[.)]\s+",
            r"^[a-zA-Z][.)]\s+",
            r"^[-*]\s+\[[ xX]\]\s+",
        ]

        return any(
            re.match(pattern, stripped)
            for pattern in patterns
        )

    # ========================================================
    # Sentence splitting
    # ========================================================

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:
        """
        Lightweight sentence splitter.

        Avoids introducing an NLP dependency just for chunking.
        """

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if not text:
            return []

        # Split after sentence-ending punctuation.
        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9`])",
            text,
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    # ========================================================
    # Language detection
    # ========================================================

    @staticmethod
    def _detect_language(
        path: str,
    ) -> str:
        """
        Detect programming/document language from extension.
        """

        lower = path.lower()

        extensions = {
            ".py": "python",
            ".pyi": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".kts": "kotlin",
            ".dart": "dart",
            ".sql": "sql",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".md": "markdown",
            ".markdown": "markdown",
            ".rst": "rst",
            ".txt": "text",
        }

        for extension, language in extensions.items():
            if lower.endswith(extension):
                return language

        return "text"


# ============================================================
# Convenience function
# ============================================================

def chunk_document(
    content: str,
    path: str = "",
    category: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[dict]:
    """
    Convenience wrapper around DocumentChunker.
    """

    chunker = DocumentChunker(
        max_chars=max_chars,
        min_chars=min_chars,
        overlap_chars=overlap_chars,
    )

    return chunker.chunk_document(
        content=content,
        path=path,
        category=category,
    )


# ============================================================
# Debug / standalone testing
# ============================================================

if __name__ == "__main__":
   
     sample = ""
