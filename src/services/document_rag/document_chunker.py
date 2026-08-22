from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


class UploadedDocumentChunker:
    """
    Standalone chunker for uploaded documents.

    IMPORTANT:
    - No GitHub imports.
    - No dependency on the project's GitHub DocumentChunker.
    - Works with arbitrary extracted text supplied by DocumentService.
    - Preserves pages, headings, sections, paragraphs and tables when the
      extractor exposes them in the text.
    - Uses hierarchical + sliding-window chunks so both tiny paragraphs and
      very long sections remain retrievable.
    """

    PAGE_RE = re.compile(
        r"(?:^|\n)\s*\[Page\s+(\d+)\]\s*\n?",
        re.IGNORECASE,
    )

    MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")

    NUMBERED_HEADING_RE = re.compile(
        r"^\s*((?:chapter|section)\s+[\w.-]+|"
        r"\d+(?:\.\d+){0,6}[.)]?)\s+(.{2,180})\s*$",
        re.IGNORECASE,
    )

    ALL_CAPS_HEADING_RE = re.compile(
        r"^\s*([A-Z][A-Z0-9 &:/,().'\-]{3,140})\s*$"
    )

    TABLE_LINE_RE = re.compile(r"^\s*(?:\|.*\||.*\t.*|.* {3,}.*)$")

    def __init__(
        self,
        target_chars: int = 1600,
        max_chars: int = 2600,
        overlap_chars: int = 300,
        min_chunk_chars: int = 20,
        min_chars: int | None = None,
    ) -> None:
        # Older main.py versions passed min_chars=180. That value was
        # accidentally acting as a hard discard threshold, which is bad for
        # short but meaningful paragraphs. Keep the argument accepted for
        # compatibility, but use the document-RAG minimum instead.
        # The real minimum is controlled by min_chunk_chars.
        self.target_chars = max(600, target_chars)
        self.max_chars = max(self.target_chars, max_chars)
        self.overlap_chars = max(
            0, min(overlap_chars, self.max_chars // 2)
        )
        self.min_chunk_chars = max(1, min_chunk_chars)

    def chunk_documents(
        self,
        documents: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []

        for document in documents:
            result.extend(self.chunk_document(document))

        self._renumber(result)
        return result

    def chunk_document(
        self,
        document: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        content = self._clean(str(document.get("content") or ""))
        if not content:
            return []

        filename = str(
            document.get("filename")
            or document.get("path")
            or "uploaded_document"
        )
        document_id = str(
            document.get("document_id")
            or document.get("id")
            or filename
        )

        pages = self._split_pages(content)
        chunks: List[Dict[str, Any]] = []

        for page_number, page_text in pages:
            page_text = self._clean(page_text)
            if not page_text:
                continue

            sections = self._split_sections(page_text)

            for section_title, section_body, heading_level in sections:
                body = self._clean(section_body)
                if not body:
                    continue

                # Keep heading attached to the first chunk. This is extremely
                # useful when the user asks about "Methodology", "Results",
                # "Chapter 3", etc.
                prefix = (
                    f"{section_title}\n\n"
                    if section_title
                    else ""
                )

                pieces = self._make_pieces(body)

                if not pieces:
                    pieces = [body]

                for piece_index, piece in enumerate(pieces):
                    text = (
                        prefix + piece
                        if piece_index == 0 and prefix
                        else piece
                    ).strip()

                    if len(text) < self.min_chunk_chars:
                        continue

                    chunks.append(
                        {
                            "content": text,
                            "source": "upload",
                            "source_type": "uploaded_document",
                            "document_id": document_id,
                            "filename": filename,
                            "document_path": document.get(
                                "path", filename
                            ),
                            "content_type": document.get(
                                "content_type", ""
                            ),
                            "page": page_number,
                            "page_start": page_number,
                            "page_end": page_number,
                            "section": section_title,
                            "heading": section_title,
                            "heading_level": heading_level,
                            "section_path": (
                                [section_title]
                                if section_title
                                else []
                            ),
                            "chunk_index_in_section": piece_index,
                            "char_count": len(text),
                        }
                    )

        self._renumber(chunks)
        return chunks

    # ------------------------------------------------------------------
    # Page / section parsing
    # ------------------------------------------------------------------

    def _split_pages(
        self,
        text: str,
    ) -> List[Tuple[Optional[int], str]]:
        matches = list(self.PAGE_RE.finditer(text))

        if not matches:
            return [(None, text)]

        pages: List[Tuple[Optional[int], str]] = []

        prefix = text[: matches[0].start()].strip()
        if prefix:
            pages.append((None, prefix))

        for i, match in enumerate(matches):
            start = match.end()
            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(text)
            )

            page_text = text[start:end].strip()
            if page_text:
                pages.append((int(match.group(1)), page_text))

        return pages

    def _split_sections(
        self,
        text: str,
    ) -> List[Tuple[str, str, Optional[int]]]:
        lines = text.splitlines()

        sections: List[Tuple[str, str, Optional[int]]] = []
        current_title = ""
        current_level: Optional[int] = None
        current_lines: List[str] = []

        for line in lines:
            heading = self._detect_heading(line)

            if heading is not None:
                if current_lines:
                    sections.append(
                        (
                            current_title,
                            "\n".join(current_lines),
                            current_level,
                        )
                    )
                    current_lines = []

                current_title, current_level = heading
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(
                (
                    current_title,
                    "\n".join(current_lines),
                    current_level,
                )
            )

        return sections

    def _detect_heading(
        self,
        line: str,
    ) -> Optional[Tuple[str, int]]:
        stripped = line.strip()

        if not stripped or len(stripped) > 180:
            return None

        match = self.MARKDOWN_HEADING_RE.match(line)
        if match:
            return match.group(2).strip(), len(match.group(1))

        match = self.NUMBERED_HEADING_RE.match(line)
        if match:
            return (
                f"{match.group(1)} {match.group(2)}".strip(),
                match.group(1).count(".") + 1,
            )

        # Plain extracted PDFs often lose font/style metadata.
        # Short title-case lines are therefore treated as headings too.
        if (
            1 <= len(stripped.split()) <= 10
            and len(stripped) <= 100
            and not stripped.endswith((".", "?", "!", ":"))
            and any(ch.isalpha() for ch in stripped)
            and (
                stripped.istitle()
                or self.ALL_CAPS_HEADING_RE.fullmatch(stripped)
            )
        ):
            return stripped, 1

        return None

    # ------------------------------------------------------------------
    # Chunk construction
    # ------------------------------------------------------------------

    def _make_pieces(self, text: str) -> List[str]:
        paragraphs = self._paragraphs(text)

        if not paragraphs:
            return []

        pieces: List[str] = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.max_chars:
                if current:
                    pieces.append(current)
                    current = ""

                pieces.extend(self._split_long_block(paragraph))
                continue

            candidate = (
                f"{current}\n\n{paragraph}".strip()
                if current
                else paragraph
            )

            if len(candidate) <= self.target_chars:
                current = candidate
            else:
                if current:
                    pieces.append(current)

                # Keep a small paragraph together even if it is above target.
                current = paragraph

        if current:
            pieces.append(current)

        return self._add_overlap(pieces)

    def _split_long_block(self, text: str) -> List[str]:
        """
        Split very long paragraphs/code/table blocks without losing content.
        Prefer sentence boundaries; fall back to character windows.
        """
        sentences = re.split(
            r"(?<=[.!?。！？])\s+(?=[A-Z0-9\"'(\[])",
            text,
        )

        if len(sentences) == 1:
            return self._window_split(text)

        pieces: List[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > self.max_chars:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.extend(self._window_split(sentence))
                continue

            candidate = (
                f"{current} {sentence}".strip()
                if current
                else sentence
            )

            if len(candidate) <= self.target_chars:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = sentence

        if current:
            pieces.append(current)

        return self._add_overlap(pieces)

    def _window_split(self, text: str) -> List[str]:
        step = max(1, self.max_chars - self.overlap_chars)
        pieces: List[str] = []

        for start in range(0, len(text), step):
            piece = text[start : start + self.max_chars].strip()
            if piece:
                pieces.append(piece)

            if start + self.max_chars >= len(text):
                break

        return pieces

    def _add_overlap(self, pieces: List[str]) -> List[str]:
        if len(pieces) <= 1 or self.overlap_chars <= 0:
            return pieces

        result = [pieces[0]]

        for piece in pieces[1:]:
            previous = result[-1]
            overlap = previous[-self.overlap_chars :].strip()

            if overlap and len(overlap) + len(piece) + 2 <= self.max_chars:
                result.append(f"{overlap}\n\n{piece}")
            else:
                result.append(piece)

        return result

    @staticmethod
    def _paragraphs(text: str) -> List[str]:
        raw = re.split(r"\n\s*\n+", text)

        result = []
        for paragraph in raw:
            paragraph = UploadedDocumentChunker._clean(paragraph)
            if paragraph:
                result.append(paragraph)

        return result

    @staticmethod
    def _clean(text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _renumber(chunks: List[Dict[str, Any]]) -> None:
        for index, chunk in enumerate(chunks):
            chunk["document_chunk_index"] = index
            document_id = chunk.get("document_id", "document")
            chunk["chunk_id"] = f"{document_id}:{index}"
