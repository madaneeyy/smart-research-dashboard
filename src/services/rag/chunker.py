class TextChunker:
    """
    Splits research/GitHub context into smaller chunks
    suitable for retrieval.
    """

    DEFAULT_CHUNK_SIZE = 250
    DEFAULT_OVERLAP = 40

    @classmethod
    def chunk_text(
        cls,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> list[str]:
        """
        Split text into overlapping chunks.

        Parameters
        ----------
        text:
            Input text.

        chunk_size:
            Approximate number of words per chunk.

        overlap:
            Number of words shared between consecutive chunks.

        Returns
        -------
        list[str]
            List of text chunks.
        """

        if not text or not text.strip():
            return []

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0"
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative"
            )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        words = text.split()

        chunks = []

        start = 0

        step = chunk_size - overlap

        while start < len(words):

            end = start + chunk_size

            chunk = " ".join(
                words[start:end]
            )

            if chunk.strip():
                chunks.append(chunk)

            start += step

        return chunks