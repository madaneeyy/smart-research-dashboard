"""
Export persistent document chunks from Supabase for retrieval evaluation.

Usage:
    python export_document_chunks.py 32ee9c3813254ab589f67a58f98e41c5

The script reads SUPABASE_URL and SUPABASE_KEY from the project's .env.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client


def main():
    if len(sys.argv) != 2:
        print("Usage: python export_document_chunks.py <document_id>")
        raise SystemExit(2)

    document_id = sys.argv[1].strip()
    load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY")

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be present in .env"
        )

    supabase = create_client(url, key)

    response = (
        supabase.table("document_chunks")
        .select("chunk_index,content,page,document_id")
        .eq("document_id", document_id)
        .order("chunk_index")
        .execute()
    )

    chunks = response.data or []

    if not chunks:
        raise RuntimeError(
            f"No chunks found for document_id={document_id}"
        )

    output = Path(f"fyp_chunks_{document_id[:8]}.json")
    output.write_text(
        json.dumps({"chunks": chunks}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Exported {len(chunks)} chunks")
    print(f"File: {output.resolve()}")


if __name__ == "__main__":
    main()
