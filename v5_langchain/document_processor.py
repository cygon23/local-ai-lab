"""
document_processor.py
---------------------
Same as V4 — splits documents into chunks for indexing.
No LangChain changes here; we keep it consistent with our ChromaDB schema.
"""

import re
from pathlib import Path

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def process_document(file_path: str, source_name: str) -> list[dict]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = _read_pdf(file_path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    chunks = _chunk_text(text, source_name)
    return chunks


def _read_pdf(file_path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return f"PDF read error: {e}"


def _chunk_text(text: str, source: str) -> list[dict]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "chunk_index": chunk_index,
                    "start_char": start,
                }
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_index += 1
    return chunks