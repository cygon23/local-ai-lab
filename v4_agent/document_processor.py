"""
document_processor.py
----------------------
Loads documents and splits them into chunks for indexing.

WHY DO WE CHUNK?
  You can't embed an entire 50-page PDF as one vector.
  One vector = one meaning. A 50-page document has many meanings.

  If you embed the whole thing, the vector is a blurry average of everything.
  When you search, you'll never get a precise match.

  Solution: split the document into small overlapping chunks.
  Each chunk gets its own vector. Search finds the right chunk.

THE CHUNKING STRATEGY WE USE — SLIDING WINDOW:

  Document text: "AAAA BBBB CCCC DDDD EEEE"
  Chunk size: 3 units, Overlap: 1 unit

  Chunk 1: "AAAA BBBB CCCC"
  Chunk 2:      "BBBB CCCC DDDD"   ← overlaps with chunk 1 by 1 unit
  Chunk 3:           "CCCC DDDD EEEE"

  WHY OVERLAP?
    If an important sentence falls right at the boundary between two chunks,
    without overlap it would be split and both chunks would have half a thought.
    Overlap ensures every sentence appears complete in at least one chunk.

CHUNK SIZE TRADEOFFS:
  Too small (e.g. 100 chars) → too granular, loses context
  Too large (e.g. 2000 chars) → loses precision, noisy embeddings
  Sweet spot: 500-1000 characters with 100-200 char overlap
  We use 800 chars / 150 char overlap — good for most documents.

SUPPORTED FILE TYPES (V3):
  - .txt  → plain text, read directly
  - .md   → markdown, read as text
  - .pdf  → requires pypdf
  - .py   → source code files (useful for coding assistants)
  Future: .docx, .csv, .json (V4+)
"""

import re
from pathlib import Path


CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # character overlap between chunks


def load_text_file(filepath: str) -> str:
    """Load a plain text or markdown file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_pdf(filepath: str) -> str:
    """
    Extract text from a PDF.
    Requires: pip install pypdf
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError("pypdf is required for PDF support. Run: pip install pypdf")


def load_document(filepath: str) -> tuple[str, str]:
    """
    Load a document and return (text_content, file_type).
    Dispatches to the right loader based on file extension.
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext in [".txt", ".md", ".py", ".js", ".ts", ".json", ".csv"]:
        return load_text_file(filepath), ext
    elif ext == ".pdf":
        return load_pdf(filepath), ".pdf"
    else:
        # Try to load as text anyway
        try:
            return load_text_file(filepath), ext
        except Exception:
            raise ValueError(f"Unsupported file type: {ext}")


def clean_text(text: str) -> str:
    """
    Basic text cleaning before chunking.
    - Collapse multiple blank lines into one
    - Strip leading/trailing whitespace
    - Normalize line endings
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)   # max 2 consecutive newlines
    text = re.sub(r" {2,}", " ", text)         # collapse multiple spaces
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Split text into overlapping chunks by character count.

    Returns a list of chunk strings.

    HOW IT WORKS:
      Start at position 0.
      Take `chunk_size` characters → that's chunk 1.
      Move forward by (chunk_size - overlap) → start of chunk 2.
      Repeat until end of document.

    We try to break at sentence/word boundaries when possible
    to avoid cutting words in half.
    """
    text = clean_text(text)
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            # Last chunk — take everything remaining
            chunks.append(text[start:].strip())
            break

        # Try to break at a sentence boundary (. ! ?)
        # Look backwards from `end` for a period
        break_point = end
        for i in range(end, max(start + overlap, end - 200), -1):
            if text[i] in ".!?\n":
                break_point = i + 1
                break

        chunk = text[start:break_point].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward, accounting for overlap
        start = break_point - overlap

    return chunks


def process_document(filepath: str, filename: str = None) -> list[dict]:
    """
    Full pipeline: load → clean → chunk → return list of chunk dicts.

    Each chunk dict contains everything ChromaDB needs:
    {
        "text": "the actual chunk content",
        "source": "filename.pdf",
        "chunk_index": 0,
        "total_chunks": 12,
        "chunk_id": "filename.pdf_chunk_0"
    }

    WHY INCLUDE METADATA?
      When ChromaDB retrieves a chunk, we want to show the user
      WHERE the answer came from. "According to your file X, page/chunk Y..."
      This is called provenance — essential for trustworthy RAG.
    """
    if filename is None:
        filename = Path(filepath).name

    text, file_type = load_document(filepath)
    chunks = chunk_text(text)

    chunk_dicts = []
    for i, chunk_text_content in enumerate(chunks):
        chunk_dicts.append({
            "text": chunk_text_content,
            "source": filename,
            "file_type": file_type,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_id": f"{filename}_chunk_{i}"
        })

    return chunk_dicts