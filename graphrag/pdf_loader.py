import re
from pathlib import Path

import fitz  # PyMuPDF

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_MAX_PAGES


def read_pdf(pdf_path, max_pages=DEFAULT_MAX_PAGES):
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = []
    with fitz.open(path) as pdf:
        page_limit = len(pdf) if max_pages == 0 else min(len(pdf), max_pages)
        print(f"Reading {page_limit} of {len(pdf)} PDF pages...")

        for page_index in range(page_limit):
            text = pdf[page_index].get_text("text").strip()
            if text:
                pages.append({"page": page_index + 1, "text": text})

    if not pages:
        raise ValueError(
            "No text extracted. The PDF may be scanned and require OCR."
        )
    return pages


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Create overlapping character chunks."""
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            sentence_boundary = text.rfind(". ", start, end)
            if sentence_boundary > start + chunk_size // 2:
                end = sentence_boundary + 1
            else:
                space_boundary = text.rfind(" ", start, end)
                if space_boundary > start + chunk_size // 2:
                    end = space_boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks
