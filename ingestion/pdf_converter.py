# ingestion/pdf_converter.py
# Converts PDF pages into images for ColPali embedding.
# Each page becomes one "chunk" — ColPali's page-level embedding IS the chunking strategy.

import fitz  # pymupdf
from PIL import Image
import io
from config import get_settings


def pdf_to_images(pdf_path: str) -> list[tuple[int, Image.Image]]:
    """
    Render each page of a PDF to a PIL Image.

    Returns a list of (page_number, image) tuples (1-indexed page numbers).
    """
    settings = get_settings()
    doc = fitz.open(pdf_path)
    pages: list[tuple[int, Image.Image]] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        mat = fitz.Matrix(settings.dpi / 72, settings.dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        pages.append((page_index + 1, img))

    doc.close()
    return pages
