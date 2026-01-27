"""
PDF extraction module using PyMuPDF.

Extracts embedded images and page renders from PDF documents.
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


@dataclass
class ExtractedImage:
    """An image extracted from a PDF."""

    image: Image.Image
    page_num: int
    index: int  # Index within the page
    bbox: Optional[tuple[float, float, float, float]] = None  # x0, y0, x1, y1
    xref: Optional[int] = None  # PDF object reference


@dataclass
class ExtractedPage:
    """A rendered page from a PDF."""

    page_num: int
    image: Image.Image
    width: int
    height: int
    embedded_images: list[ExtractedImage] = field(default_factory=list)
    text: str = ""


class PDFExtractor:
    """Extract images and text from PDF documents using PyMuPDF."""

    def __init__(self, dpi: int = 150, extract_text: bool = True):
        """
        Initialize the PDF extractor.

        Args:
            dpi: Resolution for page rendering
            extract_text: Whether to extract text from pages
        """
        if fitz is None:
            raise ImportError(
                "PyMuPDF is required for PDF extraction. "
                "Install with: pip install PyMuPDF"
            )
        self.dpi = dpi
        self.extract_text = extract_text
        self._zoom = dpi / 72.0  # 72 is the default PDF DPI

    def extract(self, pdf_path: Path | str) -> Iterator[ExtractedPage]:
        """
        Extract pages and images from a PDF.

        Args:
            pdf_path: Path to the PDF file

        Yields:
            ExtractedPage objects for each page
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        try:
            for page_num in range(len(doc)):
                yield self._extract_page(doc, page_num)
        finally:
            doc.close()

    def _extract_page(self, doc: "fitz.Document", page_num: int) -> ExtractedPage:
        """Extract a single page from the document."""
        page = doc[page_num]

        # Render page as image
        mat = fitz.Matrix(self._zoom, self._zoom)
        pix = page.get_pixmap(matrix=mat)
        page_image = Image.open(io.BytesIO(pix.tobytes("png")))

        # Extract text if requested
        text = page.get_text() if self.extract_text else ""

        # Extract embedded images
        embedded_images = self._extract_embedded_images(page, page_num)

        return ExtractedPage(
            page_num=page_num,
            image=page_image,
            width=pix.width,
            height=pix.height,
            embedded_images=embedded_images,
            text=text,
        )

    def _extract_embedded_images(
        self, page: "fitz.Page", page_num: int
    ) -> list[ExtractedImage]:
        """Extract embedded images from a page."""
        images = []
        image_list = page.get_images(full=True)

        for idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
                if base_image:
                    image_bytes = base_image["image"]
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                    # Try to get bounding box
                    bbox = None
                    for img_rect in page.get_image_rects(xref):
                        bbox = (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1)
                        break

                    images.append(
                        ExtractedImage(
                            image=pil_image,
                            page_num=page_num,
                            index=idx,
                            bbox=bbox,
                            xref=xref,
                        )
                    )
            except Exception:
                # Skip images that can't be extracted
                continue

        return images

    def get_page_count(self, pdf_path: Path | str) -> int:
        """Get the number of pages in a PDF."""
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count

    def extract_text_only(self, pdf_path: Path | str) -> str:
        """Extract all text from a PDF without images."""
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)
