from typing import List, Dict, Any
import io
import fitz  # PyMuPDF
import docx2txt
from app.core.logging import app_logger


class PageText:
    def __init__(self, page_number: int, text: str):
        self.page_number = page_number
        self.text = text


class DocumentParser:
    """Extracts text content page-by-page from PDFs, DOCX, and TXT files, preserving exact page boundaries."""

    @staticmethod
    def parse_pdf(file_bytes: bytes) -> List[PageText]:
        pages: List[PageText] = []
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page_idx in range(len(doc)):
                    page = doc[page_idx]
                    page_text = page.get_text("text").strip()
                    # 1-based page numbering for human/client readability
                    if page_text:
                        pages.append(PageText(page_number=page_idx + 1, text=page_text))
            app_logger.info(f"Successfully parsed PDF: extracted {len(pages)} non-empty pages.")
            return pages
        except Exception as e:
            app_logger.error(f"Error parsing PDF with PyMuPDF: {e}")
            raise ValueError(f"Failed to parse PDF document: {str(e)}")

    @staticmethod
    def parse_docx(file_bytes: bytes) -> List[PageText]:
        try:
            text = docx2txt.process(io.BytesIO(file_bytes)).strip()
            # DOCX does not naturally have physical page breaks; treat as page 1 or split on form feeds
            return [PageText(page_number=1, text=text)] if text else []
        except Exception as e:
            app_logger.error(f"Error parsing DOCX: {e}")
            raise ValueError(f"Failed to parse DOCX document: {str(e)}")

    @staticmethod
    def parse_txt(file_bytes: bytes) -> List[PageText]:
        try:
            text = file_bytes.decode("utf-8", errors="replace").strip()
            return [PageText(page_number=1, text=text)] if text else []
        except Exception as e:
            app_logger.error(f"Error parsing TXT: {e}")
            raise ValueError(f"Failed to parse TXT document: {str(e)}")

    @classmethod
    def parse(cls, filename: str, file_bytes: bytes) -> List[PageText]:
        lower_name = filename.lower()
        if lower_name.endswith(".pdf"):
            return cls.parse_pdf(file_bytes)
        elif lower_name.endswith(".docx"):
            return cls.parse_docx(file_bytes)
        elif lower_name.endswith((".txt", ".md", ".csv")):
            return cls.parse_txt(file_bytes)
        else:
            raise ValueError(f"Unsupported file format: {filename}. Supported: PDF, DOCX, TXT, MD, CSV.")
