from typing import List, Dict, Any
from app.services.parser import PageText
from app.core.config import settings


class Chunk:
    def __init__(self, content: str, page_number: int, chunk_index: int, metadata: Dict[str, Any] = None):
        self.content = content
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.metadata = metadata or {}


class DocumentChunker:
    """Splits extracted page text into overlapping semantic chunks while strictly preserving page references."""

    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> List[str]:
        """Recursive character splitting strategy using paragraphs, lines, sentences, and words."""
        separators = ["\n\n", "\n", ". ", "; ", " ", ""]
        
        def recursive_split(current_text: str, sep_idx: int) -> List[str]:
            if len(current_text) <= self.chunk_size or sep_idx >= len(separators):
                return [current_text] if current_text else []
            
            separator = separators[sep_idx]
            splits = current_text.split(separator) if separator else list(current_text)
            
            chunks = []
            current_chunk = ""
            
            for part in splits:
                candidate = (current_chunk + separator + part) if current_chunk else part
                if len(candidate) <= self.chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    # If single part is larger than chunk_size, split it with next separator
                    if len(part) > self.chunk_size and sep_idx + 1 < len(separators):
                        sub_chunks = recursive_split(part, sep_idx + 1)
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = part
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks

        raw_chunks = recursive_split(text, 0)
        
        # Apply overlapping window
        if self.chunk_overlap <= 0 or len(raw_chunks) <= 1:
            return [c for c in raw_chunks if c]

        overlapped_chunks = []
        for i, chunk in enumerate(raw_chunks):
            if i > 0 and self.chunk_overlap > 0:
                prev_overlap = raw_chunks[i - 1][-self.chunk_overlap:]
                combined = f"{prev_overlap} {chunk}".strip()
                overlapped_chunks.append(combined)
            else:
                overlapped_chunks.append(chunk)

        return [c for c in overlapped_chunks if len(c) > 10]

    def chunk_pages(self, pages: List[PageText]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        global_index = 0

        for page in pages:
            page_text_splits = self._split_text(page.text)
            for split_content in page_text_splits:
                all_chunks.append(
                    Chunk(
                        content=split_content,
                        page_number=page.page_number,
                        chunk_index=global_index,
                        metadata={"char_count": len(split_content)},
                    )
                )
                global_index += 1

        return all_chunks
