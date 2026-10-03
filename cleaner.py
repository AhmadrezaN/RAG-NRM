"""
PDF Preprocessing & Page-Aware Structural Chunking Module
"""

import re
import pymupdf as fitz
from config import BOOK_STRUCTURE, PDF_PATH
from collections import Counter


def extract_and_clean_page(page: fitz.Page) -> list[dict]:
    """
    Extracts text line-by-line from a PDF page, removes headers/footers,
    and identifies paragraph indents.
    """
    page_dict = page.get_text("dict")
    raw_lines = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # Text blocks only
            continue
        for line in block.get("lines", []):
            line_text = "".join([span["text"] for span in line.get("spans", [])]).strip()
            if line_text:
                x_start = round(line["bbox"][0], 1)
                raw_lines.append({"text": line_text, "x": x_start})

    if not raw_lines:
        return []

    # Find dominant margin to flag indents
    x_counts = Counter([item["x"] for item in raw_lines])
    standard_margin = x_counts.most_common(1)[0][0] if x_counts else 0.0

    cleaned_lines = []
    for item in raw_lines:
        text = item["text"]
        x_pos = item["x"]

        # Filter running headers/footers/metadata
        if re.search(
            r"(New Religious Movements|Elements in New Religious Movements|Glossary of Technical Terms|References)",
            text,
            re.IGNORECASE,
        ):
            continue
        if re.search(
            r"(Shaftesbury Road|Cambridge|Downloaded from|https://|IP address:)",
            text,
            re.IGNORECASE,
        ):
            continue
        if re.match(r"^\d+$", text):  # Page numbers
            continue

        is_indent = (x_pos - standard_margin) >= 8.0
        cleaned_lines.append({"text": text, "is_indent": is_indent})

    return cleaned_lines


def reconstruct_section_paragraphs(sub_lines: list[dict]) -> list[str]:
    """
    Reconstructs continuous paragraphs for a given set of page lines.
    Handles hyphenation across lines and paragraph indents.
    """
    paragraphs = []
    current_lines = []

    for item in sub_lines:
        text = item["text"].strip()
        is_indent = item["is_indent"]

        if not text:
            continue

        if is_indent and current_lines:
            full_p = " ".join(current_lines)
            paragraphs.append(re.sub(r"\s+", " ", full_p))
            current_lines = [text]
        else:
            if current_lines:
                last_line = current_lines[-1]
                if last_line.endswith("-") and not last_line.endswith(" -"):
                    current_lines[-1] = last_line[:-1] + text  # Merge hyphenated words
                else:
                    current_lines.append(text)
            else:
                current_lines.append(text)

    if current_lines:
        full_p = " ".join(current_lines)
        paragraphs.append(re.sub(r"\s+", " ", full_p))

    return paragraphs


def slice_and_chunk_subsections(
    doc: fitz.Document,
    book_structure: list[dict],
    max_words: int = 300,
    overlap_words: int = 40,
) -> list[dict]:
    """
    Extracts text per subsection using page ranges directly from BOOK_STRUCTURE,
    reconstructs paragraphs, and creates chunks with metadata.
    """
    chunks = []

    for section in book_structure:
        section_id = section.get("section_id")
        chapter_title = section.get("title", "Unknown Chapter")
        subsections = section.get("subsections", [])

        for sub in subsections:
            sub_title = sub.get("title")
            start_page = sub.get("start_page", 1)
            end_page = sub.get("end_page", start_page)

            if not sub_title:
                continue

            # 1. Extract lines ONLY from the subsection's specific PDF page range (0-indexed)
            sub_lines = []
            for p_num in range(start_page - 1, end_page):
                if p_num < len(doc):
                    page_lines = extract_and_clean_page(doc[p_num])
                    sub_lines.extend(page_lines)

            if not sub_lines:
                print(f"[Warning] No text extracted for '{chapter_title} -> {sub_title}' (pages {start_page}-{end_page})")
                continue

            # 2. Reconstruct clean paragraphs for this subsection
            paragraphs = reconstruct_section_paragraphs(sub_lines)

            # Prepend sub_title header to the very first paragraph of the subsection
            if paragraphs:
                paragraphs[0] = f"### {sub_title}\n\n" + paragraphs[0]

            # 3. Chunk paragraphs into blocks of <= max_words
            current_chunk_paragraphs = []
            current_word_count = 0
            chunk_idx = 0

            for paragraph in paragraphs:
                p_words = paragraph.split()
                p_word_count = len(p_words)

                # CASE A: Paragraph itself exceeds max_words -> Split with overlap window
                if p_word_count > max_words:
                    if current_chunk_paragraphs:
                        chunks.append({
                            "content": "\n\n".join(current_chunk_paragraphs),
                            "metadata": {
                                "section_id": section_id,
                                "chapter": chapter_title,
                                "subsection": sub_title,
                                "start_page": start_page,
                                "end_page": end_page,
                                "chunk_index": chunk_idx,
                            },
                        })
                        chunk_idx += 1
                        current_chunk_paragraphs = []
                        current_word_count = 0

                    step = max_words - overlap_words
                    for i in range(0, p_word_count, step):
                        sub_p_words = p_words[i : i + max_words]
                        chunks.append({
                            "content": " ".join(sub_p_words),
                            "metadata": {
                                "section_id": section_id,
                                "chapter": chapter_title,
                                "subsection": sub_title,
                                "start_page": start_page,
                                "end_page": end_page,
                                "chunk_index": chunk_idx,
                            },
                        })
                        chunk_idx += 1
                        if i + max_words >= p_word_count:
                            break
                    continue

                # CASE B: Paragraph fits into current chunk
                if current_word_count + p_word_count <= max_words:
                    current_chunk_paragraphs.append(paragraph)
                    current_word_count += p_word_count

                # CASE C: Exceeds max_words -> Flush chunk & start new one
                else:
                    chunks.append({
                        "content": "\n\n".join(current_chunk_paragraphs),
                        "metadata": {
                            "section_id": section_id,
                            "chapter": chapter_title,
                            "subsection": sub_title,
                            "start_page": start_page,
                            "end_page": end_page,
                            "chunk_index": chunk_idx,
                        },
                    })
                    chunk_idx += 1
                    current_chunk_paragraphs = [paragraph]
                    current_word_count = p_word_count

            # Flush remaining paragraphs
            if current_chunk_paragraphs:
                chunks.append({
                    "content": "\n\n".join(current_chunk_paragraphs),
                    "metadata": {
                        "section_id": section_id,
                        "chapter": chapter_title,
                        "subsection": sub_title,
                        "start_page": start_page,
                        "end_page": end_page,
                        "chunk_index": chunk_idx,
                    },
                })

    return chunks