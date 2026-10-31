"""Extract ToC + per-leaf text from PDFs via pymupdf, mirroring the paper's
parsing approach (it used pymupdf for SearchTome books too)."""
from __future__ import annotations

import fitz

from .toc import Entry, ToC


def extract_book(pdf_path: str) -> tuple[list[Entry], "fitz.Document"]:
    doc = fitz.open(pdf_path)
    raw = doc.get_toc()  # [level, title, page1-based]
    entries, counters, pages = [], {}, []
    for i, (level, title, page) in enumerate(raw):
        counters[level] = counters.get(level, 0) + 1
        for k in list(counters):
            if k > level:
                counters.pop(k)
        num = ".".join(str(counters[k]) for k in sorted(counters) if k <= level)
        is_leaf = all(raw[j][0] <= level for j in range(i + 1, len(raw)))
        entries.append(Entry(num, title, is_leaf, f"{num} {title}"))
        pages.append(page)
    return entries, doc


def to_paper_lines(entries: list[Entry]) -> list[str]:
    return [("leaf " if e.is_leaf else "") + e.render for e in entries]


def leaves_texts(pdf_path: str) -> dict[str, str]:
    """leaf '<num> <title>' -> text of its page range (page i .. next entry page-1)."""
    entries, doc = extract_book(pdf_path)
    raw = doc.get_toc()
    texts: dict[str, str] = {}
    bounds = [p for _, _, p in raw] + [doc.page_count + 1]
    for i, e in enumerate(entries):
        if not e.is_leaf:
            continue
        start, end = bounds[i] - 1, min(bounds[i + 1] - 1, doc.page_count)
        text = " ".join(doc[p].get_text() for p in range(start, end)).strip()
        texts[e.render] = text
    return texts
