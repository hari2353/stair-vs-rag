import fitz  # pymupdf


def make_pdf(toc, n_pages=3, page_text=None):
    """Synthetic PDF with an outline + distinct text per page."""
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        text = (page_text or {}).get(i, f"PAGE{i} filler content about topic {i}")
        page.insert_text((72, 72), text)
    doc.set_toc(toc)
    return doc.tobytes()


def write_pdf(path, toc, n_pages=3, page_text=None):
    open(path, "wb").write(make_pdf(toc, n_pages, page_text))
