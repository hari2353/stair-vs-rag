import pytest
import fitz

from pdf_fixtures import make_pdf, write_pdf
from stair_vs_rag.ingest import extract_book, leaves_texts, to_paper_lines

TOC = [
    [1, "Fundamental Change in Education", 1],
    [2, "Deepening Education and Learning", 1],
    [3, "Origins and controversies", 1],
    [2, "Setting learning goals", 2],
]

PAGE_TEXT = {
    0: "Watson and Skinner argued that learning is a change in observable behaviour. "
       "Budget one hour per week of design time; set learning goals first. Future technology changes fast."
}


@pytest.fixture()
def book_path(tmp_path):
    p = tmp_path / "book.pdf"
    write_pdf(p, TOC, n_pages=2, page_text=PAGE_TEXT)
    return str(p)


def test_extract_book_numbering_and_leaves(book_path):
    entries = extract_book(book_path)[0]
    nums = [(e.num, e.title, e.is_leaf) for e in entries]
    assert ("1", "Fundamental Change in Education", False) == nums[0]
    assert ("1.1", "Deepening Education and Learning", False) == nums[1]
    assert ("1.1.1", "Origins and controversies", True) == nums[2]
    assert ("1.2", "Setting learning goals", True) == nums[3]


def test_to_paper_lines_match_stair_format(book_path):
    entries = extract_book(book_path)[0]
    lines = to_paper_lines(entries)
    assert "1 Fundamental Change in Education" == lines[0]
    assert "leaf 1.1.1 Origins and controversies" == lines[2]
    assert "leaf 1.2 Setting learning goals" == lines[3]


def test_leaf_text_scoped_to_section_pages(book_path):
    leaves = leaves_texts(book_path)
    assert "1.1.1 Origins and controversies" in leaves
    assert "Watson and Skinner" in leaves["1.1.1 Origins and controversies"]
