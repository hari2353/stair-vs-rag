"""The STAIR prompt, exactly per the paper's Table 2 training sample:
prompt + (query, ToC), completion = the correct leaf entry."""
from __future__ import annotations

_TEMPLATE = """
Pick the best section from the table of contents below
which can answer the user query below. Only generate
the section name, do not generate any explanation!
"""


def build_stair_prompt(book: str, toc_lines: list[str], query: str) -> str:
    toc = "\n".join(toc_lines)
    return (
        f"### Book: {book}\n"
        f"{_TEMPLATE}"
        f"\n### Table of Contents\n{toc}\n"
        f"\n### Query: {query}\n"
        "### Section:"
    )
