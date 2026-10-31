FIXTURE_TOC = """1 Chapter 1: Fundamental Change in Education
1.1 Deepening Education and Learning
leaf 1.1.1 Origins and controversies
leaf 1.1.2 Providing pedagogical instruction
12 Chapter 12: Supporting Teachers and Teaching
leaf 12.7 Step five: master the technology
leaf 12.8 Step six: set appropriate learning goals
leaf 12.9 Step seven: design course structure and learning
13 Chapter 13: Making Decisions About Teaching and Learning
leaf 13.7 Building the future
"""

# paper Table 2 sample prompt shape
SAMPLE_PROMPT = """### Book: Teaching in a Digital Age

Pick the best section from the table of contents below
which can answer the user query below. Only generate
the section name, do not generate any explanation!

### Table of Contents
1 Chapter 1: Fundamental Change in Education
1.1 Deepening Education and Learning
leaf 1.1.1 Origins and controversies
...
leaf 12.9 Step seven: design course structure and learning
...
### Query: What is the suggested time allocation for students?
### Section: leaf
12.9 Step seven: design course structure and learning"""
