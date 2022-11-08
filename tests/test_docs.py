# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase

BASE_DIR = Path(__file__).parents[1]


class TestDocs(TestCase):
    def test_readme(self) -> None:
        self._out = ""

        path = BASE_DIR / "README.md"
        if not path.exists():
            self.fail(f"README file not found: {path}")
        self.check_code_snippets_in_file(path)

    def check_code_snippets_in_file(self, doc_path: Path) -> None:  # noqa: C901
        # Extract lines of Python code from the README.md file.

        lines = []
        num_code_lines = 0
        num_code_lines_in_block = 0
        is_code = False
        is_md = False
        is_rst = False
        with doc_path.open() as doc_file:
            for line_index, orig_line in enumerate(doc_file):
                line = orig_line.strip("\n")
                if line.startswith("```python"):
                    # Start markdown code block.
                    if is_rst:
                        self.fail(
                            "Markdown code block found after restructured text block "
                            "in same file."
                        )
                    is_code = True
                    is_md = True
                    line = ""
                    num_code_lines_in_block = 0
                elif is_code and is_md and line.startswith("```"):
                    # Finish markdown code block.
                    if not num_code_lines_in_block:
                        self.fail(f"No lines of code in block: {line_index + 1}")
                    is_code = False
                    line = ""
                elif is_code and is_rst and line.startswith("```"):
                    # Can't finish restructured text block with markdown.
                    self.fail(
                        "Restructured text block terminated with markdown format '```'"
                    )

                elif is_code:
                    # Process line in code block.
                    if is_rst:
                        # Restructured code block normally indented with four spaces.
                        if len(line.strip()):
                            if not line.startswith("    "):
                                self.fail(
                                    f"Code line needs 4-char indent: {repr(line)}: "
                                    f"{doc_path}"
                                )
                            # Strip four chars of indentation.
                            line = line[4:]

                    if len(line.strip()):
                        num_code_lines_in_block += 1
                        num_code_lines += 1
                else:
                    line = ""
                lines.append(line)

        print(f"{num_code_lines} lines of code in {doc_path}")

        # Write the code into a temp file.
        tempfile = NamedTemporaryFile("w+")
        source = "\n".join(lines) + "\n"
        tempfile.writelines(source)
        tempfile.flush()

        exec(
            compile(source=source, filename=doc_path, mode="exec"), globals(), globals()
        )
