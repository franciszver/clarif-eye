"""Every row in the README's capability table that claims a capability is
"demonstrated" must name a real, importable symbol (issue #79 / P9.0).

The table lives under the "## LangGraph capabilities this repo demonstrates"
heading in README.md, one markdown table, with each row's second column
holding one or more ``module.symbol`` references as inline code spans
(backticks). This test parses that table, then for every reference actually
imports the module and getattr()s the symbol - a row can claim whatever
prose it likes, but a symbol that doesn't exist makes the row, and the
test, fail.
"""

import importlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"

TABLE_HEADING = "## LangGraph capabilities this repo demonstrates"

# A reference is a backtick-quoted dotted path with at least one dot, e.g.
# `clarif_eye.graph.build_graph`. Requiring a dot excludes stray inline
# code spans (a filename, a single word) that aren't meant to be a
# module.symbol reference at all.
_REFERENCE_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*\.[A-Za-z_][A-Za-z0-9_]*)`")


def _capability_table_text():
    """The raw text of the table under TABLE_HEADING, up to the next
    "## " heading or end of file."""
    text = README_PATH.read_text(encoding="utf-8")
    assert TABLE_HEADING in text, f"README.md is missing the {TABLE_HEADING!r} heading"
    after = text.split(TABLE_HEADING, 1)[1]
    return after.split("\n## ", 1)[0]


def _table_rows(table_text):
    """Every markdown table row (a line starting with '|'), excluding the
    header row and the '|---|---|' separator row."""
    rows = []
    for line in table_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if re.fullmatch(r"\|[\s:|-]+\|", stripped):
            continue  # separator row, e.g. |---|---|
        rows.append(stripped)
    # First remaining row is the header (e.g. "| Capability | Demonstrated by |").
    assert rows, "README.md's capability table has no rows"
    return rows[1:]


def _symbol_references(table_text):
    """Every `module.symbol` reference found anywhere in the table."""
    return _REFERENCE_RE.findall(table_text)


def test_capability_table_exists_with_rows():
    table_text = _capability_table_text()
    rows = _table_rows(table_text)
    assert len(rows) >= 5, "expected several capability rows, found " f"{len(rows)}"


def test_every_referenced_symbol_is_importable():
    table_text = _capability_table_text()
    references = _symbol_references(table_text)
    assert references, "no `module.symbol` references found in the capability table"

    failures = []
    for reference in references:
        module_name, symbol_name = reference.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            failures.append(f"{reference}: module {module_name!r} does not import ({exc})")
            continue
        if not hasattr(module, symbol_name):
            failures.append(f"{reference}: {module_name!r} has no attribute {symbol_name!r}")

    assert not failures, "\n".join(failures)


def test_every_row_naming_a_demonstrated_capability_has_a_reference():
    """Every row in the table must cite at least one `module.symbol`
    reference - a row with none would be an unverifiable claim."""
    table_text = _capability_table_text()
    rows = _table_rows(table_text)

    failures = [row for row in rows if not _REFERENCE_RE.search(row)]
    assert not failures, "rows with no importable symbol reference:\n" + "\n".join(failures)


def test_a_row_pointing_at_a_nonexistent_symbol_is_caught():
    """Mutation proof: a reference to a symbol that does not exist must
    fail the importability check, so the check above is not a no-op."""
    fake_reference = "clarif_eye.graph.this_symbol_does_not_exist"
    module_name, symbol_name = fake_reference.rsplit(".", 1)
    module = importlib.import_module(module_name)
    assert not hasattr(module, symbol_name), (
        "test setup is broken: the fake symbol actually exists"
    )
