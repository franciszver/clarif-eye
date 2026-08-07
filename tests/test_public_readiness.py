"""Checks for issue #19 / P7.1 (public-readiness): LICENSE and README exist
with the required content, and committed docs carry no process-meta or
internal-strategy language before the repository stops being private.

WORD LIST NOTES (process-meta check): "agent" alone is not included, because
"user agent" is ordinary web terminology that would false-positive on any
doc discussing browsers or HTTP. The check instead matches "the agent",
"the subagent", and "the orchestrator" as whole phrases, which are the
actual process-meta usages this project's working process produces.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DOCS_TO_CHECK = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "ACCESSIBILITY.md",
    REPO_ROOT / "docs" / "SCENARIOS.md",
    REPO_ROOT / "CONTRIBUTING.md",
]

# Process-meta / internal-strategy language that must not appear in a
# committed, public-facing doc. Scoped to whole phrases to avoid false
# positives (see module docstring for "agent").
PROCESS_META_PATTERNS = [
    re.compile(r"\bthe orchestrator\b", re.IGNORECASE),
    re.compile(r"\bthe subagent\b", re.IGNORECASE),
    re.compile(r"\bthe agent\b", re.IGNORECASE),
    re.compile(r"\bred-first\b", re.IGNORECASE),
    re.compile(r"\bfix-first\b", re.IGNORECASE),
    re.compile(r"\bthe owner\b", re.IGNORECASE),
    # Internal decision codes like "D7" or "D12".
    re.compile(r"\bD\d{1,2}\b"),
    # Internal phase/issue codes like "P7.1".
    re.compile(r"\bP\d\.\d\b"),
]


def test_license_file_exists_and_names_mit_and_copyright_holder():
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.exists(), "LICENSE file is missing"
    text = license_path.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Francisco Zver" in text
    assert "2026" in text


def test_readme_exists_and_documents_setup():
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.exists(), "README.md is missing"
    text = readme_path.read_text(encoding="utf-8")
    assert "pip install" in text
    assert "OPENROUTER_API_KEY" in text
    assert "python app.py" in text
    assert "LICENSE" in text


def test_readme_links_live_app_and_states_cold_start():
    readme_path = REPO_ROOT / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    url = "https://clarif-eye.onrender.com"
    assert url in text, "README does not link the deployed app URL"

    index = text.index(url)
    start = max(0, index - 300)
    end = min(len(text), index + 300)
    nearby = text[start:end]
    cold_start_pattern = re.compile(
        r"\b6\d(\.\d+)?\s*seconds?\b.{0,80}"
        r"(first request|wak(e|ing|es)|sleep(s|ing)?)"
        r"|"
        r"(first request|wak(e|ing|es)|sleep(s|ing)?).{0,80}"
        r"\b6\d(\.\d+)?\s*seconds?\b",
        re.IGNORECASE | re.DOTALL,
    )
    assert cold_start_pattern.search(nearby), (
        "README does not mention the cold-start wait (a number around 60 "
        "seconds together with wording about the first request / waking / "
        "sleeping) near the deployed URL"
    )


def test_committed_docs_have_no_process_meta_language():
    failures = []
    for path in DOCS_TO_CHECK:
        text = path.read_text(encoding="utf-8")
        for pattern in PROCESS_META_PATTERNS:
            match = pattern.search(text)
            if match:
                failures.append(f"{path.relative_to(REPO_ROOT)}: found {match.group(0)!r}")
    assert not failures, "\n".join(failures)
