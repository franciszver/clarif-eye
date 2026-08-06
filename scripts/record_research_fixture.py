"""MANUAL-ONLY fixture recorder for the research node.

Takes a query string, runs the REAL research node
(clarif_eye.research.run_research's search + fetch, via the real DDGS
search backend and a real httpx.Client) exactly once, and writes the
extracted text to tests/fixtures/, for later use as a recorded fixture in
offline tests.

This is NOT part of the pytest suite (tests must stay offline) and must NOT
be run by an automated agent - only by a human, or an orchestrator that has
explicitly decided to spend a real network lookup. It makes exactly one
search request and, at most, one page fetch (top result only - see
run_research's module docstring).

Usage:
    python scripts/record_research_fixture.py "some search query"

Writes:
    tests/fixtures/research_scraper_data.json - {"query": ..., "scraper_data": ...}
"""

import json
import sys
from pathlib import Path

from clarif_eye import research

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} \"some search query\"", file=sys.stderr)
        raise SystemExit(1)

    query = sys.argv[1]

    # Exactly one search call plus, at most, one page fetch: reuse the real
    # node end to end (run_research) rather than reimplementing its steps,
    # so the recorded fixture matches production behaviour exactly.
    result = research.run_research(query, "")
    scraper_data = result["scraper_data"]

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / "research_scraper_data.json").write_text(
        json.dumps({"query": query, "scraper_data": scraper_data}, indent=2),
        encoding="utf-8",
    )

    print(f"Query: {query}")
    print(f"scraper_data length: {len(scraper_data)}")
    print(f"Written to: {FIXTURES_DIR / 'research_scraper_data.json'}")


if __name__ == "__main__":
    main()
