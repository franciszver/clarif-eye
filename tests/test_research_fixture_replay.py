"""Check recorded research fixture evidence (P4.2 / #14).

Unlike vision/analysis/synth, scripts/record_research_fixture.py records
run_research()'s OUTPUT (the already-extracted scraper_data), not a raw
page to feed back through the extraction/sanitisation code - run_research
does the real search, the real fetch, and the real HTML-to-text extraction
in one call, and there is no raw-HTML fixture to "replay" through that
extraction step separately. So this is a lighter check than the vision/
analysis/synth replay tests: it inspects the recorded evidence's SHAPE
(non-empty, no leaked markup, within the production size cap), not a
re-run of production parsing logic against it. See docs/SCENARIOS.md for
why this is honestly weaker evidence than the other three nodes' fixture
replays.

No fixture has been recorded yet (scripts/record_research_fixture.py makes
a real network call and is manual-only - see its module docstring - so
this task did not run it). Until someone runs it and commits tests/
fixtures/research_scraper_data.json, this test SKIPS rather than failing -
same discipline the vision/analysis fixture-replay tests use for a missing
fixture.

No network calls: this test only reads a JSON file already on disk.
"""

import json
from pathlib import Path

import pytest

from clarif_eye.research import _EXTRACTED_TEXT_MAX_CHARS

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "research_scraper_data.json"


def test_recorded_scraper_data_is_well_formed_evidence():
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Fixture not found: {FIXTURE_PATH.name}. This test requires "
            "recorded output from a live search + page fetch (see scripts/"
            "record_research_fixture.py). Run it locally, then commit the "
            "fixture file."
        )

    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert "query" in data and data["query"].strip() != ""
    scraper_data = data["scraper_data"]
    assert isinstance(scraper_data, str)
    assert scraper_data.strip() != "", "recorded scraper_data is empty - the fixture proves nothing"
    # run_research's own extraction cap - real evidence must respect it.
    assert len(scraper_data) <= _EXTRACTED_TEXT_MAX_CHARS + len(" [truncated]")
    # Script/style content must never survive extraction into scraper_data.
    assert "<script" not in scraper_data.lower()
    assert "<style" not in scraper_data.lower()
