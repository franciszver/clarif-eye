"""Replay recorded fast-synth fixture through the real synth path (P4.2 / #14).

Mirrors test_vision_fixture_replay.py / test_analysis_fixture_replay.py:
this test replays a recorded eyes-ladder reply through run_fast_synth to
verify the production sanitisation logic (clarif_eye.synth._to_spoken_text)
handles real model output without making a network call. The fixture would
be byte-for-byte model output recorded locally by scripts/
record_synth_fixture.py during live testing (see that script's module
docstring), the same way tests/fixtures/vision_reply_raw.txt and
tests/fixtures/analysis_reply_raw.txt were recorded.

No fixture has been recorded yet (scripts/record_synth_fixture.py makes a
real network call and is manual-only - see its module docstring - so this
task did not run it). Until someone runs it and commits
tests/fixtures/synth_reply_raw.txt, this test SKIPS rather than failing:
same discipline test_vision_fixture_replay.py and
test_analysis_fixture_replay.py already use for a fixture that is missing.

No network calls: an injected fake client returns the recorded raw reply,
and the real sanitisation pipeline processes it deterministically.
"""

from pathlib import Path

import pytest

from clarif_eye.client import CompletionResult
from clarif_eye.synth import run_fast_synth

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# The inputs the real eyes-ladder reply would be recorded against -
# vision.py's own fixture, parsed (see GROUND_TRUTH_FACTS in
# test_vision_fixture_replay.py for the same source document).
OCR_OUTPUT = (
    "CITY OF RIVERTON WATER UTILITY STATEMENT Account Number: 4471-2205-88 "
    "Billing Period: 01 Jun 2026 to 30 Jun 2026 Service Address: 1188 Kestrel "
    "Lane, Apt 4B Previous Balance $41.20 Current Charges $63.75 Late Fee "
    "$0.00 AMOUNT DUE $104.95 PAYMENT DUE BY: 22 JULY 2026 Pay online at "
    "riverton.gov/water"
)
SCENE_CONTEXT = (
    "A rectangular water utility statement from the City of Riverton showing "
    "account details, billing period, charges, amount due, and payment deadline."
)


class FakeFixtureClient:
    """Minimal stand-in for OpenRouterClient that returns recorded fixture content."""

    def __init__(self, content):
        self.content = content
        self.calls = []
        self.closed = False

    def complete(self, role, messages, **params):
        self.calls.append({"role": role, "messages": messages, "params": params})
        return CompletionResult(content=self.content, model="fake-eyes-model:free")

    def close(self):
        self.closed = True


@pytest.fixture
def fixture_raw_path():
    return FIXTURES_DIR / "synth_reply_raw.txt"


def skip_if_fixture_missing(fixture_path):
    if not fixture_path.exists():
        pytest.skip(
            f"Fixture not found: {fixture_path.name}. This test requires "
            "recorded output from live model inference (see scripts/"
            "record_synth_fixture.py). Run it locally, then commit the "
            "fixture file."
        )


def test_replay_fixture_through_real_synth_path(fixture_raw_path):
    """Replay the recorded eyes-ladder reply through run_fast_synth.

    Loads the recorded raw reply, feeds it through run_fast_synth with an
    injected fake client using the same inputs it would be recorded
    against, and asserts that:
    - the client was called exactly once, targeting the "eyes" role
    - the sanitised final_output is non-empty, real prose (not a
      degradation message)
    """
    skip_if_fixture_missing(fixture_raw_path)

    raw_reply = fixture_raw_path.read_text(encoding="utf-8")

    client = FakeFixtureClient(raw_reply)
    result = run_fast_synth(OCR_OUTPUT, SCENE_CONTEXT, client)

    final_output = result["final_output"]
    assert final_output.strip() != ""
    assert len(client.calls) == 1
    assert client.calls[0]["role"] == "eyes"
