"""Replay recorded analysis fixture through the real analysis path (FIX 2).

Mirrors test_vision_fixture_replay.py: this test replays a recorded brain-
ladder reply through run_analysis to verify the production parsing and
FIX-1 number-verification logic handle real model output without making a
network call. The fixture is byte-for-byte model output recorded locally
during live testing (see analysis.py's module docstring and vision.py's
fixture for the same 1000x700 synthetic utility bill this was recorded
against).

No network calls: an injected fake client returns the recorded raw reply,
and the real run_analysis pipeline processes it deterministically.

NOTE (FIX 3): after the bare-domain URL fix, to_spoken_text now turns
"riverton.gov/water" into "a web link", so the tracked
analysis_reply_parsed.json (which records what to_spoken_text produced at
RECORD time, before that fix existed) no longer byte-matches the CURRENT
pipeline's output. That is expected - the parsed fixture is a historical
record, not a live contract - so this test asserts ground-truth values
survive and does not byte-compare against analysis_reply_parsed.json.
"""

from pathlib import Path

import pytest

from clarif_eye.client import CompletionResult
from clarif_eye.analysis import run_analysis, _numbers_verified

# Ground truth for the 1000x700 synthetic utility bill fixture (same
# document vision_fixture_replay.py verifies OCR against).
GROUND_TRUTH_FACTS = [
    "4471-2205-88",  # Account Number
    "$104.95",  # AMOUNT DUE
    "$41.20",  # Previous Balance
    "$63.75",  # Current Charges
    "22 JULY 2026",  # PAYMENT DUE BY
]

# The inputs the real brain reply was recorded against - vision.py's own
# fixture, parsed (see vision_reply_parsed.json / GROUND_TRUTH_FACTS in
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
        return CompletionResult(content=self.content, model="fake-brain-model:free")

    def close(self):
        self.closed = True


@pytest.fixture
def fixture_raw_path():
    return Path(__file__).parent / "fixtures" / "analysis_reply_raw.txt"


def skip_if_fixture_missing(fixture_path):
    if not fixture_path.exists():
        pytest.skip(
            f"Fixture not found: {fixture_path.name}. "
            "This test requires recorded output from live model inference. "
            "Run locally to generate, then commit the fixture files."
        )


def test_replay_fixture_through_real_analysis_path(fixture_raw_path):
    """Replay the recorded brain reply through run_analysis and verify it survives.

    Loads the recorded raw reply (the model's verbatim reply), feeds it
    through run_analysis with an injected fake client using the same inputs
    it was recorded against, and asserts that:
    - the ground-truth ID/amounts/date survive verbatim in final_output
    - the output passes FIX 1's number-verification check (it must - this
      is real, faithful model output, not an invented number)
    - the client was called exactly once, targeting the "brain" role
    """
    skip_if_fixture_missing(fixture_raw_path)

    raw_reply = fixture_raw_path.read_text()

    client = FakeFixtureClient(raw_reply)
    result = run_analysis(OCR_OUTPUT, SCENE_CONTEXT, "", client)

    final_output = result["final_output"]
    for fact in GROUND_TRUTH_FACTS:
        assert fact in final_output, f"Expected fact {fact!r} not found in final_output"

    # It must have actually passed FIX 1, not degraded into the fallback
    # message that happens to also contain no invented numbers.
    assert _numbers_verified(final_output, OCR_OUTPUT, SCENE_CONTEXT, "")
    assert "could not be verified" not in final_output

    assert len(client.calls) == 1
    assert client.calls[0]["role"] == "brain"


def test_corrupting_the_fixture_fails_verification(fixture_raw_path):
    """Mutation proof: an injected fabricated number must fail FIX 1's check.

    Reads the real fixture, corrupts one genuine amount into a value that
    does not appear anywhere in the inputs, and confirms run_analysis
    degrades instead of speaking the corrupted number as fact. The file on
    disk is never touched by this test.
    """
    skip_if_fixture_missing(fixture_raw_path)

    raw_reply = fixture_raw_path.read_text()
    assert "$104.95" in raw_reply, "fixture no longer contains the expected ground truth"
    corrupted_reply = raw_reply.replace("$104.95", "$999.99")

    client = FakeFixtureClient(corrupted_reply)
    result = run_analysis(OCR_OUTPUT, SCENE_CONTEXT, "", client)

    final_output = result["final_output"]
    assert "$999.99" not in final_output
    assert "could not be verified" in final_output
