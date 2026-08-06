"""Replay recorded vision fixture through the real parser (issue #5 / P1.2).

This test replays a recorded nemotron-3-nano-omni reply through run_vision
to verify that production parsing logic handles real model output without
making a network call. The fixture is byte-for-byte model output recorded
locally during live testing.

No network calls: an injected fake client returns the recorded raw reply,
and the real parsing pipeline processes it deterministically.
"""

import json
from pathlib import Path

import pytest

from clarif_eye.client import CompletionResult
from clarif_eye.vision import run_vision


class FakeFixtureClient:
    """Minimal stand-in for OpenRouterClient that returns recorded fixture content."""

    def __init__(self, content):
        self.content = content
        self.calls = []
        self.closed = False

    def complete(self, role, messages, **params):
        self.calls.append({"role": role, "messages": messages, "params": params})
        return CompletionResult(content=self.content, model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")

    def close(self):
        self.closed = True


# Ground truth for the 1000x700 synthetic utility bill fixture
GROUND_TRUTH_FACTS = [
    "CITY OF RIVERTON",
    "WATER UTILITY STATEMENT",
    "4471-2205-88",  # Account Number
    "01 Jun 2026",  # Billing Period start
    "30 Jun 2026",  # Billing Period end
    "1188 Kestrel Lane",  # Service Address
    "Apt 4B",
    "$41.20",  # Previous Balance
    "$63.75",  # Current Charges
    "$0.00",  # Late Fee
    "$104.95",  # AMOUNT DUE
    "22 JULY 2026",  # PAYMENT DUE BY
    "riverton.gov/water",
]


FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Ground-truth facts are specific to the one recorded fixture we have today
# (the synthetic utility bill, OLD OCR_TEXT:/SCENE: format - see P1.8/#29:
# this fixture is expected to keep passing via the legacy fallback parser,
# not the new sentinel format, because it's real recorded evidence and
# fixtures are not edited). A future fixture recorded with the new
# sentinel-delimited prompt (P1.8/#29) will be discovered automatically
# below and only gets the format-agnostic assertions, not this fact list.
GROUND_TRUTH_FACTS_BY_FIXTURE = {
    "vision_reply_raw.txt": GROUND_TRUTH_FACTS,
}


def _discover_vision_fixture_pairs():
    """Find every (raw, parsed) fixture pair under tests/fixtures/.

    Iterates over whichever `vision_reply_raw*.txt` fixtures exist rather
    than hardcoding one filename, so a new fixture recorded later (e.g.
    with the new sentinel prompt) is picked up without editing this test -
    and if none exist, the test below skips cleanly instead of failing.
    """
    pairs = []
    for raw_path in sorted(FIXTURES_DIR.glob("vision_reply_raw*.txt")):
        parsed_path = FIXTURES_DIR / raw_path.name.replace("_raw", "_parsed").replace(
            ".txt", ".json"
        )
        if parsed_path.exists():
            pairs.append((raw_path, parsed_path))
    return pairs


_FIXTURE_PAIRS = _discover_vision_fixture_pairs()
_PARAMS = _FIXTURE_PAIRS or [pytest.param(None, None, marks=pytest.mark.skip(
    reason="No vision_reply_raw*.txt fixtures found under tests/fixtures/. "
    "This test requires recorded output from live model inference. "
    "Run locally to generate, then commit the fixture files."
))]
_IDS = [raw.name for raw, _parsed in _FIXTURE_PAIRS] or ["no-fixtures"]


@pytest.mark.parametrize("fixture_raw_path,fixture_parsed_path", _PARAMS, ids=_IDS)
def test_replay_fixture_through_real_parser(fixture_raw_path, fixture_parsed_path):
    """Replay each recorded vision reply through run_vision and verify parsing.

    Loads the raw fixture (model's verbatim reply), feeds it through run_vision
    with an injected fake client, and asserts that:
    - All ground-truth facts appear in the parsed ocr_output (only checked for
      fixtures that have a known fact list - see GROUND_TRUTH_FACTS_BY_FIXTURE)
    - scene_context is a real sentence (not empty or a fragment)
    - scene_context does not leak marker strings (OCR_TEXT:, SCENE:)
    - complexity_flag is a bool and matches the recorded JSON
    """
    # Load the recorded raw reply and expected parsed result
    raw_reply = fixture_raw_path.read_text()
    expected_parsed = json.loads(fixture_parsed_path.read_text())

    # Replay through the real parser
    client = FakeFixtureClient(raw_reply)
    result = run_vision("dummy_base64_data", client)

    # Verify all ground-truth facts are in the OCR output (fixture-specific)
    ocr_output = result["ocr_output"]
    for fact in GROUND_TRUTH_FACTS_BY_FIXTURE.get(fixture_raw_path.name, []):
        assert fact in ocr_output, f"Expected fact {fact!r} not found in ocr_output"

    # Verify scene_context is non-trivial (real sentence, not fragment)
    scene_context = result["scene_context"]
    assert isinstance(scene_context, str)
    assert scene_context.strip(), "scene_context must not be empty"
    words = scene_context.split()
    assert len(words) >= 4, f"scene_context must be more than a fragment: {scene_context!r}"

    # Verify marker strings do not leak into scene_context
    assert "OCR_TEXT:" not in scene_context, "OCR_TEXT: marker leaked into scene_context"
    assert "SCENE:" not in scene_context, "SCENE: marker leaked into scene_context"

    # Verify complexity_flag is a bool and matches the recorded JSON
    assert isinstance(result["complexity_flag"], bool)
    assert result["complexity_flag"] == expected_parsed["complexity_flag"], (
        f"complexity_flag mismatch: got {result['complexity_flag']}, "
        f"expected {expected_parsed['complexity_flag']}"
    )

    # Verify that the client was called exactly once
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["role"] == "eyes"
    assert isinstance(call["messages"], list)
