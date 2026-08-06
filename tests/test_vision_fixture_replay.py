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


@pytest.fixture
def fixture_raw_path():
    """Resolve fixture file path relative to this test file."""
    return Path(__file__).parent / "fixtures" / "vision_reply_raw.txt"


@pytest.fixture
def fixture_parsed_path():
    """Resolve fixture file path relative to this test file."""
    return Path(__file__).parent / "fixtures" / "vision_reply_parsed.json"


def skip_if_fixture_missing(fixture_path):
    """Skip the test with a clear message if the fixture file is absent."""
    if not fixture_path.exists():
        pytest.skip(
            f"Fixture not found: {fixture_path.name}. "
            "This test requires recorded output from live model inference. "
            "Run locally to generate, then commit the fixture files."
        )


def test_replay_fixture_through_real_parser(fixture_raw_path, fixture_parsed_path):
    """Replay the recorded nemotron reply through run_vision and verify parsing.

    Loads the raw fixture (model's verbatim reply), feeds it through run_vision
    with an injected fake client, and asserts that:
    - All ground-truth facts appear in the parsed ocr_output
    - scene_context is a real sentence (not empty or a fragment)
    - scene_context does not leak marker strings (OCR_TEXT:, SCENE:)
    - complexity_flag is a bool and matches the recorded JSON
    """
    skip_if_fixture_missing(fixture_raw_path)
    skip_if_fixture_missing(fixture_parsed_path)

    # Load the recorded raw reply and expected parsed result
    raw_reply = fixture_raw_path.read_text()
    expected_parsed = json.loads(fixture_parsed_path.read_text())

    # Replay through the real parser
    client = FakeFixtureClient(raw_reply)
    result = run_vision("dummy_base64_data", client)

    # Verify all ground-truth facts are in the OCR output
    ocr_output = result["ocr_output"]
    for fact in GROUND_TRUTH_FACTS:
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
