"""MANUAL-ONLY fixture recorder for the fast-synthesis node.

Takes the already-recorded vision fixture (tests/fixtures/
vision_reply_parsed.json - see scripts/record_vision_fixture.py) as input,
runs the REAL fast-synth node (clarif_eye.synth.run_fast_synth) against the
REAL OpenRouter "eyes" ladder using OPENROUTER_API_KEY from the
environment, and writes both the raw model reply and the sanitised final
output to tests/fixtures/.

This is NOT part of the pytest suite (tests must stay offline) and must
NOT be run by an automated agent - only by a human, or an orchestrator
that has explicitly decided to spend a real API call. It makes exactly one
network request (one "eyes" ladder attempt sequence).

Usage:
    OPENROUTER_API_KEY=... python scripts/record_synth_fixture.py

Reads:
    tests/fixtures/vision_reply_parsed.json - ocr_output/scene_context input

Writes:
    tests/fixtures/synth_reply_raw.txt      - the raw model reply text
    tests/fixtures/synth_reply_parsed.json  - the final sanitised output
"""

import json
from pathlib import Path

from clarif_eye.client import OpenRouterClient
from clarif_eye import synth

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def main():
    vision_fixture_path = FIXTURES_DIR / "vision_reply_parsed.json"
    vision_result = json.loads(vision_fixture_path.read_text(encoding="utf-8"))
    ocr_output = vision_result["ocr_output"]
    scene_context = vision_result["scene_context"]

    # Exactly one network call: use the real synth prompt/message shape
    # (clarif_eye.synth._build_messages) so the recorded raw reply matches
    # what the node actually sends, then reuse the same sanitisation logic
    # run_fast_synth would apply, without triggering a second live request.
    client = OpenRouterClient()
    try:
        raw_result = client.complete(
            "eyes", synth._build_messages(ocr_output, scene_context)
        )
        raw_reply = raw_result.content
        served_by = raw_result.model
    finally:
        client.close()

    final_output = synth._to_spoken_text(raw_reply)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / "synth_reply_raw.txt").write_text(raw_reply, encoding="utf-8")
    (FIXTURES_DIR / "synth_reply_parsed.json").write_text(
        json.dumps({"final_output": final_output}, indent=2), encoding="utf-8"
    )

    print(f"Served by: {served_by}")
    print(f"Raw reply written to: {FIXTURES_DIR / 'synth_reply_raw.txt'}")
    print(f"Final output written to: {FIXTURES_DIR / 'synth_reply_parsed.json'}")


if __name__ == "__main__":
    main()
