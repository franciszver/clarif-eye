"""MANUAL-ONLY fixture recorder for the deep-analysis node.

Takes the already-recorded vision fixture (tests/fixtures/
vision_reply_parsed.json - see scripts/record_vision_fixture.py) as input,
runs the REAL analysis node (clarif_eye.analysis.run_analysis) against the
REAL OpenRouter "brain" ladder using OPENROUTER_API_KEY from the
environment, and writes both the raw model reply and the sanitised final
output to tests/fixtures/.

This is NOT part of the pytest suite (tests must stay offline) and must
NOT be run by an automated agent - only by a human, or an orchestrator
that has explicitly decided to spend a real API call. It makes exactly one
network request (one "brain" ladder attempt sequence). scraper_data is
passed as "" (no research module output is recorded here) - see
run_analysis's contract for why "" is treated as "no external context
available", not a sentinel.

Usage:
    OPENROUTER_API_KEY=... python scripts/record_analysis_fixture.py

Reads:
    tests/fixtures/vision_reply_parsed.json  - ocr_output/scene_context input

Writes:
    tests/fixtures/analysis_reply_raw.txt    - the raw model reply text
    tests/fixtures/analysis_reply_parsed.json - the final sanitised output
"""

import json
from pathlib import Path

from clarif_eye.client import OpenRouterClient
from clarif_eye import analysis

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def main():
    vision_fixture_path = FIXTURES_DIR / "vision_reply_parsed.json"
    vision_result = json.loads(vision_fixture_path.read_text(encoding="utf-8"))
    ocr_output = vision_result["ocr_output"]
    scene_context = vision_result["scene_context"]
    scraper_data = ""

    # Exactly one network call: use the real analysis prompt/message shape
    # (clarif_eye.analysis._build_messages) so the recorded raw reply
    # matches what the node actually sends, then reuse the same
    # sanitisation logic run_analysis would apply, without triggering a
    # second live request.
    client = OpenRouterClient()
    try:
        raw_result = client.complete(
            "brain", analysis._build_messages(ocr_output, scene_context, scraper_data)
        )
        raw_reply = raw_result.content
        served_by = raw_result.model
    finally:
        client.close()

    final_output = analysis._to_spoken_text(raw_reply)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / "analysis_reply_raw.txt").write_text(raw_reply, encoding="utf-8")
    (FIXTURES_DIR / "analysis_reply_parsed.json").write_text(
        json.dumps({"final_output": final_output}, indent=2), encoding="utf-8"
    )

    print(f"Served by: {served_by}")
    print(f"Raw reply written to: {FIXTURES_DIR / 'analysis_reply_raw.txt'}")
    print(f"Final output written to: {FIXTURES_DIR / 'analysis_reply_parsed.json'}")


if __name__ == "__main__":
    main()
