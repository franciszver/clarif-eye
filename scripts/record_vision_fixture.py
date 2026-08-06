"""MANUAL-ONLY fixture recorder for the vision node.

Takes a real image file, base64-encodes it, and runs the REAL vision node
(clarif_eye.vision.run_vision) against the REAL OpenRouter "eyes" ladder
using OPENROUTER_API_KEY from the environment. Writes both the raw model
reply and the parsed result to tests/fixtures/, for later use as a
recorded fixture in offline tests.

This is NOT part of the pytest suite (tests must stay offline) and must
NOT be run by an automated agent - only by a human, or an orchestrator
that has explicitly decided to spend a real API call. It makes exactly one
network request (one "eyes" ladder attempt sequence).

Usage:
    OPENROUTER_API_KEY=... python scripts/record_vision_fixture.py path/to/photo.jpg

Writes:
    tests/fixtures/vision_reply_raw.txt    - the raw model reply text
    tests/fixtures/vision_reply_parsed.json - the parsed run_vision() result
"""

import base64
import json
import sys
from pathlib import Path

from clarif_eye.client import OpenRouterClient
from clarif_eye import vision

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} path/to/image.jpg", file=sys.stderr)
        raise SystemExit(1)

    image_path = Path(sys.argv[1])
    image_bytes = image_path.read_bytes()
    image_data = base64.b64encode(image_bytes).decode("ascii")

    # Exactly one network call: use the real vision prompt/message shape
    # (clarif_eye.vision._build_messages) so the recorded raw reply matches
    # what the node actually sends, then reuse the same parsing logic
    # run_vision would apply, without triggering a second live request.
    client = OpenRouterClient()
    try:
        raw_result = client.complete("eyes", vision._build_messages(image_data))
        raw_reply = raw_result.content
        served_by = raw_result.model
    finally:
        client.close()

    parsed_fields = vision._parse_reply(raw_reply)
    if parsed_fields is None:
        parsed = {"ocr_output": "", "scene_context": "", "complexity_flag": False}
    else:
        ocr_output, scene_context = parsed_fields
        parsed = {
            "ocr_output": ocr_output,
            "scene_context": scene_context,
            "complexity_flag": vision._complexity_flag(ocr_output),
        }

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / "vision_reply_raw.txt").write_text(raw_reply, encoding="utf-8")
    (FIXTURES_DIR / "vision_reply_parsed.json").write_text(
        json.dumps(parsed, indent=2), encoding="utf-8"
    )

    print(f"Served by: {served_by}")
    print(f"Raw reply written to: {FIXTURES_DIR / 'vision_reply_raw.txt'}")
    print(f"Parsed result written to: {FIXTURES_DIR / 'vision_reply_parsed.json'}")


if __name__ == "__main__":
    main()
