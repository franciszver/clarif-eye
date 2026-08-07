"""MANUAL-ONLY end-to-end smoke test for the Gradio UI handler (P4.2 / #14).

Calls the REAL clarif_eye.ui.build_resources() and clarif_eye.ui.
handle_submit() with a real photo: the real OpenRouterClient, the real TTS
provider chain, the real search/fetch seams, and the real compiled graph -
the same object graph app.py wires up, minus ever calling demo.launch().
No Gradio server is started: handle_submit is a plain function and can be
called directly (see ui.py's module docstring, "TESTABLE without launching
a server"). This makes real network calls: at least one OpenRouter request,
and (depending on which path the router picks) a DuckDuckGo search, a page
fetch, and a TTS provider request.

This is NOT part of the pytest suite (tests must stay offline) and must NOT
be run by an automated agent - only by a human, or an orchestrator that has
explicitly decided to spend real API calls. It never launches a server and
never opens a network port.

Usage:
    OPENROUTER_API_KEY=... python scripts/ui_smoke.py path/to/photo.jpg
"""

import sys
from pathlib import Path

from PIL import Image

from clarif_eye.ui import build_resources, handle_submit


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} path/to/photo.jpg", file=sys.stderr)
        raise SystemExit(1)

    image_path = Path(sys.argv[1])
    image = Image.open(image_path)

    resources = build_resources()
    if resources.client is None:
        print(f"error: {resources.client_error}", file=sys.stderr)
        raise SystemExit(1)

    audio_path, text = handle_submit(image, resources)

    print(f"Audio path: {audio_path!r}")
    print(f"Description: {text!r}")

    if not text or not text.strip():
        print("FAIL: handle_submit returned no description text.", file=sys.stderr)
        raise SystemExit(1)

    print("PASS: handle_submit produced a description.")


if __name__ == "__main__":
    main()
