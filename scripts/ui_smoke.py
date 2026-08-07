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

FOLLOW-UP ROUND TRIP (issue #82 / P9.3): pass a question as a second
argument and this runs the photo AND then that question, on ONE thread_id,
against the same live resources - the real-stack pairing for
tests/test_followup.py, whose fakes assert that a follow-up costs one brain
call and no vision call. Nothing here can prove the call count against
production (only the recorded fake can); what this proves is the part the
fake cannot - that a real checkpointed thread really does still hold the
photo's ocr_output/scene_context by the time the question arrives, and that
a real model answers from it.

Usage:
    OPENROUTER_API_KEY=... python scripts/ui_smoke.py path/to/photo.jpg
    OPENROUTER_API_KEY=... python scripts/ui_smoke.py path/to/photo.jpg "what is the expiry date?"
"""

import sys
import uuid
from pathlib import Path

from PIL import Image

from clarif_eye.ui import build_resources, handle_ask_staged, handle_submit


def main():
    if len(sys.argv) not in (2, 3):
        print(
            f'Usage: python {sys.argv[0]} path/to/photo.jpg ["a follow-up question"]',
            file=sys.stderr,
        )
        raise SystemExit(1)

    image_path = Path(sys.argv[1])
    question = sys.argv[2] if len(sys.argv) == 3 else None
    image = Image.open(image_path)

    resources = build_resources()
    if resources.client is None:
        print(f"error: {resources.client_error}", file=sys.stderr)
        raise SystemExit(1)

    # One thread_id for BOTH calls - that is what makes the follow-up able
    # to read the photo run's stored state (see clarif_eye.ui's
    # thread_configurable / build_interface's per-session gr.State).
    thread_id = str(uuid.uuid4())

    audio_path, text = handle_submit(image, resources, thread_id=thread_id)

    print(f"Audio path: {audio_path!r}")
    print(f"Description: {text!r}")

    if not text or not text.strip():
        print("FAIL: handle_submit returned no description text.", file=sys.stderr)
        raise SystemExit(1)

    print("PASS: handle_submit produced a description.")

    if question is None:
        return

    # Drain the staged generator exactly as Gradio does; the last yield
    # carries the final (status, audio, text).
    updates = list(handle_ask_staged(question, resources, thread_id=thread_id))
    _status, answer_audio, answer_text = updates[-1]

    print(f"Question: {question!r}")
    print(f"Answer audio path: {answer_audio!r}")
    print(f"Answer: {answer_text!r}")

    if not answer_text or not answer_text.strip():
        print("FAIL: handle_ask_staged returned no answer text.", file=sys.stderr)
        raise SystemExit(1)

    print("PASS: a follow-up question on the same thread produced an answer.")


if __name__ == "__main__":
    main()
