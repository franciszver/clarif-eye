"""MANUAL-ONLY smoke test for the TTS node's real edge-tts provider.

This script makes REAL network calls to Microsoft's edge-tts service. It is
NOT part of the pytest suite (tests must stay offline) and must NOT be run
by an automated agent - only by a human, or an orchestrator that has
explicitly decided to spend a real network call.

Synthesises a short text string (default below, or the first CLI argument)
with the real EdgeTtsProvider, writes an mp3 via clarif_eye.tts.run_tts, and
prints the resulting path, its size in bytes, and its duration if it can be
determined without adding a new dependency (falls back to "unknown" rather
than failing the script).

Usage:
    python scripts/tts_smoke.py
    python scripts/tts_smoke.py "Some other sentence to speak aloud."
"""

import sys
import wave
from pathlib import Path

from clarif_eye.tts import run_tts

DEFAULT_TEXT = "The image shows a coffee cup on a kitchen counter."


def _mp3_duration_seconds(path):
    """Best-effort mp3 duration, or None if it can't be determined.

    No mp3-decoding dependency is added for this (out of scope for this
    issue - see tts.py's module docstring on avoiding new dependencies).
    `wave` only understands WAV, so this always returns None for a real
    mp3 file; it's here so a future WAV-based provider gets a duration for
    free, and so this function's failure mode is visibly a "don't know",
    never a crash.
    """
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate) if rate else None
    except (wave.Error, EOFError, OSError):
        return None


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT

    result = run_tts(text)
    path = result["audio_file_path"]

    if not path:
        print("TTS failed: audio_file_path is empty.", file=sys.stderr)
        raise SystemExit(1)

    size_bytes = Path(path).stat().st_size
    duration = _mp3_duration_seconds(path)

    print(f"Path: {path}")
    print(f"Size: {size_bytes} bytes")
    print(f"Duration: {duration if duration is not None else 'unknown'}")


if __name__ == "__main__":
    main()
