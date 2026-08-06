"""MANUAL-ONLY smoke test for the TTS node's real providers.

This script makes REAL network calls to Microsoft's edge-tts service and/or
Google's gTTS endpoint. It is NOT part of the pytest suite (tests must stay
offline) and must NOT be run by an automated agent - only by a human, or an
orchestrator that has explicitly decided to spend a real network call.

Synthesises a short text string (default below, or the first CLI argument)
and writes an mp3 via clarif_eye.tts.run_tts, then prints the resulting
path, its size in bytes, which provider served it, and its duration if it
can be determined without adding a new dependency (falls back to "unknown"
rather than failing the script).

By default this exercises the FULL provider chain (DEFAULT_PROVIDER_CHAIN -
edge-tts, then gTTS), so a human running it end to end gets the same
failover behaviour a real request would see. `--provider` smoke-tests one
provider in isolation instead - useful for confirming a single provider is
healthy without invoking the whole chain.

Usage:
    python scripts/tts_smoke.py
    python scripts/tts_smoke.py "Some other sentence to speak aloud."
    python scripts/tts_smoke.py --provider edge
    python scripts/tts_smoke.py --provider gtts "Some other sentence."
"""

import argparse
import sys
import wave
from pathlib import Path

from clarif_eye.tts import EdgeTtsProvider, GttsProvider, get_last_tts_result, run_tts

DEFAULT_TEXT = "The image shows a coffee cup on a kitchen counter."

# Name -> provider factory, for --provider. Kept as a plain dict here (not
# reusing tts.DEFAULT_PROVIDER_CHAIN) so this script can smoke-test one
# provider by name without depending on the chain's internal ordering.
_PROVIDER_FACTORIES = {"edge": EdgeTtsProvider, "gtts": GttsProvider}


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT)
    parser.add_argument(
        "--provider",
        choices=sorted(_PROVIDER_FACTORIES),
        default=None,
        help="Smoke-test a single provider instead of the full default chain.",
    )
    args = parser.parse_args()

    if args.provider is not None:
        result = run_tts(args.text, provider=_PROVIDER_FACTORIES[args.provider]())
    else:
        result = run_tts(args.text)  # exercises the full DEFAULT_PROVIDER_CHAIN

    path = result["audio_file_path"]
    last = get_last_tts_result()

    if not path:
        print("TTS failed: audio_file_path is empty.", file=sys.stderr)
        if last is not None:
            for attempt in last.attempts:
                print(f"  {attempt.provider}: {attempt.outcome} - {attempt.detail}", file=sys.stderr)
        raise SystemExit(1)

    size_bytes = Path(path).stat().st_size
    duration = _mp3_duration_seconds(path)

    print(f"Path: {path}")
    print(f"Provider: {last.provider if last is not None else 'unknown'}")
    print(f"Size: {size_bytes} bytes")
    print(f"Duration: {duration if duration is not None else 'unknown'}")


if __name__ == "__main__":
    main()
