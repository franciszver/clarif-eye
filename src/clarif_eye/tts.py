"""Text-to-speech node logic (issue #11 / P3.1): turn the spoken script into audio.

Last stage of the pipeline: final_output (already sanitised by
speech.to_spoken_text) becomes an mp3 file, and audio_file_path is set to
its path. This module must NEVER let a raw exception escape into the
graph - final_output is the user's ONLY payload, and losing it to an
unhandled exception would mean total silence, the worst possible failure
for a visually impaired user. Every failure mode (no provider available,
provider raising, provider producing no file, provider producing an EMPTY
file, blank final_output, any unexpected exception) degrades to
audio_file_path = "" rather than raising. Issue #12 owns turning that
into a text-only fallback for the user (surfaced by issue #13's UI); this
module deliberately does NOT invent one here.

PROVIDER SEAM
-------------
No ABC/Protocol - this codebase avoids ABCs (see vision.py/synth.py/
research.py, which all use plain duck-typed injectable objects instead).
A provider is any object exposing `synthesize(text, out_path) -> None`
that raises TtsError on failure. Issue #12 adds a second provider (and a
text-only fallback) as a drop-in implementer of this same seam - run_tts's
shape (provider, out_dir both injectable, one dict key back) is written so
that addition needs no change here.

EDGE-TTS / SYNC BRIDGE
-----------------------
edge-tts's Communicate.save() is async. The graph is entirely synchronous
(LangGraph's StateGraph.invoke), so EdgeTtsProvider.synthesize wraps the
async call with asyncio.run(). asyncio.run() creates a brand new event
loop, runs the coroutine to completion, and tears the loop down again -
safe here because run_tts is called at most once per graph invocation,
synchronously, and never from inside an already-running event loop. It
would raise RuntimeError if called from async code (e.g. from inside an
async web framework's request handler); this module only ever targets the
synchronous LangGraph graph built in graph.py, so that never applies here.
`edge_tts` itself is imported lazily, inside synthesize() - never at
module import time - both to match this codebase's existing lazy-import
convention (see research._default_searcher) and so importing this module
does not require edge-tts to be installed at all when only the injectable
seam is exercised (as every test does).

FILE LIFECYCLE
----------------
mp3 files are written to a per-process temp directory (tempfile), never
into the repo - `.gitignore` already ignores *.mp3, but this module does
not rely on that. Every call gets a fresh uuid4-based filename - a fixed
name would let concurrent requests on a shared Space collide, with one
user's audio overwriting another's mid-read.

A Space process runs for days with no restart between requests, so an
unbounded number of mp3s would eventually fill the disk. Bound chosen:
after each successful write, the output directory is pruned down to the
MAX_KEPT_FILES most-recently-modified *.mp3 files, oldest first. This is
a simple, local, dependency-free bound - no scheduler or background
thread - that fits a single long-running process (the expected shape of
a Hugging Face Space free-tier Space). It deliberately never deletes the
file a call is about to return, and only prunes down to a count, not a
TTL, so a slow listener (issue #13's UI streaming the file back) is never
at risk of the file disappearing out from under a still-fresh request.

Every produced file is verified non-empty AND checked for a real mp3
signature (an ID3 tag or an MPEG frame sync) before being reported as
success - a zero-byte or garbage "audio" file that "succeeded" would be
silence to a user who cannot tell the difference.
"""

import asyncio
import tempfile
import uuid
from pathlib import Path

# One sensible default voice, not a selection framework (CLAUDE.md
# Simplicity First / issue scope: voice selection is explicitly out of
# scope for this issue).
DEFAULT_VOICE = "en-US-AriaNeural"

# See "FILE LIFECYCLE" above.
MAX_KEPT_FILES = 20


class TtsError(Exception):
    """Raised by a provider's synthesize() when audio could not be produced."""


class EdgeTtsProvider:
    """Real provider backed by Microsoft's edge-tts service (network I/O)."""

    def __init__(self, voice=DEFAULT_VOICE):
        self.voice = voice

    def synthesize(self, text, out_path):
        """Synthesise `text` to the mp3 file at `out_path`, or raise TtsError.

        Bridges edge-tts's async API to this seam's synchronous contract -
        see the module docstring's "EDGE-TTS / SYNC BRIDGE" section.
        """
        try:
            import edge_tts
        except ImportError as exc:
            raise TtsError(f"edge-tts is not installed: {exc}") from exc
        try:
            asyncio.run(self._synthesize_async(edge_tts, text, out_path))
        except TtsError:
            raise
        except Exception as exc:
            raise TtsError(f"edge-tts synthesis failed: {exc}") from exc

    async def _synthesize_async(self, edge_tts, text, out_path):
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(out_path))


# --- Output directory: bounded, per-process, lazily created -----------------

_default_out_dir_path = None


def _default_out_dir():
    """Lazily create (once per process) a temp dir to hold this process's mp3s."""
    global _default_out_dir_path
    if _default_out_dir_path is None:
        _default_out_dir_path = Path(tempfile.mkdtemp(prefix="clarif_eye_tts_"))
    return _default_out_dir_path


def _default_provider():
    """Factory for the real provider. Called lazily (never at import time)."""
    return EdgeTtsProvider()


def _looks_like_audio(data):
    """True if `data` starts with a real mp3 signature (ID3 tag or MPEG frame sync).

    Not a full audio validator - just enough to catch the two realistic
    failure shapes (a zero-byte file, or a provider that wrote something
    that isn't actually mp3 audio) rather than reporting silence as
    success.
    """
    if not data:
        return False
    if data[:3] == b"ID3":
        return True
    return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0


def _prune_old_files(out_dir):
    """Delete the oldest *.mp3 files in `out_dir` beyond MAX_KEPT_FILES.

    Sorted by mtime so the most recently written files (including the one
    this call just produced) are always kept - see "FILE LIFECYCLE" above.
    A file that fails to delete (e.g. removed by something else already)
    is skipped rather than raising; pruning is best-effort housekeeping,
    not something a request's success should depend on.
    """
    files = sorted(
        (p for p in out_dir.iterdir() if p.is_file() and p.suffix == ".mp3"),
        key=lambda p: p.stat().st_mtime,
    )
    excess = len(files) - MAX_KEPT_FILES
    if excess <= 0:
        return
    for p in files[:excess]:
        try:
            p.unlink()
        except OSError:
            pass


def run_tts(final_output, provider=None, out_dir=None):
    """Synthesise `final_output` to an mp3 file; return a tts_node state update.

    Always returns {"audio_file_path": <str>} - "" on any failure (blank
    final_output, the provider raising, the provider producing no file or
    an empty/non-audio file, or any unexpected exception) - never raises,
    except KeyboardInterrupt/SystemExit (BaseException, not Exception),
    same contract as every other node in this pipeline.

    `provider`/`out_dir` are injectable (tests pass a fake provider and
    tmp_path so no test ever touches the network or a fixed path); when
    omitted, a real EdgeTtsProvider and a bounded per-process temp
    directory (see _default_out_dir) are used.
    """
    text = (final_output or "").strip()
    if not text:
        return {"audio_file_path": ""}

    if provider is None:
        provider = _default_provider()
    out_dir = Path(out_dir) if out_dir is not None else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{uuid.uuid4().hex}.mp3"
    try:
        provider.synthesize(text, str(out_path))
    except TtsError:
        return {"audio_file_path": ""}
    except Exception:
        # Contract (module docstring): no raw exception may escape into
        # the graph, including a provider that raises something other
        # than TtsError. Never swallows KeyboardInterrupt/SystemExit,
        # which derive from BaseException, not Exception - same pattern
        # as vision.py/synth.py/research.py.
        return {"audio_file_path": ""}

    if not out_path.exists():
        return {"audio_file_path": ""}
    try:
        data = out_path.read_bytes()
    except OSError:
        return {"audio_file_path": ""}
    if not _looks_like_audio(data):
        # A zero-byte or non-audio file that "succeeded" would be silence
        # to the user - never report that as success. Clean it up so it
        # doesn't sit around counting against the file bound for nothing.
        try:
            out_path.unlink()
        except OSError:
            pass
        return {"audio_file_path": ""}

    _prune_old_files(out_dir)
    return {"audio_file_path": str(out_path)}
