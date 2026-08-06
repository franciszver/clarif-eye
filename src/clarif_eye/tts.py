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
that raises TtsError on failure.

PROVIDER CHAIN (issue #12 / P3.2)
----------------------------------
Owner decision D3: edge-tts is an UNOFFICIAL, reverse-engineered Microsoft
endpoint that has broken/rate-limited from cloud IPs before, and Hugging
Face Spaces is exactly such an IP. run_tts therefore tries a CHAIN of
providers in order and uses the first one that produces verified audio,
so a single upstream break does not take down all audio output.

Second provider chosen: GttsProvider, backed by gTTS (Google Translate's
TTS endpoint) - a different vendor's endpoint than edge-tts's, so it does
not share edge-tts's failure mode. Also pure-python and small, like
edge-tts. pyttsx3 was considered and rejected: it drives an OS-level
speech engine (e.g. espeak on Linux) that a Hugging Face Space container
is not guaranteed to have installed, which is itself a single point of
failure this issue exists to remove.

run_tts accepts either the old single-`provider` seam (kept for existing
callers/tests) or a `providers=` sequence; when neither is given it uses
DEFAULT_PROVIDER_CHAIN (EdgeTtsProvider, then GttsProvider). Each
provider's output goes through the SAME audio verification
(_looks_like_audio) - a fallback provider that writes a zero-byte file is
treated as a failure and the chain continues, exactly like the first
provider would be.

When every provider in the chain fails, run_tts still returns
{"audio_file_path": ""} (never raises) so the graph reaches END and the
caller falls back to text - but the state schema is exactly 7 keys (see
state.py) and gains no 8th key to carry *why*. Instead, run_tts records a
structured TtsResult (this call's per-provider attempts) at module level,
readable via get_last_tts_result(), and a caller distinguishes "the whole
chain was tried and failed" from "there was no text to speak in the first
place" via the public predicate is_chain_exhausted() - never by matching
an English message, the same vision.is_degraded_scene pattern.

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
from dataclasses import dataclass
from pathlib import Path

# One sensible default voice, not a selection framework (CLAUDE.md
# Simplicity First / issue scope: voice selection is explicitly out of
# scope for this issue).
DEFAULT_VOICE = "en-US-AriaNeural"

# gTTS language code for the fallback provider - same "one sensible
# default, not a selection framework" reasoning as DEFAULT_VOICE above.
DEFAULT_LANG = "en"

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


class GttsProvider:
    """Fallback provider backed by gTTS (Google Translate's TTS endpoint).

    INDEPENDENT of EdgeTtsProvider: gTTS talks to a different vendor's
    endpoint entirely, so an outage or rate-limit on Microsoft's edge-tts
    endpoint (see module docstring's "PROVIDER CHAIN" / owner decision D3)
    does not affect this provider, and vice versa. Like edge-tts, gTTS is
    itself an unofficial/reverse-engineered client of a public web
    endpoint - not a fully independent failure mode from "some unofficial
    TTS endpoint breaks" - but it IS independent of edge-tts specifically,
    which is what this issue exists to protect against. `gtts` is imported
    lazily, inside synthesize(), matching EdgeTtsProvider's convention.
    """

    def __init__(self, lang=DEFAULT_LANG):
        self.lang = lang

    def synthesize(self, text, out_path):
        """Synthesise `text` to the mp3 file at `out_path`, or raise TtsError."""
        try:
            from gtts import gTTS
        except ImportError as exc:
            raise TtsError(f"gTTS is not installed: {exc}") from exc
        try:
            gTTS(text=text, lang=self.lang).save(str(out_path))
        except TtsError:
            raise
        except Exception as exc:
            raise TtsError(f"gTTS synthesis failed: {exc}") from exc


# Default provider chain, tried in order - a module constant, not a
# registry/plugin system (CLAUDE.md Simplicity First). Each entry is a
# zero-arg factory (the provider classes themselves), constructed lazily by
# _default_providers() so importing this module never requires either
# provider's real dependency to be installed.
DEFAULT_PROVIDER_CHAIN = (EdgeTtsProvider, GttsProvider)


# --- Output directory: bounded, per-process, lazily created -----------------

_default_out_dir_path = None


def _default_out_dir():
    """Lazily create (once per process) a temp dir to hold this process's mp3s."""
    global _default_out_dir_path
    if _default_out_dir_path is None:
        _default_out_dir_path = Path(tempfile.mkdtemp(prefix="clarif_eye_tts_"))
    return _default_out_dir_path


def _default_providers():
    """Factory for the real provider chain. Called lazily (never at import time)."""
    return [factory() for factory in DEFAULT_PROVIDER_CHAIN]


def _provider_name(provider):
    """Identify a provider for attempts/observability: its `name` if it has
    one (tests use this to tell two fakes apart), else its class name."""
    return getattr(provider, "name", None) or type(provider).__name__


# --- Chain observability: structured, not English prose ---------------------
#
# Named outcomes, in the same spirit as vision.py's DEGRADED_* constants and
# client.py's Attempt.category - a caller inspects these fields, never
# substring-matches a message.

OUTCOME_SUCCESS = "success"
OUTCOME_ERROR = "error"
OUTCOME_NO_FILE = "no_file"
OUTCOME_INVALID_AUDIO = "invalid_audio"


@dataclass(frozen=True)
class ProviderAttempt:
    """One provider's outcome within a single run_tts() chain attempt."""

    provider: str
    outcome: str
    detail: str


@dataclass(frozen=True)
class TtsResult:
    """The full outcome of one run_tts() call: which provider (if any) won,
    and every provider tried along the way."""

    audio_file_path: str
    attempts: tuple
    provider: str | None


# Module-level record of the most recent run_tts() call - NOT part of the
# 7-key state schema (state.py); see "PROVIDER CHAIN" in the module
# docstring for why this lives here instead. run_tts is called at most once
# per graph invocation, synchronously, so there is no concurrent-request
# contention for this single-process model (same assumption FILE LIFECYCLE
# above already relies on).
_last_result = None


def get_last_tts_result():
    """Return the TtsResult of the most recently completed run_tts() call,
    or None if run_tts has never been called in this process."""
    return _last_result


def is_chain_exhausted(result=None):
    """True if `result` (defaults to get_last_tts_result()) reached
    audio_file_path == "" because every provider in the chain was tried
    and failed - as opposed to there being no text to speak at all (blank
    final_output never calls a provider, so attempts is empty and this
    returns False). Structural check on the recorded attempts, following
    the vision.is_degraded_scene pattern - never matches an English
    message.
    """
    if result is None:
        result = _last_result
    if result is None:
        return False
    return result.audio_file_path == "" and len(result.attempts) > 0


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


def run_tts(final_output, provider=None, providers=None, out_dir=None):
    """Synthesise `final_output` to an mp3 file; return a tts_node state update.

    Always returns {"audio_file_path": <str>} - "" on any failure (blank
    final_output, or every provider in the chain raising / producing no
    file / producing an empty or non-audio file / raising an unexpected
    exception) - never raises, except KeyboardInterrupt/SystemExit
    (BaseException, not Exception), same contract as every other node in
    this pipeline.

    Providers are tried IN ORDER; the first to produce verified audio
    (see _looks_like_audio) wins and the rest are never tried. `providers`
    (a sequence) is the new chain seam; `provider` (a single object) is
    kept for the old single-provider seam existing callers/tests use and
    is equivalent to providers=[provider]. Passing both is not supported -
    `providers` wins if both are given. When neither is given, the real
    DEFAULT_PROVIDER_CHAIN is used. `out_dir` is injectable the same as
    before (tests pass tmp_path so no test ever touches the network or a
    fixed path); when omitted, a bounded per-process temp directory (see
    _default_out_dir) is used.

    Every provider's outcome is recorded as a ProviderAttempt and the full
    TtsResult for this call is stashed via get_last_tts_result() - see
    "PROVIDER CHAIN" in the module docstring for why that's a module-level
    accessor rather than an 8th state key.
    """
    text = (final_output or "").strip()
    if not text:
        _last_result_set(TtsResult("", (), None))
        return {"audio_file_path": ""}

    if providers is not None:
        chain = list(providers)
    elif provider is not None:
        chain = [provider]
    else:
        chain = _default_providers()

    out_dir = Path(out_dir) if out_dir is not None else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    attempts = []
    for p in chain:
        name = _provider_name(p)
        out_path = out_dir / f"{uuid.uuid4().hex}.mp3"
        try:
            p.synthesize(text, str(out_path))
        except TtsError as exc:
            attempts.append(ProviderAttempt(name, OUTCOME_ERROR, str(exc)))
            continue
        except Exception as exc:
            # Contract (module docstring): no raw exception may escape into
            # the graph, including a provider that raises something other
            # than TtsError - one bad provider must not abort the chain.
            # Never swallows KeyboardInterrupt/SystemExit, which derive
            # from BaseException, not Exception - same pattern as
            # vision.py/synth.py/research.py.
            attempts.append(ProviderAttempt(name, OUTCOME_ERROR, f"unexpected error: {exc!r}"))
            continue

        if not out_path.exists():
            attempts.append(ProviderAttempt(name, OUTCOME_NO_FILE, "provider did not write a file"))
            continue
        try:
            data = out_path.read_bytes()
        except OSError as exc:
            attempts.append(ProviderAttempt(name, OUTCOME_NO_FILE, f"could not read output file: {exc}"))
            continue
        if not _looks_like_audio(data):
            # A zero-byte or non-audio file that "succeeded" would be
            # silence to the user - never report that as success. Clean it
            # up so it doesn't sit around counting against the file bound
            # for nothing, then move on to the next provider in the chain.
            try:
                out_path.unlink()
            except OSError:
                pass
            attempts.append(
                ProviderAttempt(name, OUTCOME_INVALID_AUDIO, "output file was empty or not recognisable audio")
            )
            continue

        _prune_old_files(out_dir)
        attempts.append(ProviderAttempt(name, OUTCOME_SUCCESS, ""))
        _last_result_set(TtsResult(str(out_path), tuple(attempts), name))
        return {"audio_file_path": str(out_path)}

    _last_result_set(TtsResult("", tuple(attempts), None))
    return {"audio_file_path": ""}


def _last_result_set(result):
    global _last_result
    _last_result = result
