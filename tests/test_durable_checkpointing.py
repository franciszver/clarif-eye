"""Tests for durable checkpointing via SqliteSaver (issue #109 / P10.1).

RED FIRST: at the time this file was first committed, clarif_eye.graph had
no `make_checkpointer`, so every test here failed on the import line.

Drives the compiled graph through a paused (interrupt) state the same way
tests/test_ask_before_speaking.py does - reusing that file's own
RecordingClient/FakeSearcher/_FakeTtsProvider fakes and its dense-document
draft that fails number verification, rather than a second copy of them -
and reuses tests/test_checkpointing.py's `_invoke` shape for a plain
completed turn. See both files' own docstrings for why those fakes exist
and what they avoid touching (no model or network call).
"""

import sqlite3

from clarif_eye.client import CompletionResult
from clarif_eye.graph import DESCRIBE_ONE_NODE, RESUME_CONTINUE, build_graph, make_checkpointer
from clarif_eye.state import make_initial_state
from clarif_eye.ui import _trim_thread_to_latest_checkpoint

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from tests.test_ask_before_speaking import (
    INVENTED_DRAFT,
    FakeSearcher,
    RecordingClient,
    _FakeTtsProvider,
)


def _run_config(thread_id):
    return {
        "configurable": {
            "thread_id": thread_id,
            "client": RecordingClient(INVENTED_DRAFT),
            "searcher": FakeSearcher(),
            "tts_provider": _FakeTtsProvider(),
        }
    }


# --- (a) Restart survival: a brand-new SqliteSaver on the same file resumes
# a pause a DIFFERENT SqliteSaver instance left there. ------------------------


def test_pause_survives_a_brand_new_sqlitesaver_on_the_same_file(tmp_path):
    db_path = str(tmp_path / "checkpoints.sqlite")
    thread_id = "restart-thread"
    config = _run_config(thread_id)

    # First "process": pause a run on a SqliteSaver over db_path. The
    # connection is closed via try/finally so a failing assertion above it
    # can't leak a Windows file lock into pytest's tmp_path cleanup.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        first_graph = build_graph(checkpointer=SqliteSaver(conn))

        chunks = list(
            first_graph.stream(
                make_initial_state("base64photo"), config=config, stream_mode="updates"
            )
        )
        keys = [key for chunk in chunks for key in chunk]
        assert "__interrupt__" in keys, f"run did not pause: {keys}"

        snapshot = first_graph.get_state(config)
        # The pause is reported at the node the PER-PHOTO graph is mounted
        # at since issue #110 / P10.2 - `deep_path` is one level further down.
        assert snapshot.next == (DESCRIBE_ONE_NODE,)
        assert snapshot.interrupts, "no pending interrupt was checkpointed"
    finally:
        conn.close()

    # "Restart": a brand-new SqliteSaver, a brand-new connection, over the
    # SAME file - simulating a fresh process picking up where the last one
    # left off.
    second_conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        second_graph = build_graph(checkpointer=SqliteSaver(second_conn))

        resumed_snapshot = second_graph.get_state(config)
        assert resumed_snapshot.interrupts, "the pause did not survive the restart"

        resumed_chunks = list(
            second_graph.stream(
                Command(resume=RESUME_CONTINUE), config=config, stream_mode="updates"
            )
        )
        resumed_keys = [key for chunk in resumed_chunks for key in chunk]
        assert "tts" in resumed_keys, f"resume did not reach speech: {resumed_keys}"

        final_state = second_graph.get_state(config)
        assert "$999.99" in final_state.values["final_output"]
    finally:
        second_conn.close()


# --- (b) Factory selection ---------------------------------------------------


def test_make_checkpointer_returns_in_memory_saver_when_env_unset(monkeypatch):
    monkeypatch.delenv("CLARIFEYE_CHECKPOINT_DB", raising=False)
    saver = make_checkpointer()
    assert isinstance(saver, InMemorySaver)


def test_make_checkpointer_returns_sqlite_saver_on_named_file(monkeypatch, tmp_path):
    db_path = str(tmp_path / "app.sqlite")
    monkeypatch.setenv("CLARIFEYE_CHECKPOINT_DB", db_path)
    saver = make_checkpointer()
    try:
        assert isinstance(saver, SqliteSaver)
    finally:
        saver.conn.close()


# --- (c) delete_thread against SqliteSaver -----------------------------------


def test_delete_thread_removes_state_from_sqlite_saver(tmp_path):
    db_path = str(tmp_path / "delete.sqlite")
    # Closed via try/finally so a failing assertion can't leak a Windows
    # file lock into pytest's tmp_path cleanup.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        graph = build_graph(checkpointer=saver)
        thread_id = "delete-me"
        config = {"configurable": {"thread_id": thread_id}}

        state = make_initial_state("image-payload")
        # A short OCR keeps this on the fast path so the run completes with
        # no pause, and the point of this test is deletion, not
        # verification.
        run_config = {
            "configurable": {
                "thread_id": thread_id,
                "client": _ShortReplyClient(),
                "tts_provider": _FakeTtsProvider(),
            }
        }
        graph.invoke(state, config=run_config)

        before = graph.get_state(config)
        assert before.values, "setup is broken: nothing was checkpointed"

        saver.delete_thread(thread_id)

        after = graph.get_state(config)
        assert not after.values, "delete_thread left state behind on a SqliteSaver"
    finally:
        conn.close()


class _ShortReplyClient:
    """A vision/synth reply short and plain enough to stay on the fast path
    (vision -> fast_synth -> tts), so this file never has to drive the
    research/analysis path for a test that isn't about verification."""

    def complete(self, role, messages, **params):
        return CompletionResult(content="OCR_TEXT: hi\nSCENE: a desk", model="fake-eyes-model:free")

    def close(self):
        pass


# --- (d) Trim guard: a non-InMemorySaver's internals are never touched ------


class _RecordingStorageStub:
    """A bare stub - NOT an InMemorySaver, NOT a SqliteSaver, nothing that
    isinstance(checkpointer, InMemorySaver) could accidentally still match -
    whose `storage` attribute is a property that flips a flag the instant
    it is READ, so a test can observe whether the trim guard actually
    stopped `_trim_thread_to_latest_checkpoint` before it touched
    checkpointer-internals, rather than merely stopped it from raising.

    WHY THIS EXISTS INSTEAD OF JUST ASSERTING "no exception" (the previous
    version of this test, against a real SqliteSaver): the trim helper
    wraps its ENTIRE body in `try/except Exception: pass` (its own
    docstring: housekeeping must never take down a real photo's run over a
    langgraph version mismatch). That swallow means an AttributeError from
    reading `.storage` on an unguarded, non-InMemorySaver checkpointer is
    caught and discarded - NOT raised - so a test that only checked for "no
    exception escaped" would pass identically whether the isinstance guard
    exists or not; deleting the guard does not turn this into a red test,
    because the swallow hides it. A property that records ACCESS, not
    outcome, is observable straight through that swallow: it flips its flag
    the moment `.storage` is read, before whatever happens next (a crash,
    a silent no-op) even matters.
    """

    def __init__(self):
        self.storage_was_read = False

    @property
    def storage(self):
        self.storage_was_read = True
        return {}


def test_trim_helper_never_reads_a_non_inmemory_savers_storage():
    """MUTATION TARGET: delete the isinstance(checkpointer, InMemorySaver)
    guard inside _trim_thread_to_latest_checkpoint (clarif_eye.ui) and this
    test fails - the unguarded body reads `checkpointer.storage.get(...)`
    as its very first line, which flips _RecordingStorageStub's flag. See
    that class's own docstring for why the assertion is ACCESS-based rather
    than exception-based: the function's blanket `except Exception: pass`
    would otherwise hide the very AttributeError a naive version of this
    test might have expected to see.
    """
    stub = _RecordingStorageStub()

    _trim_thread_to_latest_checkpoint(stub, "never-touched")

    assert not stub.storage_was_read, (
        "the trim helper read a non-InMemorySaver's .storage - the "
        "isinstance guard did not stop it before touching internals"
    )
