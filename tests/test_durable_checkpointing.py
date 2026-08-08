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
from clarif_eye.graph import DEEP_PATH_NODE, RESUME_CONTINUE, build_graph, make_checkpointer
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

    # First "process": pause a run on a SqliteSaver over db_path.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    first_saver = SqliteSaver(conn)
    first_graph = build_graph(checkpointer=first_saver)
    config = _run_config(thread_id)

    chunks = list(
        first_graph.stream(make_initial_state("base64photo"), config=config, stream_mode="updates")
    )
    keys = [key for chunk in chunks for key in chunk]
    assert "__interrupt__" in keys, f"run did not pause: {keys}"

    snapshot = first_graph.get_state(config)
    assert snapshot.next == (DEEP_PATH_NODE,)
    assert snapshot.interrupts, "no pending interrupt was checkpointed"
    conn.close()

    # "Restart": a brand-new SqliteSaver, a brand-new connection, over the
    # SAME file - simulating a fresh process picking up where the last one
    # left off.
    second_conn = sqlite3.connect(db_path, check_same_thread=False)
    second_saver = SqliteSaver(second_conn)
    second_graph = build_graph(checkpointer=second_saver)

    resumed_snapshot = second_graph.get_state(config)
    assert resumed_snapshot.interrupts, "the pause did not survive the restart"

    resumed_chunks = list(
        second_graph.stream(Command(resume=RESUME_CONTINUE), config=config, stream_mode="updates")
    )
    resumed_keys = [key for chunk in resumed_chunks for key in chunk]
    assert "tts" in resumed_keys, f"resume did not reach speech: {resumed_keys}"

    final_state = second_graph.get_state(config)
    assert "$999.99" in final_state.values["final_output"]
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
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    graph = build_graph(checkpointer=saver)
    thread_id = "delete-me"
    config = {"configurable": {"thread_id": thread_id}}

    state = make_initial_state("image-payload")
    # A short OCR keeps this on the fast path so the run completes with no
    # pause, and the point of this test is deletion, not verification.
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
    conn.close()


class _ShortReplyClient:
    """A vision/synth reply short and plain enough to stay on the fast path
    (vision -> fast_synth -> tts), so this file never has to drive the
    research/analysis path for a test that isn't about verification."""

    def complete(self, role, messages, **params):
        return CompletionResult(content="OCR_TEXT: hi\nSCENE: a desk", model="fake-eyes-model:free")

    def close(self):
        pass


# --- (d) Trim guard: SqliteSaver is never poked ------------------------------


def test_trim_helper_skips_sqlite_saver_without_touching_its_internals(tmp_path):
    """MUTATION TARGET: deleting the isinstance(checkpointer, InMemorySaver)
    guard inside _trim_thread_to_latest_checkpoint (clarif_eye.ui) makes this
    test fail - SqliteSaver has no `.storage` attribute, so the unguarded
    body would raise AttributeError the instant it tried
    `checkpointer.storage.get(thread_id)`. The guard is what turns that
    crash into "run only for InMemorySaver, otherwise return immediately".
    """
    db_path = str(tmp_path / "trim-guard.sqlite")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    assert not hasattr(saver, "storage"), (
        "test assumption broken: SqliteSaver now exposes .storage, so the "
        "unguarded trim body would no longer raise AttributeError on it"
    )

    # Must return cleanly (no exception) with no state ever having been
    # written for this thread_id - the trim helper is a no-op for anything
    # that is not an InMemorySaver.
    _trim_thread_to_latest_checkpoint(saver, "never-touched")
    conn.close()
