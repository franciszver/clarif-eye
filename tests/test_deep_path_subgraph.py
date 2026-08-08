"""Tests for extracting the deep path into a subgraph (issue #84 / P9.5).

RED FIRST: at the time this file was first committed, clarif_eye.deep_path
did not exist, the parent graph still registered `research`/`analysis`/
`verify_numbers` as its own nodes, clarif_eye.ui had no
describe_document_text, and _narrate_stream still consumed FLAT stream
chunks - so every test reaching for the child graph or the text-only route
failed, while the two BYTE-IDENTITY guards at the bottom of this file passed
exactly as they do today.

WHY THE NEW SYMBOLS ARE IMPORTED INSIDE THE TESTS THAT NEED THEM rather than
at the top of this file: the two byte-identity guards are the red line of
this issue ("the user-audible sequence must stay identical"), and they are
only worth anything if they can be RUN against the pre-extraction code and
shown to pass. A module-level import of a module that does not exist yet
would have turned them into collection errors instead of a green baseline.

Same no-network discipline as the rest of this suite: real compiled graphs,
a real checkpointer, fake client, fake searcher, fake TTS provider. Nothing
here launches Gradio or opens a socket.
"""

import time

from langgraph.graph import StateGraph
from langgraph.types import Command

from clarif_eye import tts as tts_module
from clarif_eye.graph import (
    INTERRUPT_CHUNK_KEY,
    RESUME_CONTINUE,
    UNVERIFIED_NUMBER_CAVEAT,
    build_graph,
)
from clarif_eye.state import ClarifEyeState, make_initial_state
from clarif_eye.ui import (
    STATUS_NODE_RESEARCH,
    STATUS_NODE_TTS,
    STATUS_NODE_WRITING,
    STATUS_RESUMING,
    STATUS_WORKING,
    handle_resume_staged,
    handle_submit_staged,
)

from tests._stream_helpers import drain_stream_collecting_trace
from tests.test_ask_before_speaking import (
    BILL_OCR,
    BILL_SCENE,
    HONEST_DRAFT,
    INVENTED_DRAFT,
    FakeSearcher,
    RecordingClient,
    _FakeTtsProvider,
    _resources,
)
from tests.test_ui import FakeImage


def setup_function(_fn):
    # tts.is_chain_exhausted() reads module-level state left by the last real
    # run_tts() call - reset it so tests don't leak into each other.
    tts_module._last_result_set(None)


def _child_config(client, searcher=None):
    return {"configurable": {"client": client, "searcher": searcher or FakeSearcher()}}


def _parent_config(client, thread_id=None):
    configurable = {
        "client": client,
        "searcher": FakeSearcher(),
        "tts_provider": _FakeTtsProvider(),
    }
    if thread_id is not None:
        configurable["thread_id"] = thread_id
    return {"configurable": configurable}


# --- The child graph, standing on its own ---------------------------------


def test_child_graph_is_invocable_standalone_with_its_own_schema():
    """The whole point of own-schema mode: the deep path runs with NO parent
    around it, from inputs named in ITS vocabulary, and hands back a spoken
    script."""
    from clarif_eye.deep_path import build_deep_path_graph

    child = build_deep_path_graph()

    result = child.invoke(
        {"document_text": BILL_OCR, "scene_description": BILL_SCENE},
        config=_child_config(RecordingClient(HONEST_DRAFT)),
    )

    assert "$104.95" in result["final_output"]
    # Its OWN input names, not the parent's - proof the caller had to speak
    # the child's vocabulary to get here at all.
    assert result["document_text"] == BILL_OCR
    assert result["scene_description"] == BILL_SCENE


def test_child_schema_is_its_own_and_never_carries_the_photo():
    """ENCAPSULATION, asserted rather than claimed: the child's schema has
    exactly the keys the deep path needs, and image_data is not one of them -
    the base64 JPEG cannot reach the child's checkpoints at all."""
    from clarif_eye.deep_path import DeepPathState

    assert set(DeepPathState.__annotations__) == {
        "document_text",
        "scene_description",
        "scraper_data",
        "verification_hold",
        "final_output",
    }
    assert "image_data" not in DeepPathState.__annotations__
    # And it is genuinely a DIFFERENT schema, not a subset of the parent's:
    # the two input names do not exist in ClarifEyeState at all.
    assert "document_text" not in ClarifEyeState.__annotations__
    assert "scene_description" not in ClarifEyeState.__annotations__


def test_the_child_cannot_be_mounted_without_the_mapping_wrapper():
    """The mapping wrapper is LOAD-BEARING, proven by removing it.

    A child whose schema keys are a subset of the parent's with the SAME
    names can be handed straight to add_node and LangGraph maps it for free -
    which is the shared-state mode this parent graph already is, and which
    would make the wrapper vanish. Renaming the two inputs is what keeps
    own-schema mode real: mounted bare, the child raises the moment its first
    node reads a key the parent never wrote.
    """
    from clarif_eye.deep_path import build_deep_path_graph

    builder = StateGraph(ClarifEyeState)
    builder.add_node("deep_path", build_deep_path_graph())
    builder.set_entry_point("deep_path")
    bare = builder.compile()

    try:
        bare.invoke(make_initial_state("base64photo"), config=_child_config(RecordingClient(HONEST_DRAFT)))
    except KeyError as exc:
        assert "document_text" in str(exc)
    else:
        raise AssertionError("mounting the child bare should not have worked")


# --- The child inside the parent ------------------------------------------


def test_the_parent_registers_the_child_as_one_node():
    graph = build_graph()

    assert "deep_path" in graph.nodes
    # The three extracted nodes are the CHILD's now, not the parent's.
    for extracted in ("research", "analysis", "verify_numbers"):
        assert extracted not in graph.nodes


def test_child_node_events_are_visible_in_the_parent_stream():
    """stream(subgraphs=True) is what makes the child's own nodes observable
    from the top - without it the whole deep path is one opaque `deep_path`
    completion and the user hears nothing between "photo received" and
    "turning it into speech".

    This test opens the stream itself, so it pins what LangGraph EMITS. What
    the app actually CONSUMES is pinned by the two byte-identity guards at
    the bottom of this file - verified by mutation, not assumed: dropping
    subgraphs=True from clarif_eye.ui._narrate_stream turns both of them red,
    because the child's two announcements vanish from what the user hears.
    """
    graph = build_graph()

    seen = [
        (namespace, node_name)
        for namespace, chunk in graph.stream(
            make_initial_state("base64photo"),
            config=_parent_config(RecordingClient(HONEST_DRAFT)),
            stream_mode="updates",
            subgraphs=True,
        )
        for node_name in chunk
    ]

    child_events = [(ns, name) for ns, name in seen if ns]
    assert [name for _ns, name in child_events] == ["research", "analysis"]
    # Every child event is namespaced under the node the child is mounted at.
    for namespace, _name in child_events:
        assert namespace[0].startswith("deep_path:")
    # The parent's own view still names the child ONCE, as a single node.
    assert [name for ns, name in seen if not ns] == ["entry", "vision", "deep_path", "tts"]


def test_the_deadline_reaches_the_child_s_nodes():
    """config["configurable"] flows into a subgraph invoked from a parent
    node - verified empirically before this was relied on. Without it the
    deep path would silently lose the whole-pipeline deadline the moment it
    was extracted, and a blown budget would stop degrading."""
    from tests.test_pipeline_deadline import ExplosiveSearcher, FakeEyesClient, _reply

    graph = build_graph()
    searcher = ExplosiveSearcher()
    long_ocr = " ".join(["x"] * 200)
    client = FakeEyesClient(_reply(long_ocr, "a dense long document"))
    config = {
        "configurable": {
            "client": client,
            "searcher": searcher,
            "tts_provider": _FakeTtsProvider(),
            "deadline": time.monotonic() - 1.0,
        }
    }

    result, _trace = drain_stream_collecting_trace(graph, make_initial_state("imgdata"), config)

    assert searcher.called is False, "the child's research node did not see the deadline"
    assert client.calls == [], "the child's analysis node did not see the deadline"
    assert result["final_output"] != ""


# --- The interrupt, fired from inside the child ---------------------------


def test_interrupt_from_inside_the_child_reaches_the_top_level_caller():
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    config = _parent_config(RecordingClient(INVENTED_DRAFT), thread_id="child-interrupt")

    chunks = list(
        graph.stream(make_initial_state("base64photo"), config=config, stream_mode="updates")
    )

    keys = [key for chunk in chunks for key in chunk]
    assert INTERRUPT_CHUNK_KEY in keys, f"the child's pause never reached the top: {keys}"
    assert "tts" not in keys, "a paused run must not reach speech"

    snapshot = graph.get_state(config)
    # The PARENT reports the pause at the node the child is mounted at, and
    # carries the child's own interrupt payload up unchanged.
    assert snapshot.next == ("deep_path",)
    assert snapshot.interrupts
    payload = snapshot.interrupts[0].value
    assert payload["reason"] == "unverified_numbers"
    assert payload["script"] == INVENTED_DRAFT
    assert payload["numbers"] == ["$999.99"]


def test_resume_on_the_parent_config_reaches_back_into_the_child():
    from langgraph.checkpoint.memory import InMemorySaver

    client = RecordingClient(INVENTED_DRAFT)
    graph = build_graph(checkpointer=InMemorySaver())
    config = _parent_config(client, thread_id="child-resume")
    list(graph.stream(make_initial_state("base64photo"), config=config, stream_mode="updates"))
    assert len(client.brain_calls()) == 1

    list(graph.stream(Command(resume=RESUME_CONTINUE), config=config, stream_mode="updates"))

    snapshot = graph.get_state(config)
    assert INVENTED_DRAFT in snapshot.values["final_output"]
    assert UNVERIFIED_NUMBER_CAVEAT in snapshot.values["final_output"]
    assert snapshot.values["audio_file_path"]
    assert snapshot.next == ()
    assert not snapshot.interrupts
    # THE COST THAT MUST NOT DOUBLE: the child resumes at its paused node, so
    # the brain call ahead of it is not paid a second time - the wrapper node
    # re-executes, the finished child nodes do not.
    assert len(client.brain_calls()) == 1


# --- Bounded memory across the new namespace boundary ---------------------


def test_trimming_drops_the_child_namespaces_of_finished_runs():
    """Each deep-path run mounts the child under a FRESH checkpoint namespace
    (`deep_path:<task id>`), so a thread that runs the deep path repeatedly
    accumulates one dead namespace per run - the same unbounded growth issue
    #81 measured and closed for the root namespace, reopened one level down.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    from clarif_eye.ui import _trim_thread_to_latest_checkpoint

    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    for _ in range(3):
        config = _parent_config(RecordingClient(HONEST_DRAFT), thread_id="growing")
        list(graph.stream(make_initial_state("base64photo"), config=config, stream_mode="updates"))
        _trim_thread_to_latest_checkpoint(checkpointer, "growing")

    namespaces = list(checkpointer.storage["growing"])
    child_namespaces = [ns for ns in namespaces if ns]
    assert len(child_namespaces) <= 1, f"dead child namespaces piled up: {namespaces}"


# --- The second consumer: text in, description out, no photo --------------


def test_describe_document_text_returns_a_spoken_script():
    from clarif_eye.ui import describe_document_text

    resources = _resources(RecordingClient(HONEST_DRAFT))

    spoken = describe_document_text(BILL_OCR, resources)

    assert "$104.95" in spoken


def test_describe_document_text_never_asks_a_question_it_cannot_ask():
    """The text-only route has no UI, so there is nobody to answer a pause
    and nothing to resume with. The standalone child is compiled WITHOUT a
    checkpointer, so an unverifiable number degrades to the safe script the
    analysis path already produces instead of raising or hanging."""
    from clarif_eye.ui import describe_document_text

    resources = _resources(RecordingClient(INVENTED_DRAFT))

    spoken = describe_document_text(BILL_OCR, resources)

    assert "999.99" not in spoken
    assert "could not be verified" in spoken


def test_describe_document_text_never_raises():
    from clarif_eye.ui import CONFIG_ERROR_MESSAGE, NO_DOCUMENT_TEXT_MESSAGE, describe_document_text

    resources = _resources(RecordingClient(HONEST_DRAFT))

    assert describe_document_text("", resources) == NO_DOCUMENT_TEXT_MESSAGE
    assert describe_document_text("   ", resources) == NO_DOCUMENT_TEXT_MESSAGE
    assert describe_document_text(None, resources) == NO_DOCUMENT_TEXT_MESSAGE

    unconfigured = _resources(RecordingClient(HONEST_DRAFT))
    unconfigured.client = None
    assert describe_document_text(BILL_OCR, unconfigured) == CONFIG_ERROR_MESSAGE


def test_the_text_only_consumer_is_an_api_route_and_not_in_the_ui_flow():
    from clarif_eye.ui import DESCRIBE_TEXT_API_NAME, build_interface

    resources = _resources(RecordingClient(HONEST_DRAFT))
    demo = build_interface(resources)
    try:
        route = next(
            fn for fn in demo.fns.values() if getattr(fn, "api_name", None) == DESCRIBE_TEXT_API_NAME
        )
        # It is reachable over the API only - never as a control someone has
        # to tab past in the main flow. gr.api gives the endpoint synthetic
        # `Api` placeholders derived from the function's type hints instead of
        # real components, so "nothing was added to the page" is asserted as
        # "every component on this route is one of those placeholders".
        for component in list(route.inputs) + list(route.outputs):
            assert type(component).__name__ == "Api", f"the API route rendered a real control: {component}"
        assert route.fn(BILL_OCR).strip()
    finally:
        demo.close()


# --- THE RED LINE: the user-audible sequence must not change --------------
#
# These two are written against what the app does TODAY, before any
# extraction, and were run against the pre-extraction code to prove they
# pass there. They exist to go RED if the subgraph move changes a single
# announcement a blind user hears - a doubled "Turning it into speech" from
# the parent's own `deep_path` completion chunk being the specific hazard
# subgraph streaming introduces.


def _statuses(staged):
    return [status for status, _audio, _text in staged]


def test_deep_path_narration_is_byte_identical_after_extraction():
    resources = _resources(RecordingClient(HONEST_DRAFT))

    statuses = _statuses(handle_submit_staged(FakeImage(), resources, thread_id="narrate-clean"))

    assert statuses[:4] == [
        STATUS_WORKING,
        STATUS_NODE_RESEARCH,
        STATUS_NODE_WRITING,
        STATUS_NODE_TTS,
    ]
    # And nothing is said twice: the completion status closes it out, once
    # without audio and once with.
    assert len(statuses) == 6
    assert statuses[4] == statuses[5]


def test_paused_and_resumed_narration_is_byte_identical_after_extraction():
    resources = _resources(RecordingClient(INVENTED_DRAFT))

    paused = _statuses(handle_submit_staged(FakeImage(), resources, thread_id="narrate-pause"))
    assert paused[:3] == [STATUS_WORKING, STATUS_NODE_RESEARCH, STATUS_NODE_WRITING]
    # The question itself is the last thing said, and nothing is announced
    # for the node that asks it.
    assert len(paused) == 4
    assert INVENTED_DRAFT in paused[3]

    resumed = _statuses(handle_resume_staged(RESUME_CONTINUE, resources, thread_id="narrate-pause"))
    assert resumed[:2] == [STATUS_RESUMING, STATUS_NODE_TTS]
    assert len(resumed) == 4
