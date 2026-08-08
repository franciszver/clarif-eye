"""Several photos described in ONE turn (issue #110 / P10.2).

The mechanism is LangGraph's `Send`: the entry node returns one Send per
submitted photo, so every photo runs the per-photo pipeline in the SAME
superstep, and their results accumulate into one state key through a
reducer. A join node then composes a single spoken script in SUBMISSION
order, and the one `tts` node at the end speaks it - once.

WHAT THESE TESTS PIN, and why each is separate from the others:

  - the fan-out actually fans out: three photos produce three `describe_one`
    completions and exactly one `compose` completion, in one run.
  - the composed script is in SUBMISSION order, not completion order. The
    graph test cannot prove that on its own (LangGraph happened to complete
    the branches in submission order when this was written, so a join that
    simply appended would pass it), so the join function is ALSO driven
    directly with its accumulated results shuffled - the only way to make
    the ordering claim fail when it is wrong.
  - N=1 is the DEGENERATE CASE of the same fan-out, not a separate branch:
    one photo goes through the identical nodes and comes out with an
    UNLABELLED script, exactly what a single-photo turn said before this
    issue. tests/test_graph.py's single-photo tests are the rest of that
    contract.
"""

import pytest

from clarif_eye.client import CompletionResult
from clarif_eye.graph import (
    COMPOSE_NODE,
    DESCRIBE_ONE_NODE,
    build_graph,
    compose_node,
)
from clarif_eye.state import make_initial_state, make_initial_state_for_photos

from tests._stream_helpers import drain_stream_collecting_trace

# Three photos, each with its own OCR text and scene, so the composed script
# can be checked for BOTH presence and order. The marker strings are what the
# fake client below matches on: the base64 "image data" is passed through to
# the eyes call verbatim, and the OCR text comes back out in the prompt the
# writing call receives.
PHOTOS = {
    "photo-alpha": ("apples", "a fruit bowl"),
    "photo-bravo": ("bread", "a bakery shelf"),
    "photo-charlie": ("cheese", "a deli counter"),
}


class PerPhotoClient:
    """One fake model client serving every photo's branch, answering
    according to WHICH photo the call is about.

    A single shared client is what the real app injects too (one
    OpenRouterClient on config["configurable"]["client"]), so a fan-out that
    accidentally handed every branch the same photo's text would still look
    fine with the flat fake the rest of the suite uses. This one keys off the
    content of the request, so crossed wires show up as crossed OCR text in
    the composed script.
    """

    def complete(self, role, messages, **params):
        blob = str(messages)
        # The eyes call: the photo's own data is in the request.
        for marker, (ocr, scene) in PHOTOS.items():
            if marker in blob:
                return CompletionResult(
                    content=f"OCR_TEXT: {ocr}\nSCENE: {scene}", model="fake-eyes-model:free"
                )
        # The writing call: no image data, but the OCR text this branch just
        # read is in the prompt.
        for _marker, (ocr, _scene) in PHOTOS.items():
            if ocr in blob:
                return CompletionResult(content=f"This is {ocr}.", model="fake-brain-model:free")
        raise AssertionError(f"fake client got a request about no known photo: {blob[:200]}")


class FakeTtsProvider:
    """Writes a minimal valid-looking mp3 (an ID3 tag) so run_tts's own
    "looks like audio" check passes without touching the network."""

    def synthesize(self, text, out_path):
        with open(out_path, "wb") as f:
            f.write(b"ID3" + b"\x00" * 32)


def _run(graph, state):
    return drain_stream_collecting_trace(
        graph,
        state,
        {"configurable": {"client": PerPhotoClient(), "tts_provider": FakeTtsProvider()}},
    )


def test_three_photos_fan_out_to_three_branches_and_join_into_one_script():
    graph = build_graph()
    state = make_initial_state_for_photos(["photo-alpha", "photo-bravo", "photo-charlie"])

    result, trace = _run(graph, state)

    # Three branches ran, and they joined exactly once.
    assert trace.count(DESCRIBE_ONE_NODE) == 3
    assert trace.count(COMPOSE_NODE) == 1
    # One turn, one voice: tts runs once over the combined script.
    assert trace.count("tts") == 1
    # Each photo got its own vision call in its own branch.
    assert trace.count("vision") == 3

    combined = result["final_output"]
    for _marker, (ocr, _scene) in PHOTOS.items():
        assert ocr in combined
    assert combined.index("apples") < combined.index("bread") < combined.index("cheese")
    # The listener is told which photo each part is about.
    assert "Photo 1 of 3" in combined
    assert "Photo 3 of 3" in combined

    # The thread is left describing ALL the photos, so a follow-up question
    # is answered from the union of what was read - see
    # clarif_eye.followup / clarif_eye.verification for the haystack.
    assert "apples" in result["ocr_output"]
    assert "cheese" in result["ocr_output"]
    assert result["output_degraded"] is False


def test_composition_order_follows_submission_not_completion():
    """The join sorts by the submission index it was handed, so a branch
    that finished last still speaks in the position it was submitted in.

    Driven directly rather than through the graph: LangGraph completed the
    branches in submission order when this was written, so a join that
    simply appended in arrival order would pass the graph test above and
    still be wrong the first time a slow photo finished out of turn.
    """
    shuffled = [
        {
            "index": 2,
            "final_output": "This is cheese.",
            "ocr_output": "cheese",
            "scene_context": "a deli counter",
            "output_degraded": False,
        },
        {
            "index": 0,
            "final_output": "This is apples.",
            "ocr_output": "apples",
            "scene_context": "a fruit bowl",
            "output_degraded": False,
        },
        {
            "index": 1,
            "final_output": "This is bread.",
            "ocr_output": "bread",
            "scene_context": "a bakery shelf",
            "output_degraded": False,
        },
    ]

    composed = compose_node({"photo_results": shuffled})

    script = composed["final_output"]
    assert script.index("apples") < script.index("bread") < script.index("cheese")
    assert composed["output_degraded"] is False


def test_one_photo_is_the_degenerate_fan_out_and_speaks_an_unlabelled_script():
    """A single photo goes through the SAME fan-out - one Send, one branch,
    one join - and comes out with no "Photo 1 of 1" label, which is exactly
    what a single-photo turn said before this issue existed."""
    graph = build_graph()
    state = make_initial_state("photo-alpha")

    result, trace = _run(graph, state)

    assert trace.count(DESCRIBE_ONE_NODE) == 1
    assert trace.count(COMPOSE_NODE) == 1
    assert "Photo 1" not in result["final_output"]
    assert "apples" in result["final_output"]


def test_a_failing_photo_degrades_the_whole_turn_and_names_which_one():
    """One photo the eyes could not read makes the COMBINED turn degraded -
    the listener is told which photo failed, by position, and the turn is
    marked so the conversation boundary skips it (issue #93 / P9.12)."""
    composed = compose_node(
        {
            "photo_results": [
                {
                    "index": 0,
                    "final_output": "This is apples.",
                    "ocr_output": "apples",
                    "scene_context": "a fruit bowl",
                    "output_degraded": False,
                },
                {
                    "index": 1,
                    "final_output": "The photo could not be read.",
                    "ocr_output": "",
                    "scene_context": "",
                    "output_degraded": True,
                },
            ]
        }
    )

    assert composed["output_degraded"] is True
    assert "second photo" in composed["final_output"]
    # The photo that DID work is still spoken - a failure on one photo must
    # not throw away the description of another.
    assert "apples" in composed["final_output"]


def test_make_initial_state_for_photos_rejects_an_empty_submission():
    with pytest.raises(ValueError):
        make_initial_state_for_photos([])
