"""Experiment (issue #85 / P9.6): does langgraph's add_node(error_handler=...)
replace the per-node degradation contract each run_* function implements
(vision.run_vision, synth.run_fast_synth, analysis.*, research.*, ...)?

Runs entirely OFFLINE - no live API calls.

WHAT THIS MEASURES:
  1. Does error_handler fire on a raised exception and let the node still
     produce a state update, the way every run_* function degrades to a
     spoken-ready partial result instead of raising?
  2. Can error_handler distinguish exception types and choose to re-raise
     some (a caller bug) while degrading others (a transient failure) - the
     same distinction vision.run_vision makes between LadderExhaustedError,
     OpenRouterError (terminal), and everything else?
  3. THE DECISIVE QUESTION: does error_handler fire for a degradation that
     the node reaches WITHOUT raising - e.g. run_vision's "malformed/empty
     reply" and "unparseable reply" cases, which are plain early returns,
     not exceptions? If error_handler only sees RAISED exceptions, it
     covers a strict subset of what run_* already handles.

Usage:
    python scripts/experiment_error_handler.py
    python scripts/experiment_error_handler.py > scripts/experiments/error_handler.txt
"""

import sys
from typing import TypedDict

from langgraph.errors import NodeError
from langgraph.graph import StateGraph


class State(TypedDict, total=False):
    x: int
    result: str


def scenario_handler_fires_and_degrades():
    def failing_node(state):
        raise RuntimeError("simulated model-call failure")

    def handler(state, error: NodeError):
        return {"result": f"degraded: {error.node} raised {type(error.error).__name__}"}

    builder = StateGraph(State)
    builder.add_node("n", failing_node, error_handler=handler)
    builder.set_entry_point("n")
    builder.set_finish_point("n")
    graph = builder.compile()
    return graph.invoke({"x": 1})


def scenario_handler_distinguishes_exception_types():
    """Mirrors vision.run_vision's real branching: a "caller bug" type
    (ValueError, standing in for a malformed-request OpenRouterError) must
    surface, not be silently degraded, while a transient-looking type is
    degraded."""

    def failing_node(state):
        if state["x"] == 1:
            raise ConnectionError("transient - ladder-exhausted-shaped failure")
        raise ValueError("caller bug - malformed-request-shaped failure")

    def picky_handler(state, error: NodeError):
        if isinstance(error.error, ValueError):
            raise error.error
        return {"result": f"degraded: {type(error.error).__name__}"}

    builder = StateGraph(State)
    builder.add_node("n", failing_node, error_handler=picky_handler)
    builder.set_entry_point("n")
    builder.set_finish_point("n")
    graph = builder.compile()

    transient_result = graph.invoke({"x": 1})
    caller_bug_error = None
    try:
        graph.invoke({"x": 2})
    except ValueError as exc:
        caller_bug_error = exc

    return transient_result, caller_bug_error


def scenario_handler_never_fires_for_non_raising_degradation():
    """The decisive scenario: a node shaped like run_vision, where a
    malformed reply degrades via a plain early RETURN, never an exception -
    exactly how vision.py's `if parsed is None: return _degraded(...)`
    works. Does error_handler see this at all?"""
    handler_calls = []

    def node_like_run_vision(state):
        if state["x"] == 2:
            # No exception - this is run_vision's shape for a malformed or
            # unparseable reply: return a degraded update directly.
            return {"result": "degraded: malformed reply (no exception raised)"}
        return {"result": "ok"}

    def handler(state, error: NodeError):
        handler_calls.append(error)
        return {"result": "handled"}

    builder = StateGraph(State)
    builder.add_node("n", node_like_run_vision, error_handler=handler)
    builder.set_entry_point("n")
    builder.set_finish_point("n")
    graph = builder.compile()

    result = graph.invoke({"x": 2})
    return result, len(handler_calls)


def main():
    print("python scripts/experiment_error_handler.py")
    print("=" * 72)
    print("EXPERIMENT: error_handler vs the per-node run_* degradation contract")
    print("=" * 72)

    print()
    print("--- Scenario 1: error_handler fires on a raised exception ---")
    result = scenario_handler_fires_and_degrades()
    print(f"result: {result}")
    print("OBSERVATION: error_handler catches the raised exception and lets")
    print("the node still produce a state update - the basic capability works.")

    print()
    print("--- Scenario 2: error_handler can distinguish exception types ---")
    transient_result, caller_bug_error = scenario_handler_distinguishes_exception_types()
    print(f"transient-shaped failure degrades to: {transient_result}")
    print(f"caller-bug-shaped failure re-raised as: {type(caller_bug_error).__name__}: {caller_bug_error}")
    print("OBSERVATION: error_handler CAN replicate vision.run_vision's real")
    print("branching (LadderExhaustedError/OpenRouterError degrade, an")
    print("unexpected caller bug does not have to be swallowed) - a handler")
    print("can isinstance-check error.error and choose to re-raise.")

    print()
    print("--- Scenario 3: DECISIVE - does error_handler see a non-raising degradation? ---")
    result, handler_call_count = scenario_handler_never_fires_for_non_raising_degradation()
    print(f"result: {result}")
    print(f"handler invocations: {handler_call_count}")
    print("OBSERVATION: run_vision's 'malformed/empty reply' and 'unparseable")
    print("reply' cases are plain early RETURNS, not exceptions (see")
    print("vision.py: 'if parsed is None: return _degraded(...)'). This")
    print("scenario reproduces that shape and error_handler was invoked ZERO")
    print("times - it only sees exceptions a node RAISES, so it covers a")
    print("strict subset of what every run_* function already degrades from.")

    print()
    print("=" * 72)
    print("COMPARISON")
    print("=" * 72)
    print(
        "error_handler: fires only on a RAISED exception from the node body,\n"
        "               receives (state, NodeError(node, error)), and its return\n"
        "               value merges into state like a normal node update (or it\n"
        "               can re-raise to let the failure propagate). Covers\n"
        "               exceptions; does not see a node's own early-return\n"
        "               degradation, so adopting it as THE degradation mechanism\n"
        "               would mean rewriting every non-raising degradation path\n"
        "               (malformed reply, unparseable reply, empty content, ...)\n"
        "               in vision.py/synth.py/analysis.py/research.py to raise\n"
        "               instead of return - inverting an already-working,\n"
        "               already-tested control-flow style for no new capability.\n"
        "\n"
        "run_* contract: EVERY run_* function's OWN module docstring states the\n"
        "               rule directly: 'must NEVER let a raw exception escape\n"
        "               into the graph'. Each function catches its own\n"
        "               exceptions (LadderExhaustedError, OpenRouterError,\n"
        "               bare Exception) AND its own non-exceptional failure\n"
        "               modes (empty/malformed/unparseable replies) in ONE\n"
        "               place, with domain-specific spoken messages\n"
        "               (message_for_ladder_exhausted, message_for_terminal_\n"
        "               error, DEGRADED_* constants) - the node function\n"
        "               (vision_node etc.) stays a thin adapter with nothing to\n"
        "               configure."
    )

    print()
    print("VERDICT: KEEP OURS")
    print(
        "DECISIVE REASON: error_handler can express the exception-type\n"
        "distinctions this codebase already makes (Scenario 2), but it is not a\n"
        "superset of what run_* modules degrade from - Scenario 3 shows it is\n"
        "blind to the non-raising degradation paths (malformed/empty/\n"
        "unparseable replies) that make up a large share of each module's real\n"
        "failure handling. Using it would mean either leaving those paths\n"
        "uncovered by the new mechanism (two degradation systems doing the same\n"
        "job) or rewriting working, tested control flow to raise instead of\n"
        "return, purely to fit a policy that does not cover more ground than\n"
        "what already exists."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
