"""Experiment (issue #85 / P9.6): does langgraph's add_node(retry_policy=...)
replace the model-ladder failover in clarif_eye.client.OpenRouterClient.complete?

Runs entirely OFFLINE - no live API calls. A minimal StateGraph with one
node stands in for a model call; the node's behaviour (fail N times then
succeed, or fail forever) is controlled by a fake, deterministic counter,
the same way tests/ fakes an HTTP transport for OpenRouterClient.

WHAT THIS MEASURES:
  1. Does retry_policy re-invoke the SAME node body on failure, with no
     information about which attempt this is unless the node tracks it
     itself? (Verified against langgraph.types.RetryPolicy's fields:
     initial_interval, backoff_factor, max_interval, max_attempts, jitter,
     retry_on - none of these hands the node an attempt index or a
     "what failed last time" value.)
  2. Does retry_policy exhaust after max_attempts and raise, the same
     shape client.py's LadderExhaustedError gives after every rung fails?
  3. Can retry_policy express "try model A, then a DIFFERENT model B" - the
     actual behaviour of the model ladder - without the node itself holding
     a list of models and indexing into it (i.e. reimplementing the ladder
     inside the node body, which defeats the point of a declarative
     framework policy)?
  4. THE REAL INTEGRATION POINT: what happens if retry_policy wraps a node
     that already runs a full client.complete() call internally - i.e. a
     node shaped like vision_node, where the "failure" retry_policy would
     see is LadderExhaustedError, raised only AFTER the ladder inside that
     one call has already exhausted every rung within its own role budget?

Usage:
    python scripts/experiment_retry_policy.py
    python scripts/experiment_retry_policy.py > scripts/experiments/retry_policy.txt
"""

import sys
import time
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import StateGraph
from langgraph.types import RetryPolicy

from clarif_eye.client import Attempt, LadderExhaustedError


class FlakyState(TypedDict, total=False):
    result: str
    attempts_seen_by_node: int


@dataclass
class FlakyCounter:
    """Fake failing call, the same shape as a fake HTTP transport in
    tests/: fails `fail_times` times, then succeeds. `calls` is a log of
    every invocation the node made, so the experiment can show what the
    node itself observed about which attempt it was on."""

    fail_times: int
    calls: list = field(default_factory=list)

    def __call__(self):
        self.calls.append(time.monotonic())
        if len(self.calls) <= self.fail_times:
            # ConnectionError, not RuntimeError: empirically, langgraph's
            # default_retry_on (langgraph._internal._retry) treats
            # RuntimeError/ValueError/TypeError/... as NOT retryable (it
            # assumes they are programming errors) and only retries
            # ConnectionError, httpx.HTTPStatusError/requests.HTTPError for
            # 5xx, or any exception type it does not otherwise recognise.
            # A first draft of this script raised RuntimeError and every
            # scenario failed on attempt 1 with retry_policy silently doing
            # nothing - see the recorded output's own note on this.
            raise ConnectionError(f"simulated transient failure #{len(self.calls)}")
        return "ok"


def build_retry_graph(counter, retry_policy):
    """One node, gated by `retry_policy`, that calls `counter()`.

    The node has NO access to which attempt number this is except by
    reading `len(counter.calls)` itself - retry_policy passes nothing
    extra to the node function. That is the fact this experiment exists
    to pin down empirically rather than assume from the docs.
    """

    def flaky_node(state):
        counter()
        return {"result": "ok", "attempts_seen_by_node": len(counter.calls)}

    builder = StateGraph(FlakyState)
    builder.add_node("flaky", flaky_node, retry_policy=retry_policy)
    builder.set_entry_point("flaky")
    builder.set_finish_point("flaky")
    return builder.compile()


def run_retries_until_success():
    """A node that fails twice then succeeds, under a retry_policy with
    max_attempts=3 and a near-zero interval (offline, must be fast)."""
    counter = FlakyCounter(fail_times=2)
    policy = RetryPolicy(initial_interval=0.001, backoff_factor=1.0, max_attempts=3, jitter=False)
    graph = build_retry_graph(counter, policy)
    result = graph.invoke({})
    return counter, result


def run_retries_exhausted():
    """A node that always fails, under the same policy - retry_policy must
    exhaust and raise, the same shape LadderExhaustedError gives."""
    counter = FlakyCounter(fail_times=999)
    policy = RetryPolicy(initial_interval=0.001, backoff_factor=1.0, max_attempts=3, jitter=False)
    graph = build_retry_graph(counter, policy)
    try:
        graph.invoke({})
        return counter, None
    except Exception as exc:  # noqa: BLE001 - the experiment wants to see whatever type this is
        return counter, exc


def run_model_switch_attempt():
    """Try to express "attempt 1 uses model A, attempt 2 uses model B" -
    the ladder's actual behaviour - using ONLY what retry_policy hands the
    node. The node has to track its own attempt count and index into its
    own model list to do this; retry_policy contributes nothing to the
    model-selection decision itself.
    """
    models_tried = []

    def switching_node(state):
        ladder = ["fake/model-a", "fake/model-b", "fake/model-c"]
        # The node must maintain this itself - retry_policy's callback
        # signature (state) -> update never surfaces an attempt index.
        attempt_index = len(models_tried)
        model = ladder[min(attempt_index, len(ladder) - 1)]
        models_tried.append(model)
        if attempt_index < 2:
            raise ConnectionError(f"{model} unavailable")
        return {"result": model}

    policy = RetryPolicy(initial_interval=0.001, backoff_factor=1.0, max_attempts=3, jitter=False)
    builder = StateGraph(FlakyState)
    builder.add_node("switching", switching_node, retry_policy=policy)
    builder.set_entry_point("switching")
    builder.set_finish_point("switching")
    graph = builder.compile()
    result = graph.invoke({})
    return models_tried, result


def run_retry_policy_around_an_already_exhausted_ladder():
    """THE REAL INTEGRATION POINT (not a hypothetical): wrap retry_policy
    around a node shaped like vision_node, which calls client.complete()
    internally and lets LadderExhaustedError propagate. Inside ONE node
    invocation, the ladder has ALREADY tried every rung within its own
    role budget before raising - that is what LadderExhaustedError means.
    Does retry_policy's default retry_on treat that as retryable?
    """
    node_calls = []

    def vision_shaped_node(state):
        node_calls.append(1)
        # This is exactly what a real vision_node raises once client.complete
        # exhausts the eyes ladder (two rungs here, standing in for a real
        # ladder) - the SAME exception whether this is the 1st or 3rd
        # invocation, because each invocation re-runs the full ladder.
        raise LadderExhaustedError(
            "eyes",
            (
                Attempt("fake/model-a", "rate_limited", 429, "rate limited"),
                Attempt("fake/model-b", "server_error", 500, "server error"),
            ),
        )

    policy = RetryPolicy(initial_interval=0.001, backoff_factor=1.0, max_attempts=3, jitter=False)
    builder = StateGraph(FlakyState)
    builder.add_node("vision_shaped", vision_shaped_node, retry_policy=policy)
    builder.set_entry_point("vision_shaped")
    builder.set_finish_point("vision_shaped")
    graph = builder.compile()

    try:
        graph.invoke({})
        final_error = None
    except Exception as exc:  # noqa: BLE001
        final_error = exc

    return node_calls, final_error


def main():
    print("python scripts/experiment_retry_policy.py")
    print("=" * 72)
    print("EXPERIMENT: retry_policy vs the model-ladder failover in client.py")
    print("=" * 72)

    print()
    print("--- Scenario 1: node fails twice, succeeds on 3rd attempt ---")
    counter, result = run_retries_until_success()
    print(f"attempts made: {len(counter.calls)}")
    print(f"final state:   {result}")
    print("OBSERVATION: retry_policy re-invoked the SAME node body 3 times")
    print("with no framework-supplied signal distinguishing attempt 2 from")
    print("attempt 1 beyond what the node counted itself.")
    print()
    print("SURPRISE, found empirically while writing this scenario:")
    print("langgraph._internal._retry.default_retry_on treats RuntimeError,")
    print("ValueError, TypeError, OSError, and several other common")
    print("exception types as NOT retryable by default (it assumes they are")
    print("programming errors, not transient failures) - only")
    print("ConnectionError, an httpx/requests HTTPError with a 5xx status,")
    print("or an unrecognised exception type retries by default. A first")
    print("draft of this scenario raised RuntimeError and retry_policy did")
    print("not retry at all. client.py has no such distinction: it branches")
    print("on HTTP status code, never on Python exception type.")

    print()
    print("--- Scenario 2: node always fails, max_attempts=3 ---")
    counter, exc = run_retries_exhausted()
    print(f"attempts made: {len(counter.calls)}")
    print(f"raised: {type(exc).__name__}: {exc}")
    print("OBSERVATION: exhausting retry_policy raises the node's own last")
    print("exception type (ConnectionError here), not a structured,")
    print("role-aware error like client.py's LadderExhaustedError (which")
    print("carries an Attempt per rung with a machine-readable category).")

    print()
    print("--- Scenario 3: express a model SWITCH using only retry_policy ---")
    models_tried, result = run_model_switch_attempt()
    print(f"models tried (tracked by the node itself): {models_tried}")
    print(f"final state: {result}")
    print("OBSERVATION: switching models per attempt required the node to")
    print("hold its own ladder and its own attempt counter - retry_policy's")
    print("callback contract (a plain callable retried on exception) gives")
    print("the node nothing to key that switch off. This is the ladder")
    print("re-implemented inside the node body, which is exactly the code")
    print("client.py already has, just moved and with a worse error shape.")

    print()
    print("--- Scenario 4: THE REAL INTEGRATION POINT - retry_policy around")
    print("    a node that already exhausted its own ladder ---")
    node_calls, final_error = run_retry_policy_around_an_already_exhausted_ladder()
    print(f"node invocations: {len(node_calls)} (max_attempts=3)")
    print(f"raised: {type(final_error).__name__}: {final_error}")
    print("OBSERVATION, PROVEN not assumed: langgraph's default_retry_on has")
    print("no special case for LadderExhaustedError - it is not")
    print("ConnectionError, not an httpx/requests HTTPError, and not in the")
    print("excluded list (ValueError/TypeError/RuntimeError/...), so it falls")
    print("through to the catch-all 'return True'. Wrapping a vision_node-")
    print("shaped node in retry_policy(max_attempts=3) RE-RUNS THE ENTIRE")
    print("ALREADY-EXHAUSTED LADDER 3 TIMES. A real vision_node's ladder call")
    print("already spends its full role budget (ROLE_TIMEOUTS['eyes'] = 30s)")
    print("trying every rung before raising - retry_policy would silently")
    print("multiply that up to 3x (~90s) for a call that was already known to")
    print("fail every rung on the first attempt, with zero benefit: no rung")
    print("gets a fresh chance to succeed that it did not already get inside")
    print("the ladder itself.")

    print()
    print("=" * 72)
    print("COMPARISON")
    print("=" * 72)
    print(
        "retry_policy:  retries the SAME callable on the SAME input, with a\n"
        "               fixed backoff/jitter schedule and a fixed max_attempts.\n"
        "               No per-attempt parameterisation is passed to the node;\n"
        "               a node wanting different behaviour per attempt (e.g. a\n"
        "               different model) must track that itself.\n"
        "               No status-code awareness: retry_on filters by exception\n"
        "               type/predicate only, so distinguishing 'retry this\n"
        "               (429/5xx)' from 'do not retry this (401/402/403/413,\n"
        "               a caller bug)' needs the node to raise different\n"
        "               exception types and pass a matching retry_on - client.py\n"
        "               does this today via HTTP status branching, not exceptions.\n"
        "               Exhaustion surfaces as the node's own last exception, not\n"
        "               a structured, per-rung report.\n"
        "\n"
        "model ladder:  a ROLE (eyes/brain) walks an ORDERED LIST OF DIFFERENT\n"
        "               MODELS inside ONE role budget (ROLE_TIMEOUTS), skipping\n"
        "               terminal failures (401/402/403/413) immediately instead\n"
        "               of retrying them, and returning a structured\n"
        "               LadderExhaustedError with one Attempt (model, category,\n"
        "               status_code, detail) per rung tried - built for\n"
        "               failure_messages.py to turn into a spoken message\n"
        "               without parsing prose."
    )

    print()
    print("VERDICT: KEEP OURS")
    print(
        "DECISIVE REASON: retry_policy retries one callable against one\n"
        "input; the ladder's entire purpose is trying DIFFERENT models. Scenario 3\n"
        "shows that expressing the ladder's real behaviour with retry_policy\n"
        "means re-implementing the ladder's model list and attempt tracking\n"
        "inside the node anyway, while losing client.py's terminal-status\n"
        "short-circuit (never retry an auth/credit/payload failure) and its\n"
        "structured per-rung Attempt report that failure_messages.py depends\n"
        "on. Scenario 4 STRENGTHENS this: wrapping retry_policy around the\n"
        "REAL integration point (a node whose ladder has already exhausted\n"
        "itself) is not neutral, it is actively harmful - it multiplies an\n"
        "already-spent role time budget by max_attempts with no chance of a\n"
        "different outcome, silently, for a blind user already waiting on\n"
        "spoken feedback. retry_policy would be additive complexity AND a new\n"
        "latency hazard, not a replacement."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
