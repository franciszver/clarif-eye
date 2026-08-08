"""Experiment (issue #85 / P9.6): does langgraph's add_node(cache_policy=...)
+ compile(cache=...) replace ui.py's ImageResultCache / DocumentTextCache?

Runs entirely OFFLINE - no live API calls, no disk writes (langgraph's
InMemoryCache, like ImageResultCache, lives in process memory only).

WHAT THIS MEASURES:
  1. THE REALISTIC FAILURE SHAPE FOR THIS CODEBASE, LED WITH: every run_*
     function here (vision.run_vision, synth.run_fast_synth, ...) does not
     raise on a model failure - it returns a DEGRADED state update, the
     same way a normal success is returned (see each module's own
     docstring: "must NEVER let a raw exception escape into the graph").
     From cache_policy's point of view that is an ordinary successful node
     return, indistinguishable from a good answer. Does cache_policy admit
     it, and does the entry survive the underlying failure clearing?
  2. Does cache_policy skip re-running a node on a repeat call with the
     same input, the way ImageResultCache short-circuits a repeat photo?
  3. What is the cache KEY built from - is it content-keyed the way
     ImageResultCache hashes image bytes, or something else?
  4. A second, less likely failure shape: does cache_policy also admit a
     node run that RAISES an exception rather than returning normally?
     ImageResultCache's module docstring is explicit: "Only successful
     results are ever stored here... a quota/API failure must never be
     replayed to the next visitor as if it were that photo's own answer."

Usage:
    python scripts/experiment_cache_policy.py
    python scripts/experiment_cache_policy.py > scripts/experiments/cache_policy.txt
"""

import sys
from typing import TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.graph import StateGraph
from langgraph.types import CachePolicy


class State(TypedDict, total=False):
    x: int
    result: str


def build_graph(cache_policy, cache):
    calls = []

    def node(state):
        calls.append(state["x"])
        if state["x"] == 999:
            raise RuntimeError("simulated model-call failure")
        return {"result": f"computed-{state['x']}"}

    builder = StateGraph(State)
    builder.add_node("n", node, cache_policy=cache_policy)
    builder.set_entry_point("n")
    builder.set_finish_point("n")
    graph = builder.compile(cache=cache)
    return graph, calls


def scenario_degraded_result_is_cached_as_a_permanent_stale_answer():
    """THE REALISTIC SHAPE, not a hypothetical: a node built like every
    run_* function in this codebase, which degrades to a spoken-ready
    fallback UPDATE rather than raising. Submit the same input while a
    transient failure is active, then again after the failure clears -
    does the SAME input ever get a fresh answer once cache_policy has
    admitted the degraded one?
    """
    transient_failure_active = {"flag": True}
    calls = []

    def node_like_run_vision(state):
        calls.append(state["x"])
        if transient_failure_active["flag"]:
            # No exception - this is run_vision's real shape: an early
            # RETURN of a degraded, spoken-ready update.
            return {"result": "degraded: the model was unavailable, try again"}
        return {"result": f"computed-{state['x']}"}

    cache = InMemoryCache()
    builder = StateGraph(State)
    builder.add_node("n", node_like_run_vision, cache_policy=CachePolicy(ttl=None))
    builder.set_entry_point("n")
    builder.set_finish_point("n")
    graph = builder.compile(cache=cache)

    while_failing = graph.invoke({"x": 1})
    transient_failure_active["flag"] = False
    after_failure_cleared = graph.invoke({"x": 1})

    return calls, while_failing, after_failure_cleared


def scenario_repeat_input_is_served_from_cache():
    cache = InMemoryCache()
    graph, calls = build_graph(CachePolicy(ttl=None), cache)
    r1 = graph.invoke({"x": 1})
    r2 = graph.invoke({"x": 1})
    r3 = graph.invoke({"x": 2})
    return calls, [r1, r2, r3]


def scenario_failure_is_cached_and_silently_replayed():
    """The decisive scenario: submit an input that makes the node raise,
    twice. What does the SECOND call return?"""
    cache = InMemoryCache()
    graph, calls = build_graph(CachePolicy(ttl=None), cache)

    first_error = None
    try:
        graph.invoke({"x": 999})
    except Exception as exc:  # noqa: BLE001 - want the real exception
        first_error = exc

    second_error = None
    second_result = None
    try:
        second_result = graph.invoke({"x": 999})
    except Exception as exc:  # noqa: BLE001
        second_error = exc

    # Also check the streaming surface: does the replay even emit an
    # "updates" chunk, or is it completely silent downstream?
    cache2 = InMemoryCache()
    graph2, _ = build_graph(CachePolicy(ttl=None), cache2)
    try:
        graph2.invoke({"x": 999})
    except Exception:
        pass
    stream_chunks = list(graph2.stream({"x": 999}, stream_mode="updates"))

    return {
        "node_invocations": len(calls),
        "first_call_raised": f"{type(first_error).__name__}: {first_error}" if first_error else None,
        "second_call_raised": f"{type(second_error).__name__}: {second_error}" if second_error else None,
        "second_call_result": second_result,
        "stream_chunks_on_replay": stream_chunks,
    }


def main():
    print("python scripts/experiment_cache_policy.py")
    print("=" * 72)
    print("EXPERIMENT: cache_policy/compile(cache=...) vs ImageResultCache/")
    print("DocumentTextCache")
    print("=" * 72)

    print()
    print("--- Scenario 1: DECISIVE, THE REALISTIC SHAPE - a degraded result")
    print("    (no exception, exactly how run_* functions here fail) gets")
    print("    cached as a permanent stale answer ---")
    calls, while_failing, after_failure_cleared = (
        scenario_degraded_result_is_cached_as_a_permanent_stale_answer()
    )
    print(f"node invocations: {len(calls)} (calls={calls})")
    print(f"result while the transient failure was active: {while_failing}")
    print(f"result AFTER the transient failure cleared, same input: {after_failure_cleared}")
    print("OBSERVATION: node_invocations == 1 - the node ran exactly once,")
    print("while the failure was active, and returned a degraded update the")
    print("same way vision.run_vision returns _degraded(...) - no exception,")
    print("a normal return. cache_policy has no way to see that this return")
    print("was a fallback rather than a real answer, so it caches it. Once")
    print("the transient condition clears, the SAME photo/input still gets")
    print("the OLD degraded answer forever - the node is never invoked again")
    print("for that input. This is the realistic consequence for this")
    print("codebase: not silence, a STALE SPOKEN ANSWER frozen per input,")
    print("read back to a blind user as if it were current. This is exactly")
    print("the product invariant ImageResultCache's module docstring states")
    print("directly: 'Only successful results are ever stored here... a")
    print("quota/API failure must never be replayed to the next visitor as")
    print("if it were that photo's own answer.'")

    print()
    print("--- Scenario 2: repeat input is served from cache, not re-run ---")
    calls, results = scenario_repeat_input_is_served_from_cache()
    print(f"node invocations for [x=1, x=1, x=2]: {len(calls)} (calls={calls})")
    print(f"results: {results}")
    print("OBSERVATION: the second x=1 call did not re-invoke the node - a")
    print("cache hit, same shape as ImageResultCache short-circuiting a")
    print("repeat photo. The key is the pickled node INPUT by default")
    print("(langgraph.types.default_cache_key), which is content-keyed the")
    print("same way ImageResultCache hashes image bytes - a reasonable match")
    print("here.")

    print()
    print("--- Scenario 3: secondary failure shape - a RAISED exception is")
    print("    also cached, and its replay is silent rather than stale ---")
    outcome = scenario_failure_is_cached_and_silently_replayed()
    for key, value in outcome.items():
        print(f"{key}: {value}")
    print()
    print("OBSERVATION: node_invocations == 1 means the node was called")
    print("exactly ONCE across two graph.invoke({'x': 999}) calls that both")
    print("should fail. The FIRST call raised RuntimeError, as expected. The")
    print("SECOND call, hitting the cache, did NOT raise, did NOT re-run the")
    print("node, and returned the state with no 'result' key added - not an")
    print("error, not a value, nothing indicating anything went wrong. The")
    print("streaming surface confirms this is not just an invoke() quirk:")
    print("stream_mode='updates' on the replay produced ZERO chunks - total")
    print("silence downstream. This SILENCE shape is real but LESS likely to")
    print("occur in this codebase than Scenario 1's STALE ANSWER shape,")
    print("because every run_* function here is written to return a")
    print("degraded update rather than raise - raising past the node")
    print("boundary is the exception, not the norm, in this app's own code.")

    print()
    print("=" * 72)
    print("COMPARISON")
    print("=" * 72)
    print(
        "cache_policy:  content-keyed via pickling the node's input (matches\n"
        "               ImageResultCache's content-hash approach) but admits\n"
        "               ANY normal return value into the cache with no opt-out,\n"
        "               including a degraded fallback update that carries no\n"
        "               signal distinguishing it from a real answer - Scenario 1\n"
        "               shows that entry then survives forever, read back as a\n"
        "               STALE spoken answer even once the underlying failure is\n"
        "               gone. A node that raises instead (Scenario 3, the less\n"
        "               likely shape in this codebase) is also cached, and its\n"
        "               replay is SILENT (no exception, no result, no stream\n"
        "               chunk) rather than stale. There is no ttl-based\n"
        "               invalidation tied to anything except wall-clock time\n"
        "               (CachePolicy.ttl), and no equivalent of a stale-file\n"
        "               guard (this app's audio cache entries can outlive their\n"
        "               mp3 on disk; a bare InMemoryCache has no notion of an\n"
        "               external resource going stale at all).\n"
        "\n"
        "our caches:    ImageResultCache/DocumentTextCache (ui.py) only ever\n"
        "               .put() on a verified SUCCESS path (see handle_submit) -\n"
        "               a quota/API failure, and a degraded fallback answer, are\n"
        "               never written to either cache, so neither can ever be\n"
        "               replayed to a later request. Bounded LRU eviction\n"
        "               (IMAGE_CACHE_MAX_ENTRIES/DOCUMENT_CACHE_MAX_ENTRIES),\n"
        "               never persisted to disk, and ImageResultCache carries a\n"
        "               stale-file guard so a cache hit whose mp3 was pruned\n"
        "               from disk is treated as a miss, not a lying hit."
    )

    print()
    print("VERDICT: KEEP OURS")
    print(
        "DECISIVE REASON: Scenario 1 is a product-safety regression, not a\n"
        "style difference, and it is the REALISTIC one for this codebase:\n"
        "cache_policy cannot tell a degraded fallback from a real answer, so it\n"
        "caches the fallback and freezes it as that input's permanent answer -\n"
        "a blind user who submits the same photo again after a transient\n"
        "failure clears keeps hearing the old degraded message, spoken as if it\n"
        "were current. ImageResultCache's entire reason for existing (issue\n"
        "#75) is the opposite guarantee: only a verified success is ever\n"
        "stored. Adopting cache_policy as-is would need a custom cache_policy\n"
        "or a wrapper that inspects outcomes before admitting them - at which\n"
        "point it is no longer buying anything over the explicit, already-\n"
        "correct hand-rolled cache."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
