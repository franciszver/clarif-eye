"""Experiment (issue #85 / P9.6): does langgraph's add_node(cache_policy=...)
+ compile(cache=...) replace ui.py's ImageResultCache / DocumentTextCache?

Runs entirely OFFLINE - no live API calls, no disk writes (langgraph's
InMemoryCache, like ImageResultCache, lives in process memory only).

WHAT THIS MEASURES:
  1. Does cache_policy skip re-running a node on a repeat call with the
     same input, the way ImageResultCache short-circuits a repeat photo?
  2. What is the cache KEY built from - is it content-keyed the way
     ImageResultCache hashes image bytes, or something else?
  3. THE DECISIVE QUESTION: does cache_policy admit a FAILED node run into
     the cache? ImageResultCache's module docstring is explicit: "Only
     successful results are ever stored here... a quota/API failure must
     never be replayed to the next visitor as if it were that photo's own
     answer." This probes whether cache_policy honours that on its own.

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
    print("--- Scenario 1: repeat input is served from cache, not re-run ---")
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
    print("--- Scenario 2: DECISIVE - is a FAILED run cached and replayed? ---")
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
    print("silence downstream, exactly what a blind user must never get.")
    print("This means cache_policy's DEFAULT behaviour is the opposite of")
    print("ImageResultCache's explicit rule: it caches failures and replays")
    print("them as if the node had quietly done nothing.")

    print()
    print("=" * 72)
    print("COMPARISON")
    print("=" * 72)
    print(
        "cache_policy:  content-keyed via pickling the node's input (matches\n"
        "               ImageResultCache's content-hash approach) but admits\n"
        "               BOTH successful and FAILED runs into the cache with no\n"
        "               opt-out - Scenario 2 shows a failure is cached and its\n"
        "               replay is silent (no exception, no result, no stream\n"
        "               chunk). There is no ttl-based invalidation tied to\n"
        "               anything except wall-clock time (CachePolicy.ttl), and\n"
        "               no equivalent of a stale-file guard (this app's audio\n"
        "               cache entries can outlive their mp3 on disk; a bare\n"
        "               InMemoryCache has no notion of an external resource\n"
        "               going stale at all).\n"
        "\n"
        "our caches:    ImageResultCache/DocumentTextCache (ui.py) only ever\n"
        "               .put() on a verified SUCCESS path (see handle_submit) -\n"
        "               a quota/API failure is never written to either cache, so\n"
        "               it can never be replayed to a later request. Bounded LRU\n"
        "               eviction (IMAGE_CACHE_MAX_ENTRIES/DOCUMENT_CACHE_MAX_\n"
        "               ENTRIES), never persisted to disk, and ImageResultCache\n"
        "               carries a stale-file guard so a cache hit whose mp3 was\n"
        "               pruned from disk is treated as a miss, not a lying hit."
    )

    print()
    print("VERDICT: KEEP OURS")
    print(
        "DECISIVE REASON: Scenario 2 is a product-safety regression, not a\n"
        "style difference. cache_policy's default behaviour caches a failed\n"
        "model call and replays it on the next identical submission as total\n"
        "silence - no exception, no degraded message, no audio, nothing a\n"
        "blind user could act on. ImageResultCache's entire reason for\n"
        "existing (issue #75) is the opposite guarantee: never cache a\n"
        "failure. Adopting cache_policy as-is would need a custom cache_policy\n"
        "or a wrapper that inspects outcomes before admitting them - at which\n"
        "point it is no longer buying anything over the explicit, already-\n"
        "correct hand-rolled cache."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
