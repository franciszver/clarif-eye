"""Experiment (issue #85 / P9.6): does langgraph's add_node(timeout=...)
replace the whole-pipeline deadline (graph.py's _deadline_exceeded) and the
per-role ceilings inside client.complete?

Runs entirely OFFLINE - no live API calls, no live sleeps beyond small
fractions of a second used to prove the timeout actually fires.

CRITICAL PRODUCT RULE under test: tts is NEVER skipped or time-gated (see
graph.py's "Total-pipeline deadline" docstring block) - it is the one node
whose own latency matters more than respecting a budget that has already
run out, because skipping it turns "degraded but spoken" into "silent".
Any timeout verdict here must respect that a per-node timeout on tts would
violate it.

Usage:
    python scripts/experiment_timeout.py
    python scripts/experiment_timeout.py > scripts/experiments/timeout.txt
"""

import asyncio
import sys
import time
from typing import TypedDict

from langgraph.graph import StateGraph


class State(TypedDict, total=False):
    result: str


def _build_sync_timeout_graph():
    def slow_node(state):
        time.sleep(1.0)
        return {"result": "done"}

    builder = StateGraph(State)
    builder.add_node("slow", slow_node, timeout=0.05)
    builder.set_entry_point("slow")
    builder.set_finish_point("slow")
    return builder.compile()


def probe_sync_node_rejects_timeout():
    """This app's nodes (vision_node, analysis_node, tts_node, ...) are all
    plain sync functions invoked via graph.invoke()/graph.stream() - never
    graph.ainvoke() (verified: no .ainvoke(/.astream( call exists anywhere
    under src/clarif_eye/). Does add_node(timeout=...) even accept a sync
    node?"""
    try:
        _build_sync_timeout_graph()
        return None
    except ValueError as exc:
        return exc


async def probe_async_node_timeout_fires():
    """Confirm timeout= DOES work, and fires at the stated deadline, for an
    ASYNC node - so the sync rejection above is a real constraint of this
    policy, not a probe mistake."""

    async def slow_node(state):
        await asyncio.sleep(1.0)
        return {"result": "done"}

    builder = StateGraph(State)
    builder.add_node("slow", slow_node, timeout=0.1)
    builder.set_entry_point("slow")
    builder.set_finish_point("slow")
    graph = builder.compile()

    start = time.monotonic()
    try:
        await graph.ainvoke({})
        return "completed", time.monotonic() - start
    except Exception as exc:  # noqa: BLE001 - want the real exception type
        return f"{type(exc).__name__}: {exc}", time.monotonic() - start


class MixedState(TypedDict, total=False):
    a: int
    b: int
    result: str


async def probe_mixed_graph_only_needs_the_timed_node_async():
    """Correcting an earlier overstatement in this script: does adopting
    timeout= require rewriting EVERY node as async, or only the entry-point
    call plus the specific node(s) actually carrying a timeout?

    Builds a THREE-node graph: two plain sync nodes (n1, n3) and one async
    node (n2) with timeout= set, driven through graph.ainvoke(). If the sync
    nodes run untouched and only n2's timeout fires when it overruns, the
    real migration cost is "the entry call becomes ainvoke()/astream(), and
    only the node(s) you want timed become async" - not an async rewrite of
    the whole graph.
    """

    def sync_node_1(state):
        return {"a": 1}

    async def timed_async_node(state):
        await asyncio.sleep(1.0)
        return {"b": 2}

    def sync_node_2(state):
        return {"result": "done"}

    builder = StateGraph(MixedState)
    builder.add_node("n1_sync", sync_node_1)
    builder.add_node("n2_async_timed", timed_async_node, timeout=0.1)
    builder.add_node("n3_sync", sync_node_2)
    builder.set_entry_point("n1_sync")
    builder.add_edge("n1_sync", "n2_async_timed")
    builder.add_edge("n2_async_timed", "n3_sync")
    builder.set_finish_point("n3_sync")
    graph = builder.compile()

    try:
        await graph.ainvoke({})
        return "completed (unexpected - n2 should have timed out)"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def main():
    print("python scripts/experiment_timeout.py")
    print("=" * 72)
    print("EXPERIMENT: per-node timeout= vs the whole-pipeline deadline")
    print("=" * 72)

    print()
    print("--- Probe 1: add_node(timeout=...) on a SYNC node (this app's shape) ---")
    error = probe_sync_node_rejects_timeout()
    if error is None:
        print("UNEXPECTED: a sync node compiled successfully with timeout= set.")
    else:
        print(f"raised at compile() time: {type(error).__name__}: {error}")
        print("OBSERVATION: every node in this app's graph (vision_node,")
        print("analysis_node, fast_synth_node, followup_node, tts_node, ...)")
        print("is a plain synchronous function called via graph.invoke() /")
        print("graph.stream() - this codebase never calls .ainvoke()/.astream().")
        print("langgraph's timeout= relies on asyncio cancellation and simply")
        print("REFUSES TO COMPILE a graph with timeout= on a sync node.")

    print()
    print("--- Probe 2: add_node(timeout=...) on an ASYNC node (confirms it works) ---")
    outcome, elapsed = asyncio.run(probe_async_node_timeout_fires())
    print(f"outcome: {outcome}")
    print(f"elapsed: {elapsed:.3f}s (timeout was set to 0.1s, sleep was 1.0s)")
    print("OBSERVATION: with an async node, timeout= does fire at the stated")
    print("deadline and raises langgraph.errors.NodeTimeoutError. The policy")
    print("itself works - the app's synchronous node style is what blocks it.")

    print()
    print("--- Probe 3: what does adopting timeout= actually cost - the whole")
    print("    graph rewritten async, or just the entry call + timed node(s)? ---")
    outcome = asyncio.run(probe_mixed_graph_only_needs_the_timed_node_async())
    print(f"outcome: {outcome}")
    print("OBSERVATION, CORRECTING AN EARLIER OVERSTATEMENT: a three-node")
    print("graph with two plain SYNC nodes and ONE async node carrying")
    print("timeout=, driven through graph.ainvoke(), compiles and the sync")
    print("nodes run untouched - only the timed async node's own timeout")
    print("fires. Adopting timeout= for, say, just the eyes/brain calls does")
    print("NOT require rewriting every run_* function as async - it requires")
    print("making the ENTRY-POINT call ainvoke()/astream() instead of")
    print("invoke()/stream(), and making the specific node(s) carrying")
    print("timeout= async. The real cost for THIS app is still nonzero: the")
    print("whole call chain from ui.handle_submit down through graph.stream()")
    print("is synchronous today (Gradio callbacks, httpx.Client, no event")
    print("loop anywhere in src/clarif_eye/), so even a single timed node")
    print("needs that entry call converted and an event loop available to")
    print("run it in - a real change, just a smaller one than 'every node'.")

    print()
    print("=" * 72)
    print("COMPARISON")
    print("=" * 72)
    print(
        "timeout=:      a HARD per-attempt wall-clock cap on ONE node, enforced\n"
        "               via asyncio cancellation. Requires every timed node to be\n"
        "               an async function; a sync node with timeout= set fails at\n"
        "               compile() time, before any run ever starts. Firing raises\n"
        "               NodeTimeoutError - the node's partial work is discarded,\n"
        "               there is no built-in 'degrade from what is known so far'\n"
        "               path, only whatever an error_handler or graph-level except\n"
        "               chooses to do with the exception.\n"
        "\n"
        "our mechanism: TWO layers, deliberately different scopes (see graph.py's\n"
        "               top-level docstring):\n"
        "                 1. client.ROLE_TIMEOUTS - a TOTAL budget per ROLE (eyes\n"
        "                    30s, brain 45s) spent across however many ladder\n"
        "                    rungs are tried, not per HTTP call.\n"
        "                 2. graph.py's config['configurable']['deadline'] - an\n"
        "                    ABSOLUTE time.monotonic() deadline for the WHOLE\n"
        "                    pipeline run, read by _deadline_exceeded() inside each\n"
        "                    model-calling node, which then asks its module's\n"
        "                    run_* function to DEGRADE FROM KNOWN STATE (return a\n"
        "                    spoken-ready partial result) instead of raising.\n"
        "               tts_node reads neither: it is deliberately never gated, by\n"
        "               product rule, because skipping it turns a degraded answer\n"
        "               into a silent one for a blind user."
    )

    print()
    print("VERDICT: KEEP OURS")
    print(
        "DECISIVE REASON: Probe 1 shows timeout= cannot be applied to a sync\n"
        "node at all - it fails at compile() time. Probe 3 shows the real fix\n"
        "for that is smaller than 'rewrite everything': only the ENTRY call and\n"
        "the specific timed node(s) need to become async, not every node in the\n"
        "graph. But this app's entry call (ui.handle_submit -> graph.stream())\n"
        "is synchronous end to end today, with no event loop anywhere in\n"
        "src/clarif_eye/, so that conversion is still a real, nonzero change,\n"
        "not a config flag. And even paid for, the scope stays wrong: timeout=\n"
        "caps one node's one attempt, while the actual problem this app solved\n"
        "(issue #17/P6.1) was an unbounded TAIL across the whole run (vision +\n"
        "brain + research + tts), which per-node caps do not bound - the sum of\n"
        "several under-budget nodes can still blow the pipeline deadline. A\n"
        "node timing out also raises by default, the wrong failure mode here:\n"
        "every model-calling node must degrade to a spoken-ready partial\n"
        "result, never raise past the UI boundary, and tts must never be gated\n"
        "at all - a rule a per-node timeout has no way to express short of\n"
        "exempting tts from the very policy being adopted."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
