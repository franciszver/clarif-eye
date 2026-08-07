"""Offline test for scripts/benchmark_pipeline.py's stage-timing math
(issue P6.1 / #17; migrated to completion-based timing by issue #80 / P9.1).

graph.stream(..., stream_mode="updates") yields one chunk per node the
MOMENT it completes, never when it started (see clarif_eye.graph's module
docstring) - so this script stamps a node's timestamp at COMPLETION, the
opposite of the old entry-based trace it replaced. Attributing duration as
"gap since the previous completion" (or `start`, for the first node) is
therefore the natural accounting: each node's own duration lands on itself,
and the last node's duration comes for free from its own completion
timestamp, with no separate `end` value needed to measure it.

This test is purely offline: it builds a synthetic completions list by hand
and calls _stage_durations directly. It does NOT build or invoke the graph.
"""

from scripts.benchmark_pipeline import _stage_durations


def test_stage_durations_attributes_time_to_the_node_that_spent_it():
    """Nodes completed at t=1,4,10,12 with start=0 must be attributed
    1,3,6,2 (own completion minus previous completion, first node vs.
    `start`) - each node's own duration, not the off-by-one entry-based
    math this replaced."""
    completions = [
        ("vision", 1),
        ("research", 4),
        ("analysis", 10),
        ("tts", 12),
    ]
    start = 0

    durations = _stage_durations(completions, start)

    print(f"computed durations: {durations}")

    assert durations == {
        "vision": 1,
        "research": 3,
        "analysis": 6,
        "tts": 2,
    }
