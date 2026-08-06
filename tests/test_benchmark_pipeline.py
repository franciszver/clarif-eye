"""Offline test for scripts/benchmark_pipeline.py's stage-timing math
(issue P6.1 / #17).

graph._record(config, name) stamps a node's timestamp at ENTRY (see
graph.py), not at exit. Attributing duration as "gap since the previous
timestamp" therefore charges each node's wait/duration to the node that
runs AFTER it - vision reads as ~0s while its real time lands on research,
research's lands on analysis, and so on, and the last node (tts) is never
measured at all since nothing is recorded after it starts.

This test is purely offline: it builds a synthetic trace by hand and calls
_stage_durations directly. It does NOT build or invoke the graph.
"""

from scripts.benchmark_pipeline import _stage_durations


def test_stage_durations_attributes_time_to_the_node_that_spent_it():
    """Nodes recorded at t=0,1,4,10 with run end=12 must be attributed
    1,3,6,2 (next-timestamp minus own timestamp, last node vs. `end`) -
    NOT 0,1,3,6 (previous-timestamp math, the off-by-one bug)."""
    trace = [
        ("vision", 0),
        ("research", 1),
        ("analysis", 4),
        ("tts", 10),
    ]
    end = 12

    durations = _stage_durations(trace, end)

    print(f"computed durations: {durations}")

    assert durations == {
        "vision": 1,
        "research": 3,
        "analysis": 6,
        "tts": 2,
    }
    # The bug this guards against, spelled out explicitly:
    assert durations != {
        "vision": 0,
        "research": 1,
        "analysis": 3,
        "tts": 6,
    }
