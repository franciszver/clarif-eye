"""Shared test-only helper for draining a compiled graph's stream (issue
#80 / P9.1).

Not a conftest.py fixture: every call site needs a different `state`/
`config`, so a plain importable function is a better fit than a fixture
that would only ever hand back the function itself. Every test file that
used to thread a config["configurable"]["trace"] list through
graph.invoke() to observe which nodes ran now calls this instead.
"""


def drain_stream_collecting_trace(graph, state, config):
    """Run a compiled graph via stream(..., stream_mode="updates") and
    return (final_state, visited_node_names_in_order).

    Each stream chunk is keyed by the node that just completed and holds
    that node's state update, so merging chunks into `result` and
    collecting their keys in arrival order is a drop-in replacement for
    what graph._record used to append to a caller-supplied trace list
    (removed by issue #80 / P9.1 - see clarif_eye.graph's module
    docstring).

    A None `update` is skipped rather than merged (issue #82 / P9.3): a node
    that returns a bare Command(goto=...) with no state update - the "entry"
    node does exactly this - streams as {"entry": None}, and dict.update(None)
    would raise. The node still counts as VISITED, so it stays in the trace.
    """
    result = dict(state)
    trace = []
    for chunk in graph.stream(state, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
            if update is not None:
                result.update(update)
            trace.append(node_name)
    return result, trace
