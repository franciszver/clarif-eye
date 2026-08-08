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

    SUBGRAPHS ARE INCLUDED (issue #84 / P9.5), with their namespace dropped
    from the trace. The deep path is a child graph now
    (clarif_eye.deep_path), so without `subgraphs=True` `research` and
    `analysis` would vanish from every trace in this suite and the whole
    deep path would read as one opaque "deep_path" step - which would hide
    exactly the routing these tests exist to assert on. Streamed this way, a
    deep run traces as entry, vision, research, analysis, deep_path, tts:
    the child's nodes in the order they ran, then the parent's node that
    contains them completing. This is the same stream shape
    clarif_eye.ui._narrate_stream consumes in production.
    """
    result = dict(state)
    trace = []
    for _namespace, chunk in graph.stream(
        state, config=config, stream_mode="updates", subgraphs=True
    ):
        for node_name, update in chunk.items():
            if update is not None:
                result.update(update)
            trace.append(node_name)
    return result, trace
