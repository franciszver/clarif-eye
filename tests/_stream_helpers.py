"""Shared test-only helpers for a compiled graph's stream (issue #80 /
P9.1): draining a real one, and standing in for one.

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

    NOTE FOR A FUTURE CALLER: this does NOT de-duplicate a namespaced
    "__interrupt__" chunk the way _narrate_stream does. A pause raised inside
    the child arrives twice - once namespaced, once at the parent level - so
    driving a PAUSING run through here would put "__interrupt__" in the trace
    twice. Harmless today (no caller of this helper streams a run that
    pauses), and left rather than fixed pre-emptively, but a test that starts
    doing so should either count it once or assert on the doubling
    deliberately.
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


class SingleChunkStreamMixin:
    """Gives a graph DOUBLE the .stream() signature and chunk shape
    clarif_eye.ui._narrate_stream actually consumes, over the double's own
    invoke().

    ONE COPY, ON PURPOSE (issue #84 / P9.5's simplify gate): five doubles
    across tests/test_ui.py and tests/test_accessibility.py each carried an
    identical three-line stream() - five places for the shape to drift out
    of step with production the next time it changes. It changed once
    already: streaming became the only path in (issue #80 / P9.1), and then
    the deep path became a child graph (#84 / P9.5), which made
    _narrate_stream open the stream with subgraphs=True and receive
    (namespace, chunk) PAIRS instead of bare chunks.

    ONE CHUNK, KEYED "tts", is the minimal shape that satisfies the
    consumer: a real graph's LAST chunk is always tts's, and
    clarif_eye.graph.next_node_after("tts", ...) is None, so it maps to no
    narration phrase and every staged-contract test keeps its exact yield
    sequence. The double's own invoke() still runs, so whatever it records
    or raises still happens.

    NAMESPACE IS ALWAYS () - the parent's. These doubles have no subgraph of
    their own, and a double that pretended to would be asserting something
    about LangGraph rather than about this app.
    """

    def stream(self, state, config=None, stream_mode="updates", subgraphs=False):
        chunk = {"tts": self.invoke(state, config=config)}
        yield ((), chunk) if subgraphs else chunk
