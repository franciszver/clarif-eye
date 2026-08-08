# Node policies: adopt or keep

LangGraph's `add_node` accepts four built-in policies that overlap with
mechanisms this app already had before LangGraph was introduced:
`retry_policy`, `timeout`, `cache_policy` (plus `compile(cache=...)`), and
`error_handler`. Each one was tested against its hand-rolled counterpart
with a script under `scripts/`, run offline against a controllable fake
node, with the output committed under `scripts/experiments/`. The verdicts
below come from those runs, not from reading the documentation.

| Policy | What it does | Our equivalent | Verdict | Decisive reason |
|---|---|---|---|---|
| `retry_policy` | Retries one node's callable against the same input, on a fixed backoff schedule, up to `max_attempts` times. | The model ladder in `client.py`: for a given role, tries an ordered list of *different* models inside one shared time budget, skipping auth/credit/payload failures instead of retrying them, and reporting a structured per-model result. | Keep ours | `retry_policy` retries one call; the ladder's whole point is trying different models. Expressing the ladder with `retry_policy` means the node has to track its own model list and attempt count anyway, while losing the short-circuit on unrecoverable failures and the structured per-attempt report that builds the spoken error message. Tested at the real integration point too: wrapping `retry_policy` around a node whose own ladder call has already exhausted every rung re-runs that entire already-exhausted ladder up to `max_attempts` times, silently multiplying an already-spent time budget with no chance of a different outcome. |
| `timeout` | A hard wall-clock cap on one node's one attempt, enforced through `asyncio` cancellation. | A whole-run deadline set once per request and checked inside each model-calling node, plus a separate time budget per model role. A node past the deadline returns its best available partial answer instead of failing. | Keep ours | `timeout` only works on an `async` node; every node in this app is a plain synchronous function called through a synchronous entry point, so a compiled graph using `timeout` on any of them fails before a single request runs. Only the entry call and the specific timed node(s) would need to become async, not the whole graph, but this app's entry point runs synchronously end to end with no event loop in place, so that is still a real change, not a flag flip. Even paid for, the scope stays wrong: it caps one node's one attempt, not the whole run, so a few nodes finishing just under their cap can still blow the total time a listener is waiting. Spoken output must never be skipped once a photo has been read, and a policy scoped to a single node cannot express "every node but this one." |
| `cache_policy` / `compile(cache=...)` | Skips re-running a node when it sees the same input again, keyed by a hash of that input, kept in memory. | An in-memory cache from photo content to its result, and a second one for submitted document text. Both only ever store a result once the call actually succeeded. | Keep ours | Every function in this app that calls a model reports a failure by returning a fallback answer, not by raising, so `cache_policy` cannot tell that answer apart from a real one: it caches the fallback and keeps serving it for that same photo forever, even after the failure that caused it is gone, read aloud as if it were current. A node that raises instead is also cached, replaying as complete silence, but that shape is the less likely one here, since raising past a node is the exception in this codebase, not the norm. Our caches store a result only after a real success, so a fallback answer or an outright failure is never replayed to the next person who submits the same photo. |
| `error_handler` | Runs when a node raises, and can return a state update in place of letting the exception propagate. | Every function that calls a model catches its own failures, including replies that come back malformed or empty without raising at all, and returns a spoken-ready fallback answer. | Keep ours | `error_handler` only sees exceptions a node actually raises. A large share of what these functions already handle, an empty reply, one that doesn't parse, is not an exception at all, just an early return with a fallback message. Adopting `error_handler` would mean rewriting that working code to raise instead of return, for a mechanism that ends up covering less ground than what is already there. |

## What would change a verdict

Two of the findings above are specific to how this app runs today, not to
LangGraph in general:

- The `timeout` verdict would need revisiting if the app's entry point
  moved from `graph.invoke()`/`graph.stream()` to their async
  counterparts; only that entry call and the specific node(s) carrying a
  timeout would need to change, not every node, but no part of this app
  runs on an event loop today.
- The `cache_policy` finding about caching a fallback answer is the
  default behavior; a custom cache key function or a wrapper that
  inspects the outcome before writing to the cache could close that gap,
  but at that point it is doing the same work our existing cache already
  does.

See `scripts/experiment_retry_policy.py`, `scripts/experiment_timeout.py`,
`scripts/experiment_cache_policy.py`, and `scripts/experiment_error_handler.py`
for the runnable experiments, and `scripts/experiments/*.txt` for their
recorded output.
