# Real-stack scenarios

Every mock-based test in this project stands in for one of seven external
systems. A mock encodes an assumption about how that system behaves. This
document, and the registry in `tests/test_scenario_pairing.py`, pair each
mock with a scenario that checks the assumption against the real system.

`tests/test_scenario_pairing.py` enforces the pairing automatically: it
scans `tests/*.py` for the mocking patterns this project uses
(`httpx.MockTransport(...)` and `FakeXxx`/`_FakeXxx` classes) and fails if
any such test module is not registered against a scenario. Read that file
for the registry itself.

## Manual only, not CI

Every scenario below makes a real network call, a real model inference
call, or both. None of them run in this pytest suite, and none of them run
in CI. CI ([#21](https://github.com/franciszver/clarif-eye/issues/21)) is
offline by design, and model inference must never run there. A scenario "existing" means the file is present and the pairing
is checked; it does not mean the scenario was run today, and it does not
mean the live system currently behaves as the scenario expects.

## What each kind of scenario actually proves

- **Recorder** (e.g. `scripts/record_vision_fixture.py`): makes one real
  call, writes the raw reply to `tests/fixtures/`, and asserts nothing. On
  its own it proves only that the call succeeded once.
- **Fixture replay** (e.g. `tests/test_vision_fixture_replay.py`): runs a
  recorded raw reply through the real production parsing/sanitisation code
  on every test run, offline, and asserts on the result. This is real
  evidence, checked automatically, without a live call.
- **Smoke script** (e.g. `scripts/live_smoke.py`, `scripts/tts_smoke.py`):
  makes a real call and asserts pass/fail on the spot, exiting non-zero on
  failure. Proves the system works right now, for whoever ran it.
- **Benchmark** (e.g. `scripts/benchmark_ladders.py`): makes many real
  calls and reports numbers. Not an assertion by itself, though
  `benchmark_ladders.py` does exit non-zero if any model in a ladder never
  succeeded across its runs.

A recorder is not a check. Only a fixture replay, a smoke script, or a
benchmark's own exit code checks anything.

## The seven systems

### 1. OpenRouter chat completions

Client: `clarif_eye/client.py`. Mocked with `httpx.MockTransport` in
`tests/test_client.py`, and with fake HTTP objects in
`tests/test_benchmark_script.py`, `tests/test_graph.py`, and
`tests/test_pipeline_deadline.py`.

Real-stack scenario:
- `scripts/live_smoke.py`: one real completion request against the eyes
  ladder. Exits 1 on failure. Run with
  `OPENROUTER_API_KEY=... python scripts/live_smoke.py`.
- `scripts/benchmark_ladders.py`: every rung of both ladders, `--runs`
  times each. Exits 1 if any rung never succeeded across its runs. Run
  with `OPENROUTER_API_KEY=... python scripts/benchmark_ladders.py --role all --runs 2 --image path/to/photo.jpg`.

Recorded evidence: none committed for this system directly (see vision and
analysis below for recorded replies through this client).

### 2. Vision node

`clarif_eye/vision.py`, the eyes role. Mocked with a fake client in
`tests/test_vision.py`, and replayed in `tests/test_vision_fixture_replay.py`.

Real-stack scenario:
- `scripts/record_vision_fixture.py` (recorder): one real vision call
  against a photo, writes the raw reply and parsed result to
  `tests/fixtures/`. Run with
  `OPENROUTER_API_KEY=... python scripts/record_vision_fixture.py path/to/photo.jpg`.
- `tests/test_vision_fixture_replay.py` (replay, runs on every test suite
  run): feeds each recorded raw reply through the real parser and checks
  ground-truth facts survive.

Recorded evidence: four fixture pairs are committed under
`tests/fixtures/` (`vision_reply_raw.txt` and three variants: a sentinel-
format document, a screenplay-style photo, and an adversarial
prompt-injection label).

### 3. Analysis node

`clarif_eye/analysis.py`, the brain role. Mocked with a fake client in
`tests/test_analysis.py`, and replayed in
`tests/test_analysis_fixture_replay.py`.

Real-stack scenario:
- `scripts/record_analysis_fixture.py` (recorder): one real analysis call
  against the recorded vision fixture's OCR text and scene, writes the raw
  reply to `tests/fixtures/`. Run with
  `OPENROUTER_API_KEY=... python scripts/record_analysis_fixture.py`.
- `tests/test_analysis_fixture_replay.py` (replay, runs on every test
  suite run): feeds the recorded raw reply through the real number-
  verification and sanitisation logic and checks ground-truth amounts and
  dates survive.

Recorded evidence: `tests/fixtures/analysis_reply_raw.txt` is committed.

### 4. Fast-synth node

`clarif_eye/synth.py`, the eyes role again, for the quick-description
path. Mocked with a fake client in `tests/test_synth.py`.

Real-stack scenario:
- `scripts/record_synth_fixture.py` (recorder, already existed before this
  change): one real fast-synth call against the recorded vision fixture,
  writes the raw reply to `tests/fixtures/`. Run with
  `OPENROUTER_API_KEY=... python scripts/record_synth_fixture.py`.
- `tests/test_synth_fixture_replay.py` (replay, added by this change):
  feeds the recorded raw reply through the real sanitisation logic. Skips
  cleanly, with a clear message, until the fixture is recorded.

Recorded evidence: **none yet.** No one has run
`scripts/record_synth_fixture.py` and committed its output, so the replay
test currently skips. This is the one honest gap in this pass: the
recorder and replay harness both exist, but no real model reply has been
checked through them yet. Closing it needs a real API call, which this
task could not make.

### 5. Research: search and page fetch

`clarif_eye/research.py`. Mocked with a fake searcher and
`httpx.MockTransport` in `tests/test_research.py`, and with a fake
searcher in `tests/test_tts.py`'s full-graph research-path test.

Real-stack scenario:
- `scripts/record_research_fixture.py` (recorder, already existed before
  this change): one real search plus, at most, one real page fetch for a
  query, writes the extracted text to `tests/fixtures/`. Run with
  `python scripts/record_research_fixture.py "some search query"`.
- `tests/test_research_fixture_replay.py` (added by this change): checks
  the shape of the recorded evidence when present (non-empty, no leaked
  `<script>`/`<style>` markup, within the production size cap). Skips
  cleanly until the fixture is recorded.

**Honest limitation:** unlike vision/analysis/synth, the recorder here
saves `run_research()`'s already-extracted output, not a raw page. There is
no raw HTML fixture to feed back through the extraction code, so the
replay test can only check the recorded evidence's shape, not re-run
production parsing against it the way the other three do. This is weaker
evidence by design, not by oversight.

Recorded evidence: **none yet**, same gap as fast-synth above.

### 6. TTS providers

`clarif_eye/tts.py`: edge-tts and gTTS. Mocked with a fake provider in
`tests/test_tts.py`, `tests/test_vision.py`, `tests/test_synth.py`, and
`tests/test_graph.py`.

Real-stack scenario:
- `scripts/tts_smoke.py`: synthesizes a short sentence through the real
  provider chain (or a single named provider with `--provider`), writes an
  mp3, and exits 1 if no audio came back. Run with
  `python scripts/tts_smoke.py` or
  `python scripts/tts_smoke.py --provider edge`.

Recorded evidence: none committed. `tts_smoke.py` is a live smoke script,
not a recorder; there is no offline audio-fixture replay in this project.

### 7. Gradio UI handler

`clarif_eye/ui.py`: `handle_submit` and `build_resources`. Mocked with a
fake graph and fake image in `tests/test_ui.py` and
`tests/test_accessibility.py`.

Real-stack scenario:
- `scripts/ui_smoke.py` (added by this change): builds the real resources
  (`build_resources()`) and calls the real handler (`handle_submit()`)
  with a real photo. No server is started; `handle_submit` is a plain
  function, exactly the property that lets `tests/test_ui.py` test it
  without Gradio running. Exits 1 if no description text comes back. Run
  with `OPENROUTER_API_KEY=... python scripts/ui_smoke.py path/to/photo.jpg`.

**Before this change, this was the one system with no real-stack scenario
at all.** `scripts/audit_accessibility.py` checks a running app's DOM and
ARIA attributes, which is a different concern (rendering, not the handler
logic), and it requires a server already running by hand; it is not a
stand-in for exercising `handle_submit` itself.
`scripts/benchmark_pipeline.py` exercises equivalent machinery (the
compiled graph, a real client, real TTS providers) but calls the graph
directly, not `handle_submit`, so it never touches the image-decoding or
error-handling logic that is `ui.py`'s own job. `scripts/ui_smoke.py`
closes that gap directly.

Recorded evidence: none; this is a live smoke script, not a recorder.

## Other real-stack scripts, not tied to one mock

Two scripts exercise the real stack across multiple systems at once and
are not the primary pairing for any single registry entry, but are real
evidence worth knowing about:

- `scripts/benchmark_pipeline.py`: runs the full compiled graph end to
  end, multiple times per configuration, and reports median/min/max
  latency plus an accuracy verdict from the production number-verifier.
  Not a pass/fail scenario by itself; it is a benchmark.
- `scripts/eval_injection.py`: runs vision then whichever of
  fast-synth/analysis the router picks, against a real photo, and checks
  whether an injected claim from photographed text reaches the final
  spoken output. Exits 1 if genuine content was suppressed or an attacker
  claim appears with no attribution anywhere in the output; some findings
  are reported as advisory and need a human to read the output and judge.

## Summary table

| System | Mocked in | Scenario | Recorded evidence checked automatically |
| --- | --- | --- | --- |
| OpenRouter client | test_client.py, test_benchmark_script.py, test_graph.py, test_pipeline_deadline.py | live_smoke.py, benchmark_ladders.py | no |
| Vision node | test_vision.py, test_vision_fixture_replay.py | record_vision_fixture.py | yes, 4 fixtures |
| Analysis node | test_analysis.py, test_analysis_fixture_replay.py | record_analysis_fixture.py | yes, 1 fixture |
| Fast-synth node | test_synth.py | record_synth_fixture.py | no fixture recorded yet |
| Research | test_research.py, test_tts.py | record_research_fixture.py | no fixture recorded yet |
| TTS providers | test_tts.py, test_vision.py, test_synth.py, test_graph.py | tts_smoke.py | no (live smoke only) |
| Gradio UI handler | test_ui.py, test_accessibility.py | ui_smoke.py | no (live smoke only) |
