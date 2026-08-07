"""Pairing registry for P4.2 (#14): every external-system mock gets a
real-stack scenario, and the pairing is checked automatically here.

WHY THIS SHAPE: the working process says a mock-based test of an external
system must be paired with a scenario against the real running stack -
mocks mirror assumptions, scenarios check them. A registry nobody is forced
to update is documentation, not a check, so this file also SCANS tests/*.py
for the mocking primitives this project actually uses (httpx.MockTransport,
and the FakeXxx / _FakeXxx classes every mock-based test in this repo
defines - see e.g. test_client.py's, test_vision.py's, test_tts.py's module
docstrings) and fails if any such module is not listed in SCENARIOS below.
Adding a new mock-based test without registering it here is a red test, not
a silent gap.

WHAT COUNTS AS A "REAL-STACK SCENARIO" HERE: a script under scripts/ that
makes a real network call (all of them are MANUAL-ONLY per their own module
docstrings - see docs/SCENARIOS.md), and/or a recorded-fixture replay test
that runs a real model's verbatim recorded output through the production
parsing/verification code on every test run, offline. Which of these each
entry has, and what each one actually proves (a recorder writes a fixture
and asserts nothing; a replay test asserts against real recorded output; a
smoke script asserts pass/fail against a live call), is spelled out in
docs/SCENARIOS.md - this file only checks that the pairing exists and that
every file it names is real.

MANUAL, NOT CI: every scripts/*.py scenario here makes a real network call
and is never run by this test suite or by CI (#21 owns CI; model inference
must never run there). These tests check that the pairing EXISTS - file
presence, and that mocks are accounted for - not that a live scenario still
passes today.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- The registry -------------------------------------------------------
#
# One entry per external system this project talks to. `mock_test_modules`
# lists every tests/*.py file that stands in for that system with a fake/
# MockTransport instead of a real call. `scenario_paths` lists the file(s)
# under scripts/ and/or tests/ that exercise the REAL system - see
# docs/SCENARIOS.md for what each one actually proves.

SCENARIOS = [
    {
        "system": "OpenRouter chat completions (clarif_eye.client.OpenRouterClient)",
        "mock_test_modules": [
            "tests/test_client.py",
            "tests/test_benchmark_script.py",
            "tests/test_graph.py",
            "tests/test_pipeline_deadline.py",
            "tests/test_failure_messages.py",
            "tests/test_checkpointing.py",
        ],
        "scenario_paths": [
            "scripts/live_smoke.py",
            "scripts/benchmark_ladders.py",
        ],
    },
    {
        "system": "Vision node (clarif_eye.vision, the eyes role)",
        "mock_test_modules": [
            "tests/test_vision.py",
            "tests/test_vision_fixture_replay.py",
        ],
        "scenario_paths": [
            "scripts/record_vision_fixture.py",
            "tests/test_vision_fixture_replay.py",
        ],
    },
    {
        "system": "Analysis node (clarif_eye.analysis, the brain role)",
        "mock_test_modules": [
            "tests/test_analysis.py",
            "tests/test_analysis_fixture_replay.py",
        ],
        "scenario_paths": [
            "scripts/record_analysis_fixture.py",
            "tests/test_analysis_fixture_replay.py",
        ],
    },
    {
        "system": "Fast-synth node (clarif_eye.synth, the eyes role again)",
        "mock_test_modules": [
            "tests/test_synth.py",
            "tests/test_synth_fixture_replay.py",
        ],
        "scenario_paths": [
            "scripts/record_synth_fixture.py",
            "tests/test_synth_fixture_replay.py",
        ],
    },
    {
        "system": "Research: DuckDuckGo search + page fetch (clarif_eye.research)",
        "mock_test_modules": [
            "tests/test_research.py",
            "tests/test_tts.py",
        ],
        "scenario_paths": [
            "scripts/record_research_fixture.py",
            "tests/test_research_fixture_replay.py",
        ],
    },
    {
        "system": "TTS providers: edge-tts and gTTS (clarif_eye.tts)",
        "mock_test_modules": [
            "tests/test_tts.py",
            "tests/test_vision.py",
            "tests/test_synth.py",
            "tests/test_graph.py",
            "tests/test_checkpointing.py",
        ],
        "scenario_paths": [
            "scripts/tts_smoke.py",
        ],
    },
    {
        "system": "Gradio UI handler (clarif_eye.ui.handle_submit / build_resources)",
        "mock_test_modules": [
            "tests/test_ui.py",
            "tests/test_accessibility.py",
            # Issue #82 / P9.3: drives clarif_eye.ui.handle_ask_staged with
            # a recording fake client and a fake TTS provider. Paired with
            # the SAME scenario script, which now takes an optional
            # follow-up question and runs photo-then-question on one
            # thread_id against the live stack - see its docstring for what
            # that round trip proves and what it cannot.
            "tests/test_followup.py",
            # Issue #83 / P9.4: drives the pause-and-ask flow (a run that
            # stops because a number in the drafted script could not be
            # traced back to the photo) with a recording fake client and a
            # fake TTS provider.
            #
            # HONEST NOTE ON THIS PAIRING: scripts/ui_smoke.py exercises the
            # real stack end to end, but it cannot MAKE a live model invent
            # a number on demand, so it cannot be relied on to reach the
            # paused branch. The real-model evidence for the gate itself is
            # the second scenario path below - a recorded, verbatim reply
            # from a real brain model, run through the production verifier
            # on every test run, including a corrupted copy of it that must
            # fail verification. That is what proves the check fires on
            # genuine model text rather than only on text a test wrote to
            # be caught.
            "tests/test_ask_before_speaking.py",
        ],
        "scenario_paths": [
            "scripts/ui_smoke.py",
            "tests/test_analysis_fixture_replay.py",
        ],
    },
]

# --- Mocking-primitive detection -----------------------------------------
#
# The two shapes every mock-based test in this repo actually uses (see
# test_client.py's and test_research.py's "No network calls" docstrings for
# httpx.MockTransport, and test_vision.py's/test_tts.py's/etc for the
# FakeXxx-class pattern). Matched at line start so a mention of "class Fake"
# inside a comment or a longer word never counts as a definition.
_MOCK_PRIMITIVE_RE = re.compile(r"^\s*class _?Fake\w*\b|httpx\.MockTransport\(", re.MULTILINE)

# This file itself is exempt from the scan: it deliberately names every
# mock-bearing module as data (strings), never defines a Fake* class or
# calls httpx.MockTransport itself.
_SELF = "test_scenario_pairing.py"


def _detect_mock_test_modules():
    """Every tests/*.py file that defines a FakeXxx/_FakeXxx class or calls
    httpx.MockTransport(...) - i.e. every module that mocks something
    instead of calling it for real."""
    detected = set()
    for path in sorted((REPO_ROOT / "tests").glob("*.py")):
        if path.name == _SELF:
            continue
        text = path.read_text(encoding="utf-8")
        if _MOCK_PRIMITIVE_RE.search(text):
            detected.add(f"tests/{path.name}")
    return detected


def _registered_mock_test_modules():
    registered = set()
    for entry in SCENARIOS:
        registered.update(entry["mock_test_modules"])
    return registered


# --- Checks ---------------------------------------------------------------


def test_registry_entries_have_the_required_shape():
    for entry in SCENARIOS:
        assert entry["system"], entry
        assert entry["mock_test_modules"], f"{entry['system']}: no mock_test_modules listed"
        assert entry["scenario_paths"], f"{entry['system']}: no scenario_paths listed"


def test_every_registered_mock_test_module_exists():
    missing = [
        f"{entry['system']}: {rel_path}"
        for entry in SCENARIOS
        for rel_path in entry["mock_test_modules"]
        if not (REPO_ROOT / rel_path).is_file()
    ]
    assert not missing, "Registered mock_test_modules that do not exist:\n" + "\n".join(missing)


def test_every_scenario_path_exists():
    missing = [
        f"{entry['system']}: {rel_path}"
        for entry in SCENARIOS
        for rel_path in entry["scenario_paths"]
        if not (REPO_ROOT / rel_path).is_file()
    ]
    assert not missing, (
        "Registered scenario_paths that do not exist - a scenario this "
        "registry claims pairs with a mock is missing:\n" + "\n".join(missing)
    )


def test_no_external_system_mock_is_unpaired():
    """The core check: a test module that mocks an external system (via
    httpx.MockTransport or a FakeXxx/_FakeXxx class) must be named in some
    SCENARIOS entry's mock_test_modules. An unregistered mock is the
    failing condition the issue's Done-when line describes."""
    detected = _detect_mock_test_modules()
    registered = _registered_mock_test_modules()
    unpaired = sorted(detected - registered)
    assert not unpaired, (
        "These test modules mock an external system (a FakeXxx/_FakeXxx "
        "class or httpx.MockTransport) but are not listed in any "
        "SCENARIOS entry in tests/test_scenario_pairing.py: "
        + ", ".join(unpaired)
        + ". Add the module to an existing entry's mock_test_modules, or "
        "add a new entry pairing it with a real-stack scenario."
    )
