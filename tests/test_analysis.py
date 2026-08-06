"""Tests for the deep-analysis node (issue #8 / P1.5): dense-document synthesis.

No network calls: every test substitutes a fake client object for
OpenRouterClient (same seam pattern as test_vision.py / test_synth.py).
This node is the FIRST use of the `brain` ladder (vision/fast_synth both use
`eyes`), and its whole reason to exist is exact reproduction of amounts,
dates and identifiers from dense documents (bills, receipts, medication
labels) - so, in addition to mirroring synth.py's degradation/TTS-safety
coverage, this file pins:
  - the request targets role "brain", not "eyes"
  - critical values (account number, dollar amount, date) survive verbatim
    through to_spoken_text
  - empty scraper_data still produces a useful script, not a hedge
"""

import re
from pathlib import Path

import pytest

from clarif_eye.client import CompletionResult, LadderExhaustedError, OpenRouterError
from clarif_eye.graph import build_graph, analysis_node
from clarif_eye.state import make_initial_state
from clarif_eye import analysis
from clarif_eye.analysis import run_analysis

# --- Shared TTS-safety assertion, same as test_synth.py's -------------------

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "]"
)


def assert_tts_safe(text):
    """A script that will be read aloud must contain none of these."""
    assert isinstance(text, str)
    assert text.strip() != ""
    assert "**" not in text, f"markdown bold survived: {text!r}"
    assert "__" not in text, f"markdown bold/underline survived: {text!r}"
    assert "`" not in text, f"backtick/code fence survived: {text!r}"
    assert not re.search(r"(^|\n)\s*#+\s", text), f"markdown heading survived: {text!r}"
    assert not re.search(r"(^|\n)\s*[-*•]\s", text), f"bullet point survived: {text!r}"
    assert not re.search(r"(^|\n)\s*\d+[.)]\s", text), f"numbered list survived: {text!r}"
    assert "|" not in text, f"pipe character (table) survived: {text!r}"
    assert not _EMOJI_PATTERN.search(text), f"emoji survived: {text!r}"
    assert "http://" not in text and "https://" not in text and "www." not in text, (
        f"raw URL survived: {text!r}"
    )
    assert not re.search(r"([^\w\s])\1{2,}", text), f"bare punctuation run survived: {text!r}"
    assert text.count("[") == text.count("]"), f"unbalanced brackets survived: {text!r}"
    assert text.count("(") == text.count(")"), f"unbalanced parentheses survived: {text!r}"
    assert "](" not in text, f"raw markdown link fragment survived: {text!r}"
    assert not re.search(r"</?[a-zA-Z][^<>]*>", text), f"HTML/angle-bracket tag survived: {text!r}"
    assert ", ," not in text, f"comma run survived: {text!r}"
    assert not text.startswith(","), f"leading comma survived: {text!r}"
    assert not text.endswith(","), f"trailing comma survived: {text!r}"


# --- Fake client, same shape as test_synth.py's ------------------------------


class FakeAnalysisClient:
    def __init__(self, content=None, exc=None, model="fake-brain-model:free"):
        self.content = content
        self.exc = exc
        self.model = model
        self.calls = []
        self.closed = False

    def complete(self, role, messages, **params):
        self.calls.append({"role": role, "messages": messages, "params": params})
        if self.exc is not None:
            raise self.exc
        return CompletionResult(content=self.content, model=self.model)

    def close(self):
        self.closed = True


def _assert_reasonable_message(message, *, mentions):
    assert isinstance(message, str)
    words = message.split()
    assert len(words) >= 5, f"message too short to be meaningful: {message!r}"
    assert mentions in message.lower(), f"expected {mentions!r} in message: {message!r}"


# --- CHECK B: request targets the brain role, not eyes -----------------------


def test_request_targets_brain_role_not_eyes():
    client = FakeAnalysisClient(content="This is a water utility bill. Amount due is $104.95.")

    run_analysis("Account Number: 4471-2205-88", "a utility bill", "", client)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["role"] == "brain"
    assert call["role"] != "eyes"


# --- Happy path ---------------------------------------------------------------


def test_happy_path_returns_spoken_final_output():
    client = FakeAnalysisClient(
        content="This is a water utility bill. The amount due is $104.95, due by 22 July 2026."
    )

    result = run_analysis("Account Number: 4471-2205-88", "a utility bill", "", client)

    assert "final_output" in result
    assert_tts_safe(result["final_output"])


# --- CHECK C: exact reproduction of amounts/dates/identifiers ---------------


def test_critical_values_survive_verbatim_through_sanitisation():
    ocr_output = (
        "CITY OF RIVERTON WATER UTILITY STATEMENT Account Number: 4471-2205-88 "
        "AMOUNT DUE $104.95 PAYMENT DUE BY: 22 JULY 2026"
    )
    scene_context = (
        "A rectangular water utility statement from the City of Riverton showing "
        "account details, billing period, charges, amount due, and payment deadline."
    )
    # Plausible model reply, including markup a model might add anyway.
    dirty_reply = (
        "## Water Utility Bill\n\n"
        "This is a **water utility statement** from the City of Riverton.\n\n"
        "- Account Number: 4471-2205-88\n"
        "- Amount Due: $104.95\n"
        "- Payment Due By: 22 JULY 2026\n"
    )
    client = FakeAnalysisClient(content=dirty_reply)

    result = run_analysis(ocr_output, scene_context, "", client)

    final_output = result["final_output"]
    assert_tts_safe(final_output)
    assert "4471-2205-88" in final_output
    assert "$104.95" in final_output
    assert "22 JULY 2026" in final_output


# --- CHECK E: empty scraper_data still produces a useful script -------------


def test_empty_scraper_data_still_produces_useful_script():
    client = FakeAnalysisClient(
        content="This is a water utility bill. The amount due is $104.95, due by 22 July 2026."
    )

    result = run_analysis(
        "Account Number: 4471-2205-88 AMOUNT DUE $104.95 PAYMENT DUE BY: 22 July 2026",
        "a utility bill",
        "",
        client,
    )

    assert len(client.calls) == 1
    assert_tts_safe(result["final_output"])
    # No hedging placeholder about missing web context - just proceeds.
    assert "$104.95" in result["final_output"]


def test_non_empty_scraper_data_is_included_in_the_request():
    client = FakeAnalysisClient(content="This medication is a common pain reliever.")

    run_analysis(
        "Ibuprofen 200mg", "a medication bottle", "Ibuprofen is an NSAID pain reliever.", client
    )

    call = client.calls[0]
    messages = call["messages"]
    content = messages[0]["content"]
    text = content[0]["text"] if isinstance(content, list) else content
    assert "Ibuprofen is an NSAID pain reliever." in text


# --- CHECK F: markup-heavy reply is sanitised (before/after) ----------------


def test_markup_heavy_reply_is_sanitised_into_clean_spoken_prose():
    dirty_reply = (
        "# Medication Label\n\n"
        "This is **Ibuprofen** from `MedCo`:\n\n"
        "- Dosage: 200mg\n"
        "- Refills: 2\n\n"
        "| Field | Value |\n"
        "|------|-------|\n"
        "| Dosage | 200mg |\n\n"
        "1. Take twice daily\n"
        "2. Visit https://medco.example.com for details!!!\n"
    )
    client = FakeAnalysisClient(content=dirty_reply)

    result = run_analysis("Ibuprofen 200mg", "a medication bottle", "", client)

    final_output = result["final_output"]
    assert_tts_safe(final_output)
    assert "Ibuprofen" in final_output
    assert "200mg" in final_output


# --- Input edge case: empty OCR with a vision degradation message ----------


@pytest.mark.parametrize(
    "degradation_message",
    [
        "Vision could not run right now: every available model was busy or "
        "unavailable. Please try again in a moment.",
        "Vision could not run because of a configuration problem with the "
        "service. Please tell whoever set this up.",
        "The vision model returned an empty response.",
        "The vision model's response could not be understood.",
    ],
)
def test_empty_ocr_with_vision_degradation_message_is_not_echoed_as_a_description(
    degradation_message,
):
    client = FakeAnalysisClient(content="This should never be used.")

    result = run_analysis("", degradation_message, "", client)

    assert len(client.calls) == 0
    assert_tts_safe(result["final_output"])
    assert degradation_message.split()[0] in result["final_output"]


def test_real_ocr_output_is_not_dropped_when_scene_context_is_degraded():
    result = run_analysis(
        "Account Number: 4471-2205-88",
        "The vision model's response could not be understood.",
        "",
    )

    assert_tts_safe(result["final_output"])
    assert "4471-2205-88" in result["final_output"]
    assert "could not be understood" in result["final_output"]


# --- Degradation: LadderExhaustedError --------------------------------------


def test_ladder_exhausted_degrades_without_raising():
    client = FakeAnalysisClient(exc=LadderExhaustedError("brain", ()))

    result = run_analysis("some text", "a scene", "", client)

    _assert_reasonable_message(result["final_output"], mentions="busy")
    assert_tts_safe(result["final_output"])


# --- Degradation: terminal OpenRouterError ----------------------------------


def test_openrouter_error_degrades_without_raising():
    client = FakeAnalysisClient(exc=OpenRouterError("authentication failed"))

    result = run_analysis("some text", "a scene", "", client)

    _assert_reasonable_message(result["final_output"], mentions="configuration")
    assert_tts_safe(result["final_output"])


# --- Degradation: unexpected exception types --------------------------------


@pytest.mark.parametrize("exc", [ValueError("bad"), TimeoutError("timed out"), RuntimeError("oops")])
def test_unexpected_exception_types_degrade_without_raising(exc):
    client = FakeAnalysisClient(exc=exc)

    result = run_analysis("some text", "a scene", "", client)

    _assert_reasonable_message(result["final_output"], mentions="unexpected")
    assert_tts_safe(result["final_output"])


def test_key_error_degrades_without_raising():
    client = FakeAnalysisClient(exc=KeyError("missing"))

    result = run_analysis("some text", "a scene", "", client)

    _assert_reasonable_message(result["final_output"], mentions="unexpected")
    assert_tts_safe(result["final_output"])


def test_keyboard_interrupt_is_not_swallowed():
    client = FakeAnalysisClient(exc=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        run_analysis("some text", "a scene", "", client)


# --- Degradation: non-string reply ------------------------------------------


@pytest.mark.parametrize("bad_reply", [{}, 5, None])
def test_non_string_reply_degrades_without_raising(bad_reply):
    client = FakeAnalysisClient(content=bad_reply)

    result = run_analysis("some text", "a scene", "", client)

    _assert_reasonable_message(result["final_output"], mentions="empty")
    assert_tts_safe(result["final_output"])


# --- Degradation: empty/whitespace-only reply -------------------------------


def test_empty_reply_degrades_without_raising():
    client = FakeAnalysisClient(content="   ")

    result = run_analysis("some text", "a scene", "", client)

    _assert_reasonable_message(result["final_output"], mentions="empty")
    assert_tts_safe(result["final_output"])


def test_reply_that_sanitises_to_blank_degrades_without_raising():
    client = FakeAnalysisClient(content="### | | | ***")

    result = run_analysis("some text", "a scene", "", client)

    _assert_reasonable_message(result["final_output"], mentions="empty")
    assert_tts_safe(result["final_output"])


# --- FIX 1: numeric fidelity enforced in code, not just the prompt ----------
#
# THE CENTRAL RISK (module docstring): a blind user cannot check the spoken
# script against the source document, so an invented amount/date/identifier
# is the worst output this node can produce. The prompt asks the model not
# to do this, but a prompt is not enforcement - these tests pin the
# code-level backstop in run_analysis.


def test_invented_amount_with_transposed_digit_degrades():
    ocr_output = (
        "CITY OF RIVERTON WATER UTILITY STATEMENT Account Number: 4471-2205-88 "
        "AMOUNT DUE $104.95 PAYMENT DUE BY: 22 JULY 2026"
    )
    scene_context = "a water utility statement"
    # $1,045.95 never appears anywhere in the inputs - only $104.95 does.
    client = FakeAnalysisClient(
        content="This is a water utility bill. The amount due is $1,045.95."
    )

    result = run_analysis(ocr_output, scene_context, "", client)

    final_output = result["final_output"]
    assert_tts_safe(final_output)
    assert "1,045.95" not in final_output and "1045.95" not in final_output
    assert "verified" in final_output.lower() or "not safe" in final_output.lower()


def test_invented_date_degrades():
    ocr_output = (
        "CITY OF RIVERTON WATER UTILITY STATEMENT Account Number: 4471-2205-88 "
        "AMOUNT DUE $104.95 PAYMENT DUE BY: 22 JULY 2026"
    )
    scene_context = "a water utility statement"
    # "23" never appears anywhere in the inputs - only "22 JULY 2026" does.
    client = FakeAnalysisClient(
        content="This is a water utility bill. Payment is due by 23 July 2026."
    )

    result = run_analysis(ocr_output, scene_context, "", client)

    final_output = result["final_output"]
    assert_tts_safe(final_output)
    assert "23 July" not in final_output
    assert "verified" in final_output.lower() or "not safe" in final_output.lower()


def test_legitimate_reply_with_numbers_traceable_to_inputs_passes():
    ocr_output = (
        "CITY OF RIVERTON WATER UTILITY STATEMENT Account Number: 4471-2205-88 "
        "AMOUNT DUE $104.95 PAYMENT DUE BY: 22 JULY 2026"
    )
    scene_context = "a water utility statement"
    client = FakeAnalysisClient(
        content="This is a water utility bill. Amount due is $104.95, due by 22 JULY 2026."
    )

    result = run_analysis(ocr_output, scene_context, "", client)

    final_output = result["final_output"]
    assert_tts_safe(final_output)
    assert "$104.95" in final_output
    assert "22 JULY 2026" in final_output


def test_reply_with_no_numbers_at_all_passes_with_nothing_to_verify():
    client = FakeAnalysisClient(
        content="This appears to be a plain letter with no figures on it."
    )

    result = run_analysis("some prose with no digits", "a letter", "", client)

    assert_tts_safe(result["final_output"])
    assert "letter" in result["final_output"].lower()


def test_real_fixture_reply_passes_number_verification():
    """PROVE IT: the actual recorded brain reply must pass the check."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    raw_path = fixtures_dir / "analysis_reply_raw.txt"
    if not raw_path.exists():
        pytest.skip(f"Fixture not found: {raw_path.name}")
    raw_reply = raw_path.read_text()
    ocr_output = (
        "CITY OF RIVERTON WATER UTILITY STATEMENT Account Number: 4471-2205-88 "
        "Billing Period: 01 Jun 2026 to 30 Jun 2026 Service Address: 1188 Kestrel "
        "Lane, Apt 4B Previous Balance $41.20 Current Charges $63.75 Late Fee "
        "$0.00 AMOUNT DUE $104.95 PAYMENT DUE BY: 22 JULY 2026 Pay online at "
        "riverton.gov/water"
    )
    scene_context = (
        "A rectangular water utility statement from the City of Riverton showing "
        "account details, billing period, charges, amount due, and payment deadline."
    )
    client = FakeAnalysisClient(content=raw_reply)

    result = run_analysis(ocr_output, scene_context, "", client)

    final_output = result["final_output"]
    assert_tts_safe(final_output)
    assert "4471-2205-88" in final_output
    assert "$104.95" in final_output
    assert "$41.20" in final_output
    assert "$63.75" in final_output
    assert "22 JULY 2026" in final_output


# --- FIX 4: scraper_data is capped so a huge scrape cannot silently -------
# --- truncate ocr_output/scene_context out of the model's context window --


def test_scraper_data_is_capped_with_truncation_marker():
    huge_scrape = "word " * 20000  # far larger than the cap
    messages = analysis._build_messages("some ocr text", "a scene", huge_scrape)

    text = messages[0]["content"][0]["text"]

    assert "[context truncated]" in text
    assert len(text) < len(huge_scrape)


def test_scraper_data_under_cap_is_not_truncated():
    small_scrape = "Ibuprofen is an NSAID pain reliever."
    messages = analysis._build_messages("some ocr text", "a scene", small_scrape)

    text = messages[0]["content"][0]["text"]

    assert small_scrape in text
    assert "[context truncated]" not in text


# --- FIX 5: pin empty-input and empty-scene-context-only behaviors ----------


def test_all_empty_inputs_degrades_with_no_description_message():
    client = FakeAnalysisClient(content="This should never be used.")

    result = run_analysis("", "", "", client)

    assert len(client.calls) == 0
    assert_tts_safe(result["final_output"])
    assert result["final_output"] == "No description is available for this photo."


def test_real_scene_with_empty_ocr_and_empty_scraper_calls_the_model():
    client = FakeAnalysisClient(content="This is a plain letter with no visible figures.")

    result = run_analysis("", "a handwritten letter", "", client)

    assert len(client.calls) == 1
    assert client.calls[0]["role"] == "brain"
    assert_tts_safe(result["final_output"])


# --- Client lifecycle --------------------------------------------------------


def test_self_constructed_client_is_closed(monkeypatch):
    fake = FakeAnalysisClient(content="This is a bill for $104.95.")
    monkeypatch.setattr(analysis, "_default_client", lambda: fake)

    run_analysis("some text", "a scene", "")

    assert fake.closed is True


def test_injected_client_is_not_closed():
    client = FakeAnalysisClient(content="This is a bill for $104.95.")

    run_analysis("some text", "a scene", "", client)

    assert client.closed is False


def test_self_constructed_client_is_closed_even_on_degraded_path(monkeypatch):
    fake = FakeAnalysisClient(exc=LadderExhaustedError("brain", ()))
    monkeypatch.setattr(analysis, "_default_client", lambda: fake)

    run_analysis("some text", "a scene", "")

    assert fake.closed is True


def test_no_client_constructed_when_scene_context_is_a_degradation_message(monkeypatch):
    def _boom():
        raise AssertionError("client should not be constructed on this path")

    monkeypatch.setattr(analysis, "_default_client", _boom)

    result = run_analysis("", "The vision model returned an empty response.", "")

    assert_tts_safe(result["final_output"])


# --- analysis_node: client injection, graph-facing wrapper ------------------


def test_analysis_node_accepts_an_explicit_injected_client():
    client = FakeAnalysisClient(content="This is a bill for $104.95.")
    state = {"ocr_output": "some text", "scene_context": "a scene", "scraper_data": ""}

    result = analysis_node(state, client=client)

    assert_tts_safe(result["final_output"])
    assert len(client.calls) == 1
    assert client.calls[0]["role"] == "brain"


def test_analysis_node_accepts_client_injected_via_config_configurable():
    client = FakeAnalysisClient(content="This is a bill for $104.95.")
    state = {"ocr_output": "some text", "scene_context": "a scene", "scraper_data": ""}

    result = analysis_node(state, config={"configurable": {"client": client}})

    assert_tts_safe(result["final_output"])
    assert len(client.calls) == 1


# --- CHECK G: full compiled graph runs end to end on the research path -----


def test_full_compiled_graph_runs_end_to_end_on_research_path_with_fake_client():
    vision_reply = (
        "OCR_TEXT: Account Number: 4471-2205-88 AMOUNT DUE $104.95\n"
        "SCENE: a water utility bill"
    )

    class FakeVisionThenAnalysisClient:
        def __init__(self):
            self.calls = []
            self.closed = False

        def complete(self, role, messages, **params):
            self.calls.append(role)
            if len(self.calls) == 1:
                return CompletionResult(content=vision_reply, model="fake-eyes")
            assert role == "brain"
            return CompletionResult(
                content=(
                    "This is a water utility bill. Account number 4471-2205-88. "
                    "Amount due $104.95."
                ),
                model="fake-brain",
            )

        def close(self):
            self.closed = True

    client = FakeVisionThenAnalysisClient()
    graph = build_graph()
    state = make_initial_state("base64data")

    result = graph.invoke(state, config={"configurable": {"trace": [], "client": client}})

    assert result["complexity_flag"] is True
    assert_tts_safe(result["final_output"])
    assert "4471-2205-88" in result["final_output"]
    assert result["audio_file_path"] != ""
