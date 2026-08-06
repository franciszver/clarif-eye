"""Table-driven tests for the dynamic router complexity heuristic (issue #6 / P1.3).

The architecture doc's rule is `len(ocr_output) > 50 words OR keywords ->
research`. Our one real recorded sample (a utility bill, see
tests/fixtures/vision_reply_parsed.json) is 47 words and was recorded with
complexity_flag=True - so word count alone would misroute it. These tests
pin the replacement heuristic: it must weigh data density (digits,
currency, document keywords), not just length, and its thresholds/keyword
lists must come from config, not literals.

No network calls: classify_complexity is pure Python over strings.
"""

import json
from pathlib import Path

import pytest

from clarif_eye.router import RouterConfig, RouterError, classify_complexity, load_router_config

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "vision_reply_parsed.json"


def words(n, token="lorem"):
    return " ".join([token] * n)


# --- Table-driven cases: (id, ocr_output, scene_context, expected_flag) ----

CASES = [
    (
        "empty_ocr",
        "",
        "an empty wall",
        False,
    ),
    (
        "whitespace_only_ocr",
        "   \n\t  ",
        "an empty wall",
        False,
    ),
    (
        "49_words_no_signals",
        words(49),
        "a plain sentence",
        False,
    ),
    (
        "50_words_no_signals",
        words(50),
        "a plain sentence",
        False,
    ),
    (
        "51_words_no_signals",
        words(51),
        "a plain sentence",
        False,
    ),
    (
        "short_dense_receipt",
        "Total $45.00 $12.50 $3.25 Tax $60.75",
        "a small paper receipt",
        True,
    ),
    (
        "long_simple_sign",
        words(120, "welcome"),
        "a large banner sign",
        False,
    ),
    (
        "non_latin_script",
        "こんにちは これは看板です ようこそ",
        "a shop sign in Japanese",
        False,
    ),
    (
        "very_long_document",
        words(200, "paragraph"),
        "a printed page of text",
        True,
    ),
]


@pytest.mark.parametrize("case_id,ocr,scene,expected", CASES, ids=[c[0] for c in CASES])
def test_routing_table(case_id, ocr, scene, expected):
    decision = classify_complexity(ocr, scene)
    assert decision.complexity_flag is expected, (
        f"{case_id}: expected complexity_flag={expected}, got "
        f"{decision.complexity_flag}; reason={decision.reason}"
    )


def test_real_recorded_fixture_routes_to_research():
    """The actual recorded utility bill (47 words) must route to research,
    even though the doc's naive '>50 words' rule would send it to fast."""
    fixture = json.loads(FIXTURE_PATH.read_text())
    assert len(fixture["ocr_output"].split()) < 50, "fixture is expected to be under the doc's word boundary"

    decision = classify_complexity(fixture["ocr_output"], fixture["scene_context"])

    assert decision.complexity_flag is True
    assert fixture["complexity_flag"] is True  # recorded ground truth agrees


# --- Reason string: explainable ---------------------------------------------


def test_decision_includes_a_human_readable_reason():
    decision = classify_complexity("Total $45.00 $12.50 $3.25 Tax $60.75", "a receipt")
    assert isinstance(decision.reason, str)
    assert len(decision.reason) > 0


# --- Purity: deterministic, no network, same input -> same output ----------


def test_classify_is_deterministic():
    ocr = "Total $45.00 $12.50 $3.25 Tax $60.75"
    scene = "a small paper receipt"
    first = classify_complexity(ocr, scene)
    second = classify_complexity(ocr, scene)
    assert first.complexity_flag == second.complexity_flag
    assert first.reason == second.reason


# --- Config-driven: thresholds/keywords are NOT literals in code -----------


def test_config_change_alters_routing_decision():
    ocr = words(170, "paragraph")  # over default word_count_threshold (150), no signals
    scene = "a printed page"

    default_decision = classify_complexity(ocr, scene)
    assert default_decision.complexity_flag is True

    loosened_config = load_router_config()._replace(word_count_threshold=200)
    loosened_decision = classify_complexity(ocr, scene, config=loosened_config)
    assert loosened_decision.complexity_flag is False


def test_config_loads_from_toml_file(tmp_path):
    path = tmp_path / "router.toml"
    path.write_text(
        """
        [router]
        word_count_threshold = 150
        signal_score_threshold = 2
        digit_count_threshold = 6
        currency_count_threshold = 1
        keyword_hit_threshold = 1
        keyword_strong_hit_threshold = 2
        max_avg_word_chars = 20
        chars_per_word_estimate = 2
        currency_symbols = ["$"]
        keywords = ["total", "tax"]
        dosage_units = ["mg"]
        """,
        encoding="utf-8",
    )
    config = load_router_config(path)
    assert config.word_count_threshold == 150
    assert config.keywords == ("total", "tax")


def test_custom_toml_threshold_flips_routing(tmp_path):
    path = tmp_path / "router.toml"
    path.write_text(
        """
        [router]
        word_count_threshold = 150
        signal_score_threshold = 3
        digit_count_threshold = 6
        currency_count_threshold = 1
        keyword_hit_threshold = 1
        keyword_strong_hit_threshold = 2
        max_avg_word_chars = 20
        chars_per_word_estimate = 2
        currency_symbols = ["$"]
        keywords = ["total", "tax"]
        dosage_units = ["mg"]
        """,
        encoding="utf-8",
    )
    # Only currency + digit signals fire (no keyword match), so raising
    # signal_score_threshold to 3 (from the default 2) should flip a
    # receipt that used to route to research back to fast.
    ocr = "45.00 12.50 3.25 60.75"
    scene = "a small paper receipt"

    config = load_router_config(path)
    decision = classify_complexity(ocr, scene, config=config)
    assert decision.complexity_flag is False


# --- Validation: malformed router config -----------------------------------


def test_missing_router_section_raises(tmp_path):
    path = tmp_path / "router.toml"
    path.write_text("[not_router]\nfoo = 1\n", encoding="utf-8")
    with pytest.raises(RouterError):
        load_router_config(path)


def test_missing_router_config_file_raises(tmp_path):
    with pytest.raises(RouterError):
        load_router_config(tmp_path / "does-not-exist.toml")


# --- RouterConfig can be constructed directly for tests ---------------------


def test_router_config_direct_construction():
    config = RouterConfig(
        word_count_threshold=10,
        signal_score_threshold=1,
        digit_count_threshold=2,
        currency_count_threshold=1,
        keyword_hit_threshold=1,
        keyword_strong_hit_threshold=2,
        max_avg_word_chars=20,
        chars_per_word_estimate=2,
        currency_symbols=("$",),
        keywords=("bill",),
    )
    decision = classify_complexity("bill 12 34", "a note", config=config)
    assert decision.complexity_flag is True


# --- FIX 1: medication labels must route to research ------------------------


MEDICATION_LABELS = [
    (
        "amoxicillin_label",
        "TAKE 1 TABLET BY MOUTH TWICE DAILY\nAMOXICILLIN 500 MG\nQTY 30\nREFILLS: 2",
    ),
    (
        "ibuprofen_label",
        "IBUPROFEN 200 MG\nTAKE 2 TABLETS EVERY 6 HOURS",
    ),
    (
        "lisinopril_label",
        "LISINOPRIL 10 MG TABLET\nTAKE ONE TABLET BY MOUTH DAILY\nQTY: 30 REFILLS: 0\nEXP: 03/2027",
    ),
]


@pytest.mark.parametrize("case_id,ocr", MEDICATION_LABELS, ids=[c[0] for c in MEDICATION_LABELS])
def test_medication_labels_route_to_research(case_id, ocr):
    decision = classify_complexity(ocr, "a small printed label")
    assert decision.complexity_flag is True, f"{case_id}: reason={decision.reason}"


# --- FIX 8: dosage units glued to the number ("500mg") must still count ----
# --- as a keyword hit - \bmg\b can't match a unit with no boundary --------
# --- between the digit and the letter --------------------------------------


GLUED_DOSAGE_MEDICATION_LABELS = [
    ("amoxicillin_glued", "AMOXICILLIN 500mg CAPSULES"),
    ("atorvastatin_glued", "ATORVASTATIN 20MG ONE TABLET AT NIGHT"),
    ("metformin_glued", "METFORMIN HCL 850MG TAKE ONE CAPSULE"),
    ("levothyroxine_glued_mcg", "LEVOTHYROXINE 75MCG ONE TABLET DAILY"),
]


@pytest.mark.parametrize(
    "case_id,ocr", GLUED_DOSAGE_MEDICATION_LABELS, ids=[c[0] for c in GLUED_DOSAGE_MEDICATION_LABELS]
)
def test_glued_dosage_medication_labels_route_to_research(case_id, ocr):
    decision = classify_complexity(ocr, "a small printed label")
    assert decision.complexity_flag is True, f"{case_id}: reason={decision.reason}"


def test_spaced_dosage_medication_label_still_routes_to_research():
    decision = classify_complexity("AMOXICILLIN 500 mg CAPSULES", "a small printed label")
    assert decision.complexity_flag is True, decision.reason


# --- Ordinary products with a glued number+unit must NOT be swept into -----
# --- the dosage-unit signal (regression guard, pinned) ----------------------


ORDINARY_GLUED_UNIT_PRODUCTS = [
    ("water_bottle_500ml", "SPRING WATER 500ml"),
    ("soda_can_330ml", "COLA 330ml CAN"),
    ("cereal_box_340g", "CORN FLAKES NET WT 340g"),
    ("drinks_bottle_2l", "SPARKLING WATER 2L"),
    ("distance_sign_100m", "REST AREA 100m"),
    ("opening_hours_24hr", "OPEN 24HR"),
    ("time_6pm", "CLOSES AT 6PM"),
    ("film_35mm", "KODAK 35mm FILM"),
    ("resolution_1080p", "VIDEO 1080p"),
    ("file_size_5mb", "PHOTO.JPG 5MB"),
    ("bulb_60w", "LIGHT BULB 60W"),
]


@pytest.mark.parametrize(
    "case_id,ocr", ORDINARY_GLUED_UNIT_PRODUCTS, ids=[c[0] for c in ORDINARY_GLUED_UNIT_PRODUCTS]
)
def test_ordinary_glued_unit_products_route_to_fast_synth(case_id, ocr):
    decision = classify_complexity(ocr, "a product photo")
    assert decision.complexity_flag is False, f"{case_id}: reason={decision.reason}"


# --- Regression fix: generic time/frequency words must not send ordinary ---
# --- signage to the research path -------------------------------------------


ORDINARY_SIGN_LABELS = [
    ("open_daily_24_hours", "OPEN DAILY 24 HOURS"),
    ("parking_2_hours_max", "PARKING 2 HOURS MAX 8AM-6PM DAILY"),
    ("buses_every_2_hours", "BUSES EVERY 2 HOURS DAILY 06:00 22:00"),
    ("happy_hour_every_day", "HAPPY HOUR EVERY DAY 4PM TO 6PM"),
    ("gym_hours", "GYM HOURS MON-FRI 6AM-10PM"),
]


@pytest.mark.parametrize("case_id,ocr", ORDINARY_SIGN_LABELS, ids=[c[0] for c in ORDINARY_SIGN_LABELS])
def test_ordinary_signs_with_time_words_route_to_fast_synth(case_id, ocr):
    decision = classify_complexity(ocr, "a printed sign")
    assert decision.complexity_flag is False, f"{case_id}: reason={decision.reason}"


# --- FIX 2: scene_context (model prose) must not contribute to scoring ------


def test_cereal_box_scene_prose_does_not_trigger_research():
    ocr = "8901030875021 NET WT 340g"
    scene = "This looks like it could be a receipt or invoice with a balance due"
    decision = classify_complexity(ocr, scene)
    assert decision.complexity_flag is False, decision.reason
    assert "keyword_hits=0" in decision.reason


def test_sticky_note_scene_prose_does_not_trigger_research():
    ocr = "Call 555-867-5309 for details"
    scene = "This resembles an official statement or bill"
    decision = classify_complexity(ocr, scene)
    assert decision.complexity_flag is False, decision.reason
    assert "keyword_hits=0" in decision.reason


# --- FIX 3: keyword matching is whole-word, not unanchored substring -------


UNANCHORED_SUBSTRING_CASES = [
    ("billings", "WELCOME TO BILLINGS MONTANA POP 109936 ELEV 3160"),
    ("totally", "This road is totally closed ahead"),
    ("taxi", "Please hail a taxi outside the airport"),
    ("residue", "Wipe away any residue before use"),
]


@pytest.mark.parametrize("case_id,ocr", UNANCHORED_SUBSTRING_CASES, ids=[c[0] for c in UNANCHORED_SUBSTRING_CASES])
def test_unanchored_substrings_do_not_count_as_keyword_hits(case_id, ocr):
    decision = classify_complexity(ocr, "a sign")
    assert "keyword_hits=0" in decision.reason, f"{case_id}: reason={decision.reason}"


def test_multi_word_keyword_still_matches_whole_phrase():
    decision = classify_complexity("Your order number 12345 has shipped", "a note")
    assert "keyword_hits=1" in decision.reason


# --- FIX 5: each of the 3 signals must be independently load-bearing -------
#
# Each case below fires exactly 2 of the 3 signals to reach the default
# signal_score_threshold of 2, with the 3rd signal deliberately absent (and
# keyword_hits kept below keyword_strong_hit_threshold so the keyword-alone
# override can't mask the result). Deleting any ONE signal from the scoring
# code turns one of its firing signals off, which drops that case's score
# below 2 and flips complexity_flag from True to False - i.e. removing any
# one signal fails at least one of these tests, proven manually in the
# task's CHECK F.


def test_digit_and_currency_signals_alone_reach_threshold_without_keyword():
    decision = classify_complexity("123456 $9.99", "a note")
    assert "keyword_hits=0" in decision.reason
    assert decision.complexity_flag is True, decision.reason


def test_currency_and_keyword_signals_alone_reach_threshold_without_digit():
    decision = classify_complexity("Please pay total $5", "a note")
    assert "digits=1 " in decision.reason
    assert decision.complexity_flag is True, decision.reason


def test_digit_and_keyword_signals_alone_reach_threshold_without_currency():
    decision = classify_complexity("Order number 123456", "a note")
    assert "currency_hits=0" in decision.reason
    assert decision.complexity_flag is True, decision.reason


# --- FIX 6: word-count fallback must be script-aware (CJK has no spaces) ---


def test_long_cjk_document_routes_to_research():
    # No ASCII whitespace at all, so len(text.split()) == 1 regardless of
    # length - this must fall back to a character-based estimate.
    ocr = "これは長い日本語の文章です。" * 25  # ~350 characters, 0 spaces
    assert len(ocr.split()) == 1
    decision = classify_complexity(ocr, "a printed page of Japanese text")
    assert decision.complexity_flag is True, decision.reason


# --- FIX 7: a count threshold of 0 must be rejected at load time -----------


def test_zero_digit_count_threshold_rejected(tmp_path):
    path = tmp_path / "router.toml"
    path.write_text(
        """
        [router]
        word_count_threshold = 150
        signal_score_threshold = 2
        digit_count_threshold = 0
        currency_count_threshold = 1
        keyword_hit_threshold = 1
        keyword_strong_hit_threshold = 2
        max_avg_word_chars = 20
        chars_per_word_estimate = 2
        currency_symbols = ["$"]
        keywords = ["total", "tax"]
        dosage_units = ["mg"]
        """,
        encoding="utf-8",
    )
    with pytest.raises(RouterError, match="digit_count_threshold"):
        load_router_config(path)


# --- FIX 4: config is loaded from disk exactly once, not per call ----------


def test_config_loaded_from_disk_exactly_once_across_many_calls(monkeypatch):
    from clarif_eye import router

    router._cached_default_config.cache_clear()
    calls = []
    real_load_router_config = router.load_router_config

    def counting_load(path=None):
        calls.append(path)
        return real_load_router_config(path)

    monkeypatch.setattr(router, "load_router_config", counting_load)
    try:
        for _ in range(5):
            router.classify_complexity("some ocr text", "some scene")
        assert len(calls) == 1
    finally:
        router._cached_default_config.cache_clear()
