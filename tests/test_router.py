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
        currency_symbols = ["$"]
        keywords = ["total", "tax"]
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
        currency_symbols = ["$"]
        keywords = ["total", "tax"]
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
        currency_symbols=("$",),
        keywords=("bill",),
    )
    decision = classify_complexity("bill 12 34", "a note", config=config)
    assert decision.complexity_flag is True
