"""Dynamic router complexity heuristic (issue #6 / P1.3).

Owns the decision behind state["complexity_flag"]: True routes to the slow
research + analysis path (web lookup + the `brain` model, 45s budget);
False routes to the fast_synth path. graph.py's dynamic_router keeps
reading state["complexity_flag"] unchanged - this module only supplies the
value, computed locally with no model or network call, per the
architecture doc's requirement that routing be pure Python.

THE PROBLEM WITH THE DOC'S RULE
--------------------------------
The architecture doc specifies `len(ocr_output) > 50 words OR keywords ->
research`. The one real recorded sample we have (tests/fixtures/
vision_reply_parsed.json, a utility bill) is 47 words and was recorded
complexity_flag=true. At 47 words the doc's rule sends it to the FAST
path - wrong: a bill dense with an account number, a billing period, four
dollar amounts, and a due date is exactly what a blind user wants read
carefully (amount due, deadline), not glanced at. Word count alone can't
tell a long rambling sign (simple) from a short dense receipt (not
simple).

THE CHOSEN RULE
----------------
Score three data-density signals, independent of length:
  1. digit density   - total digit characters in ocr_output
  2. currency hits   - count of currency-symbol-followed-by-digit matches
  3. keyword hits     - document/identifier keyword substrings, checked
                         across ocr_output AND scene_context (the vision
                         model's own description - e.g. "a...statement" -
                         is itself a legitimate signal)

complexity_flag is True if at least `signal_score_threshold` of those 3
signals fire, OR ocr_output has at least `word_count_threshold` words (a
much higher bar than the doc's 50 - a fallback for genuinely long
documents that don't happen to match the keyword list, not a trigger for
an ordinary long sign or sentence).

All thresholds and the keyword/currency-symbol lists live in
config/models.toml under [router] (see load_router_config), not as
literals here - this is a judgment calibration against the single real
sample we have, not a tuned corpus, and it should be easy to revise
without a code change.
"""

import re
import tomllib
from importlib import resources
from pathlib import Path
from typing import NamedTuple


class RouterError(Exception):
    """Raised for any router configuration error."""


class RouterConfig(NamedTuple):
    word_count_threshold: int
    signal_score_threshold: int
    digit_count_threshold: int
    currency_count_threshold: int
    keyword_hit_threshold: int
    currency_symbols: tuple
    keywords: tuple


class RouterDecision(NamedTuple):
    complexity_flag: bool
    reason: str


def _validate_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RouterError(f"router config {name!r} must be a non-negative int, got {value!r}")
    return value


def _validate_string_list(value, name):
    if not isinstance(value, list) or len(value) == 0:
        raise RouterError(f"router config {name!r} must be a non-empty list of strings")
    cleaned = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise RouterError(f"router config {name!r} contains a blank or non-string entry: {entry!r}")
        cleaned.append(entry)
    return tuple(cleaned)


def _validate_router_section(section):
    if not isinstance(section, dict):
        raise RouterError("model config's [router] section must be a table")

    required_int_fields = (
        "word_count_threshold",
        "signal_score_threshold",
        "digit_count_threshold",
        "currency_count_threshold",
        "keyword_hit_threshold",
    )
    for field in required_int_fields:
        if field not in section:
            raise RouterError(f"router config is missing required field {field!r}")

    values = {field: _validate_positive_int(section[field], field) for field in required_int_fields}

    if not (1 <= values["signal_score_threshold"] <= 3):
        raise RouterError(
            "router config 'signal_score_threshold' must be between 1 and 3 "
            f"(there are 3 signals), got {values['signal_score_threshold']!r}"
        )

    for field in ("currency_symbols", "keywords"):
        if field not in section:
            raise RouterError(f"router config is missing required field {field!r}")

    currency_symbols = _validate_string_list(section["currency_symbols"], "currency_symbols")
    keywords = tuple(k.lower() for k in _validate_string_list(section["keywords"], "keywords"))

    return RouterConfig(
        word_count_threshold=values["word_count_threshold"],
        signal_score_threshold=values["signal_score_threshold"],
        digit_count_threshold=values["digit_count_threshold"],
        currency_count_threshold=values["currency_count_threshold"],
        keyword_hit_threshold=values["keyword_hit_threshold"],
        currency_symbols=currency_symbols,
        keywords=keywords,
    )


def load_router_config(path=None):
    """Load and validate the [router] section of the model config.

    `path` defaults to the packaged config/models.toml (same file and
    resolution approach as registry.load_registry), never relative to the
    current working directory. Pass an explicit path (e.g. a tmp_path
    fixture) to load a standalone router config for tests.
    """
    source = Path(path) if path is not None else resources.files("clarif_eye").joinpath("config", "models.toml")
    try:
        with source.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError as e:
        raise RouterError(f"model config file not found: {source}") from e
    except tomllib.TOMLDecodeError as e:
        raise RouterError(f"model config file is not valid TOML: {source}") from e
    except OSError as e:
        raise RouterError(f"could not read model config file: {source}") from e

    if "router" not in data:
        raise RouterError("model config is missing required section [router]")

    return _validate_router_section(data["router"])


def _currency_hit_count(text, currency_symbols):
    count = 0
    for symbol in currency_symbols:
        count += len(re.findall(re.escape(symbol) + r"\d", text))
    return count


def _keyword_hit_count(text_lower, keywords):
    return sum(1 for kw in keywords if kw in text_lower)


def classify_complexity(ocr_output, scene_context, config=None):
    """Decide complexity_flag from ocr_output and scene_context.

    Pure, deterministic, no network/model call: same input always yields
    the same RouterDecision. Returns (complexity_flag, reason) so callers
    (and tests) can see WHY a given input routed where it did without a
    state schema change - state stays at exactly 7 keys.
    """
    if config is None:
        config = load_router_config()

    ocr_output = ocr_output or ""
    scene_context = scene_context or ""

    word_count = len(ocr_output.split())
    digit_count = sum(ch.isdigit() for ch in ocr_output)
    currency_count = _currency_hit_count(ocr_output, config.currency_symbols)

    combined_lower = f"{ocr_output}\n{scene_context}".lower()
    keyword_hits = _keyword_hit_count(combined_lower, config.keywords)

    digit_signal = digit_count >= config.digit_count_threshold
    currency_signal = currency_count >= config.currency_count_threshold
    keyword_signal = keyword_hits >= config.keyword_hit_threshold
    signal_score = sum((digit_signal, currency_signal, keyword_signal))

    long_document = word_count >= config.word_count_threshold
    complexity_flag = signal_score >= config.signal_score_threshold or long_document

    reason = (
        f"word_count={word_count} (long_document>={config.word_count_threshold}: {long_document}); "
        f"digits={digit_count} (>={config.digit_count_threshold}: {digit_signal}); "
        f"currency_hits={currency_count} (>={config.currency_count_threshold}: {currency_signal}); "
        f"keyword_hits={keyword_hits} (>={config.keyword_hit_threshold}: {keyword_signal}); "
        f"signal_score={signal_score}/{config.signal_score_threshold} "
        f"-> {'research' if complexity_flag else 'fast_synth'}"
    )

    return RouterDecision(complexity_flag=complexity_flag, reason=reason)
