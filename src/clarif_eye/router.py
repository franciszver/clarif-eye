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
Score three data-density signals, computed from ocr_output ONLY:
  1. digit density   - total digit characters in ocr_output
  2. currency hits   - count of currency-symbol-followed-by-digit matches
  3. keyword hits     - document/identifier/medication keyword matches
                         (whole-word, see _keyword_hit_count)

scene_context is deliberately NOT scored, even though it is still a
function parameter (callers pass it, and it is kept for future use). It
is model-generated prose describing the photo, not evidence about the
photographed document itself; scoring it made routing depend on how the
vision model chose to phrase its hedge ("could be a receipt or invoice
with a balance due") rather than on what was actually photographed, and
that hedging language could tip an ordinary product photo into the
research path.

complexity_flag is True if:
  - at least `signal_score_threshold` of the 3 signals above fire, OR
  - keyword_hits alone reaches `keyword_strong_hit_threshold` (a lower,
    single-signal bar). This exists because some high-stakes documents -
    a prescription label chief among them - carry their signal almost
    entirely in vocabulary ("TABLETS", "MG", "REFILLS") rather than in
    digit or currency density, and a wrong dosage read aloud is worse
    than an occasional unnecessary trip through the slower path, so two
    or more distinct keyword hits are treated as sufficient on their
    own. This is a judgment call, not measured, and is configurable.
  - OR ocr_output implies at least `word_count_threshold` words (a much
    higher bar than the doc's 50 - a fallback for genuinely long
    documents that don't happen to match the keyword list, not a trigger
    for an ordinary long sign or sentence). Word count is normally
    whitespace-based, but scripts without whitespace (e.g. CJK) are
    estimated from character count instead - see classify_complexity.

All thresholds and the keyword/currency-symbol lists live in
config/models.toml under [router] (see load_router_config), not as
literals here - this is a judgment calibration against the single real
sample we have, not a tuned corpus, and it should be easy to revise
without a code change.
"""

import re
import tomllib
from functools import lru_cache
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
    keyword_strong_hit_threshold: int
    max_avg_word_chars: int
    chars_per_word_estimate: int
    currency_symbols: tuple
    keywords: tuple


class RouterDecision(NamedTuple):
    complexity_flag: bool
    reason: str


def _validate_positive_int(value, name):
    # >= 1, not >= 0: a count threshold of 0 silently degenerates that
    # signal to always-true (e.g. digits >= 0 is always True), which
    # defeats the point of having the signal at all.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RouterError(f"router config {name!r} must be a positive int (>= 1), got {value!r}")
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
        "keyword_strong_hit_threshold",
        "max_avg_word_chars",
        "chars_per_word_estimate",
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
        keyword_strong_hit_threshold=values["keyword_strong_hit_threshold"],
        max_avg_word_chars=values["max_avg_word_chars"],
        chars_per_word_estimate=values["chars_per_word_estimate"],
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


@lru_cache(maxsize=1)
def _cached_default_config():
    """The packaged [router] config, loaded from disk exactly once.

    classify_complexity is called on every single request (vision.py never
    passes an explicit config), so re-parsing models.toml from disk each
    time - as the OpenRouter client used to do for the model registry -
    means one file open per request for a file that never changes at
    runtime. Cached here instead; tests that need a fresh/different config
    pass one explicitly via classify_complexity's `config` parameter (which
    bypasses this cache entirely) or call load_router_config(path) directly.
    """
    return load_router_config()


def _currency_hit_count(text, currency_symbols):
    count = 0
    for symbol in currency_symbols:
        count += len(re.findall(re.escape(symbol) + r"\d", text))
    return count


def _keyword_hit_count(text_lower, keywords):
    # Word-boundary matching, not plain substring `in`: unanchored
    # substrings produce absurd matches ("bill" inside "Billings", "tax"
    # inside "taxi", "due" inside "residue"). \b works for multi-word
    # keywords too (e.g. "invoice number") since it only anchors the two
    # ends of the phrase.
    return sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", text_lower))


def classify_complexity(ocr_output, scene_context, config=None):
    """Decide complexity_flag from ocr_output (scene_context is unused - see
    module docstring "THE CHOSEN RULE" for why).

    Pure, deterministic, no network/model call: same input always yields
    the same RouterDecision. Returns (complexity_flag, reason) so callers
    (and tests) can see WHY a given input routed where it did without a
    state schema change - state stays at exactly 7 keys.

    scene_context stays in the signature (callers already pass it, and a
    future issue may find a legitimate, deliberately-separate use for it)
    but it does not participate in scoring.
    """
    if config is None:
        config = _cached_default_config()

    ocr_output = ocr_output or ""
    scene_context = scene_context or ""

    char_count = len(ocr_output)
    word_count = len(ocr_output.split())
    # Whitespace-based word counting silently collapses scripts that don't
    # use spaces between words (e.g. CJK) into a single "word", so a long
    # CJK document falls through the long-document fallback. Detect that
    # by looking at the average characters-per-token: a plausible
    # whitespace-separated word count and a huge average length are
    # mutually implausible only in this whitespace-free case; fall back to
    # a character-based estimate when it happens.
    avg_word_chars = char_count / word_count if word_count > 0 else char_count
    if avg_word_chars > config.max_avg_word_chars:
        word_count = char_count // config.chars_per_word_estimate

    digit_count = sum(ch.isdigit() for ch in ocr_output)
    currency_count = _currency_hit_count(ocr_output, config.currency_symbols)

    keyword_hits = _keyword_hit_count(ocr_output.lower(), config.keywords)

    digit_signal = digit_count >= config.digit_count_threshold
    currency_signal = currency_count >= config.currency_count_threshold
    keyword_signal = keyword_hits >= config.keyword_hit_threshold
    signal_score = sum((digit_signal, currency_signal, keyword_signal))

    # A prescription label's evidence is almost entirely vocabulary
    # ("TABLETS", "MG", "REFILLS") with no reliable digit/currency density,
    # so two or more distinct keyword hits alone are enough - see module
    # docstring.
    keyword_strong_signal = keyword_hits >= config.keyword_strong_hit_threshold

    long_document = word_count >= config.word_count_threshold
    complexity_flag = signal_score >= config.signal_score_threshold or long_document or keyword_strong_signal

    reason = (
        f"word_count={word_count} (long_document>={config.word_count_threshold}: {long_document}); "
        f"digits={digit_count} (>={config.digit_count_threshold}: {digit_signal}); "
        f"currency_hits={currency_count} (>={config.currency_count_threshold}: {currency_signal}); "
        f"keyword_hits={keyword_hits} (>={config.keyword_hit_threshold}: {keyword_signal}, "
        f"strong>={config.keyword_strong_hit_threshold}: {keyword_strong_signal}); "
        f"signal_score={signal_score}/{config.signal_score_threshold} "
        f"-> {'research' if complexity_flag else 'fast_synth'}"
    )

    return RouterDecision(complexity_flag=complexity_flag, reason=reason)
