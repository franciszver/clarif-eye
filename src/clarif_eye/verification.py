"""Checking that numbers spoken aloud trace back to what the camera saw.

MOVED HERE FROM analysis.py, unchanged (issue #82 / P9.3's simplify gate).
It was never analysis-specific: it takes text in and text out, knows nothing
about prompts, ladders, or nodes, and it already had three importers outside
analysis.py (scripts/benchmark_ladders.py, scripts/benchmark_pipeline.py,
and tests/test_analysis_fixture_replay.py) all reaching through a private
name on a node module to get at it. Living in its own module says what it
actually is: a verification primitive any caller may use, including a future
one that is not the analysis node.

analysis.py keeps importing it under its old private name so every existing
importer of `clarif_eye.analysis._numbers_verified` keeps working - this
move changes no behavior and no call site.

WHO ENFORCES IT IS A SEPARATE QUESTION from where it lives. Today only the
deep-analysis path calls it (see analysis.run_analysis); synth.py and
followup.py do not. Whether follow-up answers should be held to this check
is a real product decision, not an oversight - it is tracked as issue #92,
to be settled alongside #83's ask-first mechanism, because the honest fix
for "the answer contains a number I cannot verify" is probably to ask rather
than to refuse. See followup.py's module docstring for the argument.
"""

import re

# Number-like tokens (amounts, dates-as-digits, identifiers) that a spoken
# script must be able to trace back to the source material. Deliberately
# loose - it is a token-equality check, not a parser - because the goal is
# to catch INVENTED numbers, not to validate formatting.
_NUMBER_TOKEN_RE = re.compile(r"\$?\d[\d,\-./:]*\d|\d")

# Separators that join multi-part identifiers (an account number like
# "4471-2205-88", a time like "10:30") rather than a decimal point. Used
# below to also register each digit-run of such an identifier as its own
# verifiable token, so a model that legitimately speaks one part of an
# identifier ("4471") still verifies. "." is deliberately excluded: it is
# how decimal amounts are written, and registering "104" as a stand-in for
# "104.95" would let a truncated dollar amount slip back through - exactly
# the leniency this check exists to close.
_IDENTIFIER_SPLIT_RE = re.compile(r"[-/:]")


def _strip_currency_punct(text):
    return text.replace("$", "").replace(",", "")


def _input_number_tokens(ocr_output, scene_context, scraper_data):
    """Whole number-like tokens (plus identifier sub-parts) from the inputs.

    Each token is a value from the source text taken as a whole - not a
    substring window into it - so "104.9" cannot pass by being contained in
    "104.95". For tokens that are hyphen/slash/colon-separated identifiers,
    the individual digit-runs are also added (see _IDENTIFIER_SPLIT_RE)
    so a partial identifier mention still verifies.
    """
    haystack = f"{ocr_output} {scene_context} {scraper_data}"
    tokens = set()
    for raw in _NUMBER_TOKEN_RE.findall(haystack):
        token = _strip_currency_punct(raw)
        tokens.add(token)
        for part in _IDENTIFIER_SPLIT_RE.split(token):
            if part:
                tokens.add(part)
    return tokens


def unverified_numbers(spoken_output, ocr_output, scene_context, scraper_data):
    """The number-like tokens in `spoken_output` that do NOT trace back to
    the inputs - in the order they were spoken, with duplicates collapsed.

    THE STRUCTURAL HALF of this module (issue #83 / P9.4). numbers_verified
    below answers "may this be spoken?" with a bool, which is all the
    analysis node ever needed while the only response to a failure was to
    refuse. Now that a failure can be turned into a QUESTION to the user
    ("this number could not be checked - continue anyway?"), the caller
    needs to know WHICH tokens failed, as tokens, so nothing downstream has
    to parse them back out of a sentence. Same comparison rules as
    numbers_verified - which is now defined in terms of this function, so
    the two can never disagree about what counts as verified.

    Tokens are returned in their ORIGINAL spoken form ("$999.99" keeps its
    currency symbol) rather than the normalised form used for comparison:
    what comes back here is read aloud to a user, so it should sound like
    what they were about to be told.
    """
    tokens = _NUMBER_TOKEN_RE.findall(spoken_output)
    if not tokens:
        return []
    input_tokens = _input_number_tokens(ocr_output, scene_context, scraper_data)
    failing = []
    for token in tokens:
        if _strip_currency_punct(token) in input_tokens:
            continue
        if token not in failing:
            failing.append(token)
    return failing


def numbers_verified(spoken_output, ocr_output, scene_context, scraper_data):
    """Check that every number-like token spoken aloud traces back to the input text.

    THE CENTRAL RISK (see analysis.py's module docstring) is a model
    inventing a plausible-sounding amount, date, or identifier that a blind
    user cannot check. The prompt asks the model not to do this, but a
    prompt is not enforcement - this is the code-level backstop: every
    numeric token found in `spoken_output` must EQUAL a whole number token
    from the combined inputs the model was actually given (see
    _input_number_tokens) - not merely appear as a substring of one, which
    would let a truncated amount like "104.9" pass by virtue of being
    contained in "104.95". Comparison is lenient (currency symbols and
    commas stripped from both sides) so "$104.95" in the output matches
    "104.95" in the OCR text. A reply with no numeric tokens at all has
    nothing to verify and trivially passes.

    Defined in terms of unverified_numbers above (issue #83 / P9.4) rather
    than repeating the comparison: two copies of this rule would be two
    places for "what counts as verified" to drift, and the whole point of
    the ask-first flow is that the question the user is asked names exactly
    the tokens this predicate rejected.
    """
    return not unverified_numbers(spoken_output, ocr_output, scene_context, scraper_data)
