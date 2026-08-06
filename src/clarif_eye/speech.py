"""Shared text-for-speech logic (P1.4 review fix, issue #8 will reuse this).

The output of `to_spoken_text` is READ ALOUD to a blind user who cannot
cross-check it against the original text. That makes silent content
alteration the worst failure mode this module can have - worse than an
awkward sentence, worse than an error message getting through unsanitised.
Every rule here is written to either strip markup/tags that are not meant
to be heard, or leave genuine content untouched; it must never "helpfully"
delete characters that happen to look like markup but are actually part of
the content (a filename, a password, a piece of arithmetic).

Both vision.py and synth.py import from here instead of each other -
`_strip_code_fence` used to be a private import from vision.py into
synth.py; that coupling is gone. Only the shared mechanics live here. Each
node keeps its own degradation *messages* local to itself.
"""

import re

# --- Code fences ----------------------------------------------------------


def strip_code_fence(text):
    """Strip a leading/trailing ``` fence line (and language tag) if present."""
    lines = text.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


# --- Sanitisation for text-to-speech ---------------------------------------

# HTML tags: "<" only counts as a tag start when immediately followed by a
# letter (an opening tag) or "/" (a closing tag), and the whole thing must
# close with ">" before the next "<". This means a genuine less-than sign
# used in prose (e.g. "5 < 10", which has no matching ">") is never touched.
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^<>]*>")

# Markdown links: [text](url) - keep the link TEXT, drop the target
# entirely. Must run BEFORE the bare-URL substitution below, otherwise the
# URL inside the parens gets replaced first and leaves a dangling "(".
_MD_INLINE_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\n]*)\)")
# Reference-style links: [text][ref] - keep the text, drop the reference.
_MD_REF_LINK_RE = re.compile(r"\[([^\]\n]+)\]\[[^\]\n]*\]")

# A line that starts (after whitespace) with a markdown heading marker.
_HEADING_RE = re.compile(r"^\s*#+\s*", re.MULTILINE)
# A line that starts (after whitespace) with a bullet marker.
_BULLET_RE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
# A line that starts (after whitespace) with a numbered-list marker.
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)

# Markdown emphasis: only strip the marker characters when they form a
# PAIRED marker around real content (no whitespace touching the marker on
# the inside, so "* 3 *" is not emphasis). A lone `_` or `*` sitting inside
# or between words (my_file_name.txt, WIFI_PASSWORD_2026, 5 * 3 = 15) never
# matches any of these and is left completely untouched. Bold (**/__) is
# matched before italic (*/_) so "**bold**" collapses in one pass rather
# than leaving stray single markers for the italic rules to (mis)handle.
_BOLD_STAR_RE = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*")
_BOLD_UNDERSCORE_RE = re.compile(r"(?<!\w)__(?!\s)(.+?)(?<!\s)__(?!\w)")
_ITALIC_STAR_RE = re.compile(r"\*(?!\s)([^*\n]+?)(?<!\s)\*")
# Underscore italics additionally require a non-word character (or
# start/end of string) immediately outside the pair, matching how markdown
# treats underscores intraword as literal text, not emphasis.
_ITALIC_UNDERSCORE_RE = re.compile(r"(?<!\w)_(?!\s)([^_\n]+?)(?<!\s)_(?!\w)")

# A markdown table separator row, e.g. "|------|-------|" or "---|---" or
# ":--|--:". Detected as: contains a pipe, contains a dash, and every other
# character is a pipe/dash/colon/whitespace.
_TABLE_SEPARATOR_CHARS_RE = re.compile(r"^[|:\-\s]*$")

_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "]"
)
# A punctuation character repeated 3+ times in a row (e.g. "!!!", "----",
# "..."), collapsed to a single occurrence rather than read aloud as noise.
_PUNCT_RUN_RE = re.compile(r"([^\w\s])\1{2,}")


def _is_table_separator_row(line):
    stripped = line.strip()
    return (
        "|" in stripped
        and "-" in stripped
        and bool(_TABLE_SEPARATOR_CHARS_RE.match(stripped))
    )


def _transform_table_rows(text):
    """Turn markdown table rows into connected spoken phrases.

    A run of bare commas ("Item , Price , , Burger , $8.00 ,") reads aloud
    as dead pauses with no sense of rows or labels, and this node's primary
    input is receipts/menus/labels, so tables are not an edge case. Each
    data row's cells are joined with ", " (empty cells dropped), and the
    row ends with a "." so consecutive rows read as separate sentences
    instead of running together. The "|---|---|" separator row is dropped
    entirely - it carries no spoken content.
    """
    out_lines = []
    for line in text.split("\n"):
        if "|" not in line:
            out_lines.append(line)
            continue
        if _is_table_separator_row(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        cells = [c for c in cells if c]
        if not cells:
            continue
        out_lines.append(", ".join(cells) + ".")
    return "\n".join(out_lines)


def to_spoken_text(text):
    """Strip/normalise markup so `text` is safe to hand to text-to-speech.

    Defensive, not cosmetic: models routinely ignore "plain prose only"
    instructions in prompts, and OCR'd source material can itself contain
    HTML or markdown, so this runs on every reply regardless of what was
    asked for. The overriding rule is that ordinary content - filenames,
    passwords, product codes, arithmetic - must survive completely
    unchanged; only actual markup/tag syntax is removed.
    """
    text = strip_code_fence(text)
    text = _HTML_TAG_RE.sub("", text)
    text = text.replace("`", "")
    text = _MD_INLINE_LINK_RE.sub(r"\1", text)
    text = _MD_REF_LINK_RE.sub(r"\1", text)
    text = _transform_table_rows(text)
    text = _HEADING_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _NUMBERED_RE.sub("", text)
    text = _BOLD_STAR_RE.sub(r"\1", text)
    text = _BOLD_UNDERSCORE_RE.sub(r"\1", text)
    text = _ITALIC_STAR_RE.sub(r"\1", text)
    text = _ITALIC_UNDERSCORE_RE.sub(r"\1", text)
    text = text.replace("|", ", ")
    text = _URL_RE.sub("a web link", text)
    text = _EMOJI_RE.sub("", text)
    text = _PUNCT_RUN_RE.sub(r"\1", text)
    # Flatten to a single line of flowing prose: a list turned into
    # separate short lines above should read as one continuous script, not
    # be read aloud with unnatural pauses between fragments.
    lines = [line.strip() for line in text.split("\n")]
    text = " ".join(line for line in lines if line)
    text = re.sub(r"\s+", " ", text).strip()
    # A leftover ", ," run or a leading/trailing comma from a dropped empty
    # cell is not something a listener should hear as a pause.
    text = re.sub(r",(\s*,)+", ",", text)
    text = re.sub(r"^\s*,\s*", "", text)
    text = re.sub(r"\s*,\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Nothing but leftover punctuation/whitespace (e.g. a reply that was
    # pure markup noise) is not speakable content - treat it the same as
    # an empty reply rather than reading stray commas and pipes aloud.
    if not re.search(r"\w", text):
        return ""
    return text
