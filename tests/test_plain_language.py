"""Enforces the plain-language skill (.claude/skills/plain-language/SKILL.md)
on the text this project actually SHIPS: the user-facing string constants in
clarif_eye.ui (status messages, labels, HOW_IT_WORKS_MARKDOWN, the diagram
label/description), the static labels gr.Blocks renders in build_interface(),
docs/ACCESSIBILITY.md, and docs/SCENARIOS.md.

WHAT THIS FILE DOES NOT CHECK, ON PURPOSE:
  - Tone, or whether the prose is good. That is a judgment call, not a
    regex's job.
  - Anything only a human reading aloud can judge (the skill's own test,
    "The test": read it aloud). No automated check replaces that.
  - Code identifiers, test names, log lines, or the long explanatory code
    comments in clarif_eye.ui: those are for maintainers, not the end user,
    and they legitimately quote the tells (em dashes, "not X, it's Y", etc.)
    while explaining fixes and history. Checking full source files would
    flag that legitimate maintainer prose.
  - .claude/skills/ or prd/: out of scope per the issue, and the skill file
    necessarily CONTAINS the strike lists it defines.

Only the SHIPPED strings above are scanned - never clarif_eye/ui.py's raw
source text, so a code comment discussing "em dashes" or "not X, it's Y" as
an example can never trip these checks.

DROPPED FROM THE VOCABULARY LIST (see SKILL.md's list), because they produce
real false positives on the current, already-clean shipped text:
  - "real": used factually throughout docs/ACCESSIBILITY.md ("a real
    browser", "a real screen reader", "a real camera") to distinguish
    machine-verified claims from human-verified ones - exactly the honesty
    the skill asks for, not filler.
  - "actually": used the same way ("what the code actually does", "the UI
    actually renders") to contrast a claim against verified behavior.
  - "navigate": the skill's own list marks this "(figurative)" as the tell;
    this project's one shipped use is literal ("navigate by heading" in a
    screen reader), which the list explicitly does not target and a bare
    word match cannot tell apart from the figurative case.

ALSO NOTE: the "Audio is playing" regression check (rule 7) is scoped to
the live UI string constants and rendered labels only, not
docs/ACCESSIBILITY.md. That document legitimately quotes the old, broken
wording in past tense while explaining the #47 fix ("no longer claims audio
'is playing' as a fact") - that is honest history, not a live claim, and
scanning the whole doc would flag the sentence that explains the fix as if
it were the bug.
"""

import re
from pathlib import Path

import gradio as gr

from clarif_eye import ui

REPO_ROOT = Path(__file__).resolve().parent.parent

EM_DASH = "—"

# Vocabulary strike-list, from SKILL.md's "Vocabulary to strike" section,
# minus the three words dropped above (see module docstring for why).
STRIKE_WORDS = [
    "delve", "intricate", "tapestry", "pivotal", "underscore", "landscape",
    "foster", "testament", "enhance", "crucial", "robust", "seamless",
    "comprehensive", "leverage", "realm", "showcase", "spearhead", "vital",
    "essential", "myriad", "plethora", "resonate", "unlock", "elevate",
    "transformative", "cutting-edge", "state-of-the-art", "dynamic",
    "synergy", "boasts", "commendable", "surpass", "primarily", "meticulous",
    "quietly", "shapes how", "this matters because", "a shift in", "lands",
    "earn", "the work", "hold space", "compound", "send the signal",
]

# "It's not X, it's Y." / "isn't X, it's Y." - SKILL.md calls this "the most
# durable AI tell on record". Scoped to the comma-linked construction the
# skill names, not a bare "not" (which is ordinary English and appears
# constantly in honest, factual writing - see the dropped-words note above
# for the same false-positive concern).
NEGATIVE_PARALLELISM_RE = re.compile(
    r"\b(?:isn'?t|is not)\b[^.!?\n]{0,80},\s*(?:it'?s|it is)\b", re.IGNORECASE
)

NOT_ONLY_BUT_ALSO_RE = re.compile(r"\bnot only\b.{0,80}?\bbut also\b", re.IGNORECASE | re.DOTALL)

# Connective chains / compulsive summary: SKILL.md's tell is these words
# opening a line (paragraph or sentence start), not appearing mid-sentence
# ("furthermore" inside a quoted sentence is not the pattern being struck).
CONNECTIVE_CHAIN_RE = re.compile(r"^(Moreover|Furthermore|Additionally)\b", re.MULTILINE)
COMPULSIVE_SUMMARY_RE = re.compile(r"^(Overall|In conclusion|In summary)\b", re.MULTILINE)

# The one product-copy trap the issue calls out by name: "Audio is playing"
# shipped and was false whenever a browser blocked autoplay (fixed under
# #47; see STATUS_SUCCESS_AUDIO's comment in ui.py and the "Audio talked
# over the announcement (#47)" entry in docs/ACCESSIBILITY.md). No status
# string may assert that audio is playing, because the code cannot
# guarantee it.
AUDIO_IS_PLAYING_RE = re.compile(r"\bis playing\b", re.IGNORECASE)


def _ui_string_constants():
    """The module-level user-facing string constants in clarif_eye.ui."""
    names = [
        "NO_IMAGE_MESSAGE",
        "CONFIG_ERROR_MESSAGE",
        "UNREADABLE_IMAGE_MESSAGE",
        "UNEXPECTED_ERROR_MESSAGE",
        "AUDIO_UNAVAILABLE_NOTE",
        "STATUS_IDLE",
        "STATUS_WORKING",
        "STATUS_SUCCESS_AUDIO",
        "STATUS_SUCCESS_TEXT_ONLY",
        "STATUS_DEGRADED",
        "HOW_IT_WORKS_MARKDOWN",
        "PIPELINE_DIAGRAM_LABEL",
        "PIPELINE_DIAGRAM_DESCRIPTION",
        "UPLOADED_PHOTO_ALT",
    ]
    return {f"clarif_eye.ui.{name}": getattr(ui, name) for name in names}


def _ui_interface_strings():
    """Static text gr.Blocks actually renders: the intro Markdown, and every
    control's label/button text - built via build_interface() (same fake,
    no-network, no-launch pattern as tests/test_accessibility.py) so this
    reads the real rendered strings rather than a hand-copied guess at what
    build_interface() contains.
    """

    class _NoopResources:
        """build_interface() never touches its `resources` arg until the
        submit button is actually clicked (see ui._submit's closure) - a
        bare placeholder is enough to build the Blocks tree."""

    demo = ui.build_interface(_NoopResources())
    try:
        texts = {}
        for index, component in enumerate(demo.blocks.values()):
            if isinstance(component, (gr.Markdown, gr.HTML)) and isinstance(component.value, str):
                texts[f"ui.build_interface()[{index}]:{type(component).__name__}.value"] = component.value
            label = getattr(component, "label", None)
            if isinstance(label, str) and label:
                texts[f"ui.build_interface()[{index}]:{type(component).__name__}.label"] = label
            if isinstance(component, gr.Button) and isinstance(component.value, str):
                texts[f"ui.build_interface()[{index}]:Button.value"] = component.value
        return texts
    finally:
        demo.close()


def _accessibility_doc_text():
    path = REPO_ROOT / "docs" / "ACCESSIBILITY.md"
    return {"docs/ACCESSIBILITY.md": path.read_text(encoding="utf-8")}


def _scenarios_doc_text():
    path = REPO_ROOT / "docs" / "SCENARIOS.md"
    return {"docs/SCENARIOS.md": path.read_text(encoding="utf-8")}


def shipped_texts():
    """Every piece of text in scope for the plain-language guard: the
    user-facing ui.py string constants, the static labels/markdown
    build_interface() renders, docs/ACCESSIBILITY.md, and
    docs/SCENARIOS.md. Returns {name: text} so failures can name exactly
    where the offending text lives.
    """
    texts = {}
    texts.update(_ui_string_constants())
    texts.update(_ui_interface_strings())
    texts.update(_accessibility_doc_text())
    texts.update(_scenarios_doc_text())
    return texts


# --- Checks -----------------------------------------------------------------
#
# Each check scans every shipped text and reports, per failure, WHERE
# (the name from shipped_texts()) and WHAT (the offending snippet), plus what
# to do instead - so a failure is actionable without re-reading this file.


def test_no_em_dashes_in_shipped_text():
    """SKILL.md rule 1: no em dashes anywhere in shipped text. A comma, a
    full stop, or a parenthetical does the job instead."""
    failures = []
    for name, text in shipped_texts().items():
        if EM_DASH in text:
            index = text.index(EM_DASH)
            snippet = text[max(0, index - 30) : index + 30]
            failures.append(f"{name}: em dash found near {snippet!r}. Replace it with a comma, full stop, or parenthetical.")
    assert not failures, "\n".join(failures)


def test_no_vocabulary_strike_list_hits_in_shipped_text():
    """SKILL.md rule "Vocabulary to strike": every hit must be deleted or
    replaced with a plain word. Matched on word boundaries so this can't
    match inside a longer, unrelated word."""
    failures = []
    for name, text in shipped_texts().items():
        for word in STRIKE_WORDS:
            pattern = r"\b" + re.escape(word) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                failures.append(f"{name}: strike-list word {word!r} found. Delete it or replace with a plain word.")
    assert not failures, "\n".join(failures)


def test_no_negative_parallelism_in_shipped_text():
    """SKILL.md: "It's not X, it's Y." - the most durable AI tell on
    record. Delete the construction and state the thing directly."""
    failures = []
    for name, text in shipped_texts().items():
        match = NEGATIVE_PARALLELISM_RE.search(text)
        if match:
            failures.append(
                f"{name}: negative-parallelism construction {match.group(0)!r}. "
                "Delete the 'not X, it's Y' framing and just state the fact."
            )
    assert not failures, "\n".join(failures)


def test_no_not_only_but_also_in_shipped_text():
    """SKILL.md: "Not only X, but also Y." Split it into two sentences or
    drop half."""
    failures = []
    for name, text in shipped_texts().items():
        match = NOT_ONLY_BUT_ALSO_RE.search(text)
        if match:
            failures.append(f"{name}: 'not only ... but also' found in {match.group(0)!r}. Split it or drop half.")
    assert not failures, "\n".join(failures)


def test_no_connective_chains_at_line_start_in_shipped_text():
    """SKILL.md: "Moreover, Furthermore, Additionally" stacked across
    paragraphs. Cut the connective and let the sentence stand on its own."""
    failures = []
    for name, text in shipped_texts().items():
        match = CONNECTIVE_CHAIN_RE.search(text)
        if match:
            failures.append(f"{name}: line starts with connective {match.group(0)!r}. Cut it.")
    assert not failures, "\n".join(failures)


def test_no_compulsive_summary_at_line_start_in_shipped_text():
    """SKILL.md: a closing line starting "Overall" or "In conclusion" that
    restates what was just said. Cut it."""
    failures = []
    for name, text in shipped_texts().items():
        match = COMPULSIVE_SUMMARY_RE.search(text)
        if match:
            failures.append(f"{name}: line starts with compulsory-summary opener {match.group(0)!r}. Cut it.")
    assert not failures, "\n".join(failures)


def test_no_status_string_claims_audio_is_playing():
    """Regression test for the one product-copy trap the issue names by
    history: "Audio is playing" shipped and was false whenever a browser
    blocked autoplay (fixed under #47 - see STATUS_SUCCESS_AUDIO's comment
    in clarif_eye/ui.py). No LIVE status string may assert playback as an
    accomplished fact, since the code cannot guarantee it: say only what is
    true in every branch (e.g. "Description ready.").

    Scoped to the actual UI string constants and rendered labels, NOT
    docs/ACCESSIBILITY.md: that document legitimately quotes the old,
    broken wording ("Audio is playing; the text is below too.") and says,
    in plain past tense, that the current wording "no longer claims audio
    'is playing' as a fact" - honest history, not a live claim. Scanning
    the whole doc would flag that honesty as if it were the bug it
    describes fixing.
    """
    failures = []
    live_texts = {
        name: text for name, text in shipped_texts().items() if name != "docs/ACCESSIBILITY.md"
    }
    for name, text in live_texts.items():
        match = AUDIO_IS_PLAYING_RE.search(text)
        if match:
            failures.append(
                f"{name}: claims audio {match.group(0)!r} as fact, but a browser can block autoplay. "
                "State only what is true in every branch (e.g. 'Description ready.')."
            )
    assert not failures, "\n".join(failures)
