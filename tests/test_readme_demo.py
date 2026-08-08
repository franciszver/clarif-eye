"""The README must embed the live-run demo media: an animated GIF plus a
narrated MP4, both checked into docs/ (issue #106).

A README that merely mentions the files by name without them existing (or
existing but empty/truncated) is not a real demo, so this test checks both
that the README embeds/links them as well-formed markdown and that the
assets themselves are present and non-trivial in size.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
GIF_PATH = REPO_ROOT / "docs" / "demo.gif"
MP4_PATH = REPO_ROOT / "docs" / "demo.mp4"

MIN_SIZE_BYTES = 10_000


def test_readme_references_demo_media():
    text = README_PATH.read_text(encoding="utf-8")
    # The lookarounds reject backtick-wrapped (inline-code) mentions specifically;
    # fenced code blocks are out of scope for this tripwire.
    assert re.search(r"(?<!`)!\[[^\]]+\]\(docs/demo\.gif\)(?!`)", text), (
        "README.md does not embed docs/demo.gif as an image"
    )
    assert re.search(r"(?<!`)\[[^\]]+\]\(docs/demo\.mp4\)(?!`)", text), (
        "README.md does not link docs/demo.mp4"
    )


def test_demo_gif_exists_and_is_not_trivial():
    assert GIF_PATH.exists(), f"{GIF_PATH} does not exist"
    assert GIF_PATH.stat().st_size > MIN_SIZE_BYTES, (
        f"{GIF_PATH} is only {GIF_PATH.stat().st_size} bytes"
    )


def test_demo_mp4_exists_and_is_not_trivial():
    assert MP4_PATH.exists(), f"{MP4_PATH} does not exist"
    assert MP4_PATH.stat().st_size > MIN_SIZE_BYTES, (
        f"{MP4_PATH} is only {MP4_PATH.stat().st_size} bytes"
    )
