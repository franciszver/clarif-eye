"""Graph state schema for Clarif-Eye.

Exactly the keys from the architecture doc's pipeline
(vision -> router -> fast synth or research + analysis -> tts). Every key
is present from the start via make_initial_state; nodes fill them in, they
never invent new keys later.
"""

from typing import TypedDict


class ClarifEyeState(TypedDict):
    image_data: str
    ocr_output: str
    scene_context: str
    complexity_flag: bool
    # Stays "" forever on the fast path - that's expected, not an error.
    # But that also means "" is indistinguishable from "research ran and
    # found nothing" on the deep path. Nothing currently disambiguates the
    # two cases; issues #7/#8 must decide whether a sentinel is needed.
    scraper_data: str
    final_output: str
    audio_file_path: str


def make_initial_state(image_data):
    """Build the initial state from `image_data`, every other key at its empty default."""
    if image_data is None or not str(image_data).strip():
        raise ValueError(
            "make_initial_state: image_data is required and must be a "
            "non-empty, non-blank string"
        )
    return ClarifEyeState(
        image_data=image_data,
        ocr_output="",
        scene_context="",
        complexity_flag=False,
        scraper_data="",
        final_output="",
        audio_file_path="",
    )
