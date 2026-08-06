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
    scraper_data: str
    final_output: str
    audio_file_path: str


def make_initial_state(image_data):
    """Build the initial state from `image_data`, every other key at its empty default."""
    return ClarifEyeState(
        image_data=image_data,
        ocr_output="",
        scene_context="",
        complexity_flag=False,
        scraper_data="",
        final_output="",
        audio_file_path="",
    )
