"""Clarif-Eye Gradio app (issue #13 / P4.1): thin Hugging Face Spaces launcher.

All testable logic lives in clarif_eye.ui - this file only builds the
Blocks layout and wires it to build_resources()/handle_submit(), which is
what keeps the app importable and checkable without ever starting a
server (see tests/test_ui.py). `app.py` at the repo root is the Spaces
convention for what to run.
"""

import gradio as gr

from clarif_eye.ui import build_resources, handle_submit

# Built ONCE for the life of this process (issue #13's core requirement):
# one OpenRouterClient, one TTS provider chain, one research
# searcher/client, injected into every request instead of each node
# constructing its own. See clarif_eye.ui's module docstring.
_resources = build_resources()


def _submit(image):
    return handle_submit(image, _resources)


with gr.Blocks(title="Clarif-Eye") as demo:
    gr.Markdown(
        "# Clarif-Eye\n"
        "Clarif-Eye describes a photo aloud for visually impaired users. "
        "Take or upload a photo below. This can take up to about 30 "
        "seconds, especially for photos with dense text."
    )
    image_input = gr.Image(
        label="Photo to describe",
        sources=["upload", "webcam"],
        type="pil",
    )
    submit_button = gr.Button("Describe this photo", variant="primary")
    audio_output = gr.Audio(label="Spoken description", autoplay=True)
    text_output = gr.Textbox(label="Description (text)", lines=6)

    submit_button.click(
        fn=_submit,
        inputs=image_input,
        outputs=[audio_output, text_output],
    )

# Enables Gradio's own queueing/progress feedback (D16): a request can take
# up to ~27s on the research path and the user has no other way to know
# work is happening.
demo.queue()

if __name__ == "__main__":
    demo.launch()
