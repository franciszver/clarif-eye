"""Clarif-Eye Gradio app (issue #13 / P4.1): thin Hugging Face Spaces launcher.

All testable logic lives in clarif_eye.ui - this file only builds the
Blocks layout (via build_interface()) and wires it to build_resources(),
which is what keeps the app importable and checkable without ever starting
a server (see tests/test_ui.py, tests/test_accessibility.py). `app.py` at
the repo root is the Spaces convention for what to run.
"""

from clarif_eye.ui import ARIA_LIVE_HEAD, build_interface, build_resources

# Built ONCE for the life of this process (issue #13's core requirement):
# one OpenRouterClient, one TTS provider chain, one research
# searcher/client, injected into every request instead of each node
# constructing its own. See clarif_eye.ui's module docstring.
_resources = build_resources()

demo = build_interface(_resources)

# Enables Gradio's own queueing/progress feedback (D16): a request can take
# up to ~27s on the research path and the user has no other way to know
# work is happening.
demo.queue()

if __name__ == "__main__":
    # head=ARIA_LIVE_HEAD (issue #15 / P5.1): marks the status control as
    # an ARIA live region - see clarif_eye.ui.ARIA_LIVE_HEAD's docstring
    # for why this must be passed to launch() rather than build_interface()
    # itself (Gradio 6.0 moved `head` off the Blocks constructor, and
    # build_interface() must stay launch-free for tests).
    demo.launch(head=ARIA_LIVE_HEAD)
