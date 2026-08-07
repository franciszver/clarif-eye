"""Clarif-Eye Gradio app (issue #13 / P4.1): thin launcher, deployed on
Render's free tier (issue #22 / P8.2).

All testable logic lives in clarif_eye.ui - this file only builds the
Blocks layout (via build_interface()) and wires it to build_resources(),
which is what keeps the app importable and checkable without ever starting
a server (see tests/test_ui.py, tests/test_accessibility.py).

RENDER'S PORT (issue #22): Render assigns the network port through the
PORT environment variable and requires the process to listen on 0.0.0.0.
Gradio's own default, 127.0.0.1:7860, is not reachable from outside the
container, so the deployed service would fail its health check without
this. resolve_bind_address() below reads PORT and is a plain function so
it can be tested without launching a server (see tests/test_deploy.py);
when PORT is not set (local development), it returns (None, None) so
demo.launch() falls back to Gradio's own defaults, unchanged from before.
"""

import os

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


def resolve_bind_address():
    """Return the (host, port) to launch with.

    Render sets PORT and requires the process to listen on 0.0.0.0 (see
    this module's docstring). When PORT is set, return ("0.0.0.0", that
    port as an int). When it is not set, return (None, None) so
    demo.launch() falls back to Gradio's own defaults (127.0.0.1:7860),
    exactly as it did before this issue - `python app.py` still works the
    same way locally. Never hardcodes a port number: whatever value Render
    assigns is read from the environment, not assumed.
    """
    port = os.environ.get("PORT")
    if port is None:
        return None, None
    return "0.0.0.0", int(port)


if __name__ == "__main__":
    # head=ARIA_LIVE_HEAD (issue #15 / P5.1): marks the status control as
    # an ARIA live region - see clarif_eye.ui.ARIA_LIVE_HEAD's docstring
    # for why this must be passed to launch() rather than build_interface()
    # itself (Gradio 6.0 moved `head` off the Blocks constructor, and
    # build_interface() must stay launch-free for tests).
    server_name, server_port = resolve_bind_address()
    demo.launch(head=ARIA_LIVE_HEAD, server_name=server_name, server_port=server_port)
