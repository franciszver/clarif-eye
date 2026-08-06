"""MANUAL-ONLY accessibility audit for the Clarif-Eye Gradio UI (issue #16 / P5.2).

DO NOT RUN AS PART OF CI OR AN AGENT SESSION. This script launches nothing
itself - it takes a URL for an ALREADY-RUNNING app (started by a human or
the orchestrator) and prints a checklist plus a small JS payload to paste
into that app's already-open DevTools console. It never opens a browser,
never makes a network call, and never starts a server.

WHY A CHECKLIST INSTEAD OF A HEADLESS-BROWSER AUDIT: pyproject.toml
declares no browser-automation dependency (no playwright/selenium/etc.),
and this project's dependency posture has stayed deliberately minimal
(CLAUDE.md "Simplicity First" / "No new runtime dependency" per issue
#16's scope). Adding a ~200MB browser-automation dependency just to run
this one audit was rejected in favour of reusing the DevTools session the
orchestrator already has open for the live P5.1 checks. If a
browser-automation dependency is ever added to this project for other
reasons, this script can be upgraded to drive it directly - the CHECKS
list below is written to stay meaningful either way.

WHAT THIS CHECKS (issue #16's minimum set):
  1. accessible-names       - every interactive element has a non-empty
                               accessible name.
  2. live-region            - the status control carries aria-live/role.
  3. description-focusable  - the description output is focusable and in
                               the tab order.
  4. no-positive-tabindex   - no element anywhere uses a positive
                               tabindex (only 0 or none).
  5. image-alt-text         - images have alt text, or are marked
                               decorative (alt="").
  6. color-not-sole-carrier - state changes are never conveyed by colour
                               alone (this app has no colour-coded status
                               indicators to begin with - the check exists
                               to catch a future regression).

Every item's elem_ids are imported from clarif_eye.ui directly (not
hand-copied strings) so this checklist cannot silently drift from what
the UI actually renders - see tests/test_accessibility.py's
test_audit_checklist_stays_in_sync_with_ui_elem_ids.

Usage (by a human or the orchestrator, against an already-running app):
    python scripts/audit_accessibility.py http://127.0.0.1:7860
"""

import sys

from clarif_eye.ui import RESULT_ELEM_ID, STATUS_ELEM_ID

CHECKS = [
    {
        "id": "accessible-names",
        "description": (
            "Every interactive element (image drop zone, Upload file button, "
            "Capture from camera button, 'Describe this photo' button, Status "
            "box, audio player, Description (text) box) has a non-empty "
            "accessible name."
        ),
        "how": (
            "Tab through the page and read the accessible name announced at "
            "each stop (or inspect each element's Accessibility pane in "
            "DevTools > Elements). Every stop must announce a name, never "
            "'button' or 'textbox' alone."
        ),
    },
    {
        "id": "live-region",
        "description": (
            f"The status control (#{STATUS_ELEM_ID}) carries "
            'aria-live="polite", aria-atomic="true", and role="status", '
            "and keeps carrying them after a submit re-renders it."
        ),
        "how": (
            f"Inspect #{STATUS_ELEM_ID} in DevTools > Elements before AND "
            "after clicking 'Describe this photo'; all three attributes "
            "must be present both times."
        ),
    },
    {
        "id": "description-focusable",
        "description": (
            f"The description output (#{RESULT_ELEM_ID} textarea) is "
            "focusable and reachable via Tab, in the order: Drop image / "
            "Upload file / Capture from camera / Describe this photo / "
            "Description (text) / Use via API / Built with Gradio / "
            "Settings."
        ),
        "how": (
            "Press Tab repeatedly from the top of the page and confirm the "
            "description textarea receives visible focus in that position, "
            "and that focus lands there automatically once a result is "
            "ready."
        ),
    },
    {
        "id": "no-positive-tabindex",
        "description": (
            "No element on the page has a positive tabindex (tabindex > 0)."
        ),
        "how": (
            "Run the JS payload below in the console; it lists any element "
            "with tabindex > 0. The list must be empty."
        ),
    },
    {
        "id": "image-alt-text",
        "description": (
            "Every <img> either has non-empty alt text or is explicitly "
            "marked decorative (alt=\"\")."
        ),
        "how": (
            "Run the JS payload below; it lists any <img> with alt=null "
            "(missing the attribute entirely). The list must be empty."
        ),
    },
    {
        "id": "color-not-sole-carrier",
        "description": (
            "No state (success/error/degraded/idle) is conveyed by colour "
            "alone - it must also be present in text. This app has no "
            "colour-coded indicators today; this check exists to catch a "
            "future regression."
        ),
        "how": (
            "Visually compare the idle, working, success, and degraded "
            "states: each must differ in TEXT (the status/description "
            "content), not merely in colour or an icon."
        ),
    },
]


def render_checklist(url: str) -> str:
    """Render the ordered manual checklist for a human/orchestrator to run
    against `url` (an already-running instance of this app)."""
    lines = [f"Accessibility audit checklist for {url}", ""]
    for i, check in enumerate(CHECKS, start=1):
        lines.append(f"{i}. [{check['id']}] {check['description']}")
        lines.append(f"   How: {check['how']}")
    return "\n".join(lines)


def js_payload() -> str:
    """A small, dependency-free JS snippet to paste into an already-open
    DevTools console against the running app. Reports positive tabindex
    values, missing accessible names, and images missing alt text -
    everything in CHECKS that a script can inspect without driving a
    browser itself."""
    return """
(function () {
  function getAccessibleName(el) {
    return (
      el.getAttribute("aria-label") ||
      (el.labels && el.labels[0] && el.labels[0].textContent) ||
      el.getAttribute("title") ||
      (el.textContent || "").trim() ||
      null
    );
  }

  const positiveTabindex = Array.from(document.querySelectorAll("[tabindex]"))
    .filter((el) => parseInt(el.getAttribute("tabindex"), 10) > 0);

  const unnamedInteractive = Array.from(
    document.querySelectorAll("button, input, textarea, [role='button']")
  ).filter((el) => !getAccessibleName(el));

  const imagesMissingAlt = Array.from(document.querySelectorAll("img")).filter(
    (el) => el.getAttribute("alt") === null
  );

  const statusEl = document.getElementById("status-live-region");
  const statusOk =
    !!statusEl &&
    statusEl.getAttribute("aria-live") === "polite" &&
    statusEl.getAttribute("role") === "status";

  console.log("positive tabindex elements:", positiveTabindex);
  console.log("unnamed interactive elements:", unnamedInteractive);
  console.log("images missing alt:", imagesMissingAlt);
  console.log("status live region correctly tagged:", statusOk);
})();
""".strip()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("Usage: python scripts/audit_accessibility.py <url>")
        return 2
    url = argv[0]
    print(render_checklist(url))
    print()
    print("Paste the following into DevTools > Console at that URL:")
    print()
    print(js_payload())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
