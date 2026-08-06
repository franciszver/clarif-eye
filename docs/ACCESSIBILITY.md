# Accessibility

Clarif-Eye is built for people who cannot see the photo they are describing.
Everyone who uses this app is assumed to be visually impaired, so what a
screen reader says matters as much as what the screen shows.

## What the product does for a screen-reader user

1. **A progress announcement.** As soon as you activate "Describe this
   photo", a live region announces: *"Photo received. Describing it now;
   this can take up to about 30 seconds."* This is announced politely (it
   will not interrupt you or steal focus) so you know the app registered
   your request instead of wondering if anything happened during the wait.
2. **A completion announcement.** When the result is ready, the same live
   region announces one of: description ready with audio playing,
   description ready as text only (if audio could not be produced), or a
   limited/degraded result. Focus is then moved to the description text so
   you land on the answer without hunting for it.
3. **A spoken description**, played automatically when audio synthesis
   succeeds.
4. **A text fallback**, always populated even when audio fails, readable
   and reachable by keyboard as a normal (read-only) text field.

## What has been verified, and how

### Machine-verified (automated tests, no browser)

These are checked on every test run via `tests/test_accessibility.py`,
which builds the real Gradio component tree and inspects it directly:

- Every interactive control (image input, upload/camera sources, submit
  button, status box, audio player, description box) has a non-empty
  accessible name (its `label`).
- The status control declares the correct `elem_id`/`elem_classes` and is
  marked non-interactive (so it can never steal focus).
- The description output declares the `elem_id` that the focus-management
  script targets.
- The injected JS shim that applies `aria-live`/`aria-atomic`/`role` uses a
  `MutationObserver` (not a one-time `window.load` handler), so it keeps
  working across Gradio's re-renders.
- The injected JS shim that makes the description output keyboard-focusable
  targets the correct element and swaps `disabled` for `readOnly` +
  `tabindex="0"`.
- The focus-management script is wrapped in `try`/`catch` so a missing or
  non-focusable element cannot throw a client-side error.
- No positive `tabindex` (`> 0`) appears anywhere in the injected markup.
- The audit checklist in `scripts/audit_accessibility.py` references the
  same `elem_id` constants the UI actually renders (import identity, not
  copied strings), so the two cannot silently drift apart.

These tests prove the *source* is structured correctly. They do not prove
anything about what a real browser renders or what a real screen reader
says - see the next two sections.

### Real-browser DOM verified (Chrome DevTools, by the owner)

During the P5.1 follow-up, the owner inspected the running app in a real
browser and confirmed:

- `aria-live="polite"`, `role="status"` are applied to the status region,
  and survive Gradio re-rendering it after a submit.
- The description output has `disabled=false`, `readOnly=true`,
  `tabIndex=0`, and sits in the tab order immediately after the submit
  button.
- The full keyboard tab order is: Drop image / Upload file / Capture from
  camera / Describe this photo / Description (text) / Use via API / Built
  with Gradio / Settings.

### Human screen-reader verified

The owner ran the live app with a screen reader and confirmed:

- **Progress announcement**: heard read aloud, verbatim: *"Photo received.
  Describing it now; this can take up to about 30 seconds."*
- **Completion announcement**: heard read aloud, verbatim: *"Description
  ready. Audio is playing; the text is below too."* (Note: the synthesized
  audio begins at the same moment and speaks over this announcement, making
  both difficult to hear - see Known defects.)

No other claim in this document is "screen-reader tested." That phrase is
not used here for anything beyond the statements above, because it has not
been confirmed for anything beyond the statements above.

## What is explicitly NOT verified

- **Post-run discoverability** - whether a screen-reader user reliably
  finds and reads the description text after the completion announcement -
  has not been human-verified.
- **A full keyboard-only pass** (using the app start to finish with no
  mouse) has not been human-verified.
- **Other screen readers and platforms** (this was checked with one screen
  reader on one platform; behavior on others - VoiceOver, TalkBack, other
  browser/screen-reader combinations - is unknown).
- **Mobile** has not been checked at all, on any platform.

## Known defects

These are being fixed:

- **Audio talks over the announcement (#47)**: When a description is ready,
  the synthesized audio begins at the same moment the screen reader is still
  announcing the completion status, making both difficult to hear. **Fix
  attempted, not yet confirmed by a human screen-reader pass:** the
  with-audio completion announcement is now short ("Description ready.")
  instead of "Description ready. Audio is playing; the text is below too.",
  since audio being available at all is the primary signal and a long
  announcement both duplicates it and collides with it. Audio no longer
  autoplays the instant it is ready; a script instead starts playback about
  1.8 seconds later, giving a screen reader time to finish the (now short)
  announcement first. If a browser blocks that deliberate playback attempt,
  the audio control stays visible and reachable so the user can press play
  manually - the status wording no longer claims audio "is playing" as a
  fact, since that may not be true. Whether the two voices actually stop
  colliding is something only a human screen-reader pass can confirm; that
  has not happened yet.
- **Images announce as "graphic" (#48)**: Images in the interface are not
  labelled, so a screen reader announces an unhelpful bare "graphic" instead
  of a meaningful description.

## Known limitations a user should know

- A request can take roughly **21-31 seconds** end to end (measured
  separately in issue #17); the progress announcement tells you this is
  expected, but there is no finer-grained progress indicator.
- **Audio may be unavailable.** If speech synthesis fails, the app falls
  back to showing (and announcing) the description as text - the text is
  always the reliable fallback, never a silent failure.
