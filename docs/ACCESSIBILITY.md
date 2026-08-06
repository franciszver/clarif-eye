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
- **Images announce as "graphic" (#48)**: A real screen-reader pass (owner,
  Narrator on Windows) found every image on the page announcing as an
  unlabelled "graphic" - repeatedly, once per image, while navigating
  ("graphic, graphic, graphic") - both Gradio's own chrome (footer logo,
  "Use via API" logo, button glyphs) and the user's own uploaded photo.
  **Fix attempted, not yet confirmed by a human screen-reader pass:** the
  aria-live shim now classifies every `img`/`svg`/`[role="img"]` it finds.
  The uploaded photo preview (identified structurally - it's the `<img>`
  inside the photo-input component's own container, never a string/URL
  match) is given a real accessible name ("The photo you submitted").
  Everything else - Gradio's chrome - is marked decorative and handled
  per element type, since a screen reader continuing to announce
  "graphic" means the node was still present in the accessibility tree,
  not just unnamed: `<img>` gets `alt=""` plus `aria-hidden="true"`;
  inline `<svg>` gets `aria-hidden="true"` plus `focusable="false"`
  (`alt=""` does nothing on `<svg>` - it only applies to `<img>` - so
  setting it alone would have changed nothing); any `[role="img"]`
  element gets `aria-hidden="true"`. Decorative nodes are also taken out
  of the tab order if they were somehow focusable. Whether this actually
  goes quiet for the chrome and actually reads out the photo's name in a
  real screen reader is something only a human screen-reader pass can
  confirm; that has not happened yet.
- **Pressing Play collides with the control announcement (#52)**: A real
  screen-reader pass (owner, Narrator on Windows) found that pressing the
  audio widget's own Play button - to replay the description, or because
  the browser had blocked the automatic playback from #47 - made the
  screen reader announce the control's activation/state change at the same
  instant the audio started, so the opening seconds of the description
  (often the most important part) were lost under that announcement. This
  is distinct from #47: #47 was this app's OWN status text colliding with
  AUTOMATIC playback; #52 is the SCREEN READER'S OWN announcement, which
  this app cannot detect or suppress, colliding with a USER GESTURE. **Fix
  attempted, not yet confirmed by a human screen-reader pass:** the same
  aria-live shim now delays the actual start of any user-initiated play by
  about a second (`USER_PLAY_DELAY_MS`), separately from the ~1.8s
  automatic-playback delay from #47 (`AUDIO_PLAY_DELAY_MS`) - the two
  delays are structured so a user gesture arriving while the automatic
  attempt is still pending does not stack into a longer wait. Playback is
  never started early and then paused again (that would produce an
  audible stutter); the real start is simply deferred. Pausing is not
  delayed. Whether this actually gives the announcement enough room to
  finish before a real screen reader is something only a human
  screen-reader pass can confirm; that has not happened yet.
- **Audio never played at all - a regression from #47/#52's first
  implementation.** After #47 and #52 shipped, the owner reported "the
  spoken description doesn't start" - audio never played, period, not just
  with imperfect timing. The deferred-playback trigger scheduled playback
  by checking, inside the existing aria-live shim's `apply()`, whether the
  `<audio>` element's `src` was truthy. That check was never satisfied in
  practice: Gradio's Svelte player assigns `audio.src` as a JS PROPERTY,
  and a property assignment produces no DOM mutation, so the
  childList/subtree `MutationObserver` that re-runs `apply()` had no
  reliable reason to fire at the moment a source actually became
  available. Diagnosed by the owner in a real browser (Chrome DevTools):
  `#audio-output audio` existed, but `getAttribute('src')` was `null`, the
  `.src` property was `""`, and there were zero `<source>` children at the
  moment `apply()` ran. **The full automated suite passed the entire time
  this was broken** - this defect was found by a human, not by the suite,
  which is exactly the honesty gap Owner decision D13 already flags for
  this whole file (automated checks assert the shim's structure, never
  that sound actually comes out of a speaker). **Fix attempted, not yet
  confirmed by a human/browser pass:** the shim now attaches one real
  `loadeddata` event listener to the `<audio>` element the first time it's
  seen (guarded against double-attachment so Gradio reusing the same DOM
  node across submissions doesn't stack listeners), forces
  `audioEl.preload = "auto"` so that event is guaranteed to fire once a
  source is assigned, and schedules the same `AUDIO_PLAY_DELAY_MS`-delayed
  playback from inside that event instead of from an `audioEl.src` check.
  The per-src dedupe runs inside the listener's callback (not as a gate on
  attaching the listener itself), so a second, third, ... submission's
  audio is scheduled too, not just the first. #52's user-gesture delay and
  the immediate-pause guarantee are unchanged. Whether audio now actually
  plays in a real browser is something only a human, driving the real
  app, can confirm; that has not happened yet.

## Known limitations a user should know

- A request can take roughly **21-31 seconds** end to end (measured
  separately in issue #17); the progress announcement tells you this is
  expected, but there is no finer-grained progress indicator.
- **Audio may be unavailable.** If speech synthesis fails, the app falls
  back to showing (and announcing) the description as text - the text is
  always the reliable fallback, never a silent failure.
