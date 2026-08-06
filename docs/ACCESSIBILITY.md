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
  announcement both duplicates it and collides with it. The gap before
  audio starts is created by the server, not the browser: `autoplay` stays
  on (see the entry below for why an earlier attempt that turned it off
  broke playback entirely) and the handler that streams results to the UI
  yields the completion status and text first, then waits about 1.8
  seconds before yielding the audio path, so Gradio does not mount the
  autoplaying player until a screen reader has had time to finish the (now
  short) announcement. If a browser blocks autoplay anyway, the audio
  control stays visible and reachable so the user can press play manually
  - the status wording no longer claims audio "is playing" as a fact,
  since that may not be true. Whether the two voices actually stop
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
  implementation, then a second, deeper regression in the fix attempted
  for it.** After #47 and #52 shipped, the owner reported "the spoken
  description doesn't start" - audio never played, period, not just with
  imperfect timing. The first diagnosis (an `audioEl.src` truthiness check
  inside the aria-live shim's `apply()` that a JS property assignment
  never triggers a `MutationObserver` for) led to a fix that swapped that
  check for a real `loadeddata` event listener on the `<audio>` element,
  with `audioEl.preload` forced to `"auto"`. **That fix also passed its
  own full test suite while audio was still completely broken in the
  browser** - it fixed the symptom it could see (a `MutationObserver`
  never firing) but not the actual cause. A real-browser check by the
  orchestrator on this branch found: run status "Description ready.", a
  correct download link to a real `.mp3` in `#audio-output` (so audio
  *was* generated), but the `<audio>` element had `preload="auto"` (the
  shim did run) with `src` absent, `readyState` 0, and zero
  `loadeddata`/`play`/`error` events ever firing. The reason: Gradio only
  assigns the `<audio>` element a source at all when its own `autoplay`
  prop is `true`. #47 had turned `autoplay` off so the app's own JS could
  control when playback started; with it off, Gradio never assigns a
  source in the first place, so there is no `loadeddata` (or any other
  media-load event) for any JS listener to ever catch - the whole
  event-based approach was a dead end, not a bug to patch further.
  **Corrected fix:** `autoplay=True` is restored on the `gr.Audio`
  component - the only thing that makes Gradio assign a source and play
  it at all - and the announcement-vs-audio gap that #47 introduced
  autoplay=False to create is now produced without any client-side JS:
  `handle_submit_staged` (a generator) yields the completion status and
  the description text first, with no audio path, then sleeps
  `AUDIO_PLAY_DELAY_MS` and yields again with the audio path, so Gradio
  only mounts the (autoplaying) player once the status announcement has
  had time to be spoken. The `loadeddata` listener, the `preload` forcing,
  and the `deferredPlaySrc`/`a11yAutoplayPending` bookkeeping have all been
  removed as dead code. #52's user-gesture play delay is unchanged: it
  wraps any JS-initiated call to `audioEl.play()`, which is still exactly
  how Gradio's own Play button (re-)triggers playback, so pressing Play
  still gets the delay; native browser autoplay does not go through that
  wrapped method, so nothing about it changed. Pausing remains immediate,
  as before. Whether audio now actually plays, and whether the
  announcement and audio no longer collide, in a real browser is something
  only a human/browser pass can confirm; that has not happened yet -
  **awaiting confirmation, not claimed fixed.**
- **Camera photos come back mirrored (#59)**: The owner, using the camera
  to photograph a bill/label/sign, found the resulting image reversed -
  Gradio's `gr.Image` webcam capture mirrors by default
  (`WebcamOptions.mirror=True`), a setting meant for selfies. Reversed text
  is not merely harder to read; the vision model cannot read it at all, so
  the description it produced was confidently wrong rather than merely
  poor, and every downstream step (density scoring, number verification,
  the spoken description) then operated on that wrong reading. **Fix
  attempted, not yet confirmed by a real capture:** the `gr.Image` call in
  `build_interface` now passes `webcam_options=gr.WebcamOptions(mirror=False)`,
  since this app has no selfie use case - every photo is of something the
  user is not looking at, so mirroring has no upside and only breaks text.
  This is checked by an automated test that inspects the constructed
  component's `webcam_options.mirror` value - a constructor-argument check
  only. Whether the captured pixels are actually un-mirrored can only be
  confirmed by taking a real photo with a real camera, which the owner will
  do; that has not happened yet - **awaiting confirmation, not claimed
  fixed.**

## Known limitations a user should know

- A request can take roughly **21-31 seconds** end to end (measured
  separately in issue #17); the progress announcement tells you this is
  expected, but there is no finer-grained progress indicator.
- **Audio may be unavailable.** If speech synthesis fails, the app falls
  back to showing (and announcing) the description as text - the text is
  always the reliable fallback, never a silent failure.
