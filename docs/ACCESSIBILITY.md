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

### Real-browser DOM verified (Chrome DevTools)

In a later accessibility pass, the developer inspected the running app in a
real browser and confirmed:

- `aria-live="polite"`, `role="status"` are applied to the status region,
  and survive Gradio re-rendering it after a submit.
- The description output has `disabled=false`, `readOnly=true`,
  `tabIndex=0`, and sits in the tab order immediately after the submit
  button.
- The full keyboard tab order is: Drop image / Upload file / Capture from
  camera / Describe this photo / Description (text) / Use via API / Built
  with Gradio / Settings.

### Human screen-reader verified

The developer reloaded the app on `main` and ran the full checklist below
with Narrator (Windows) and a real camera. His report, verbatim: "refreshed
the page, camera works, all items work." Each item was confirmed
individually:

- **Camera capture**: photographed something with text; the text in the
  resulting photo reads correctly, not reversed
  ([#59](https://github.com/franciszver/clarif-eye/issues/59)).
- **Tab through the page**: no image anywhere announces as an unlabelled
  "graphic" ([#48](https://github.com/franciszver/clarif-eye/issues/48)).
- **Progress announcement**: submitting a photo announces "Photo received.
  Describing it now..."
- **Completion timing**: waiting for a result, "Description ready."
  finishes announcing before the audio starts
  ([#47](https://github.com/franciszver/clarif-eye/issues/47)).
- **Replay**: pressing Play to replay the description, the announcement
  finishes before audio starts, and the audio plays from its first word
  ([#52](https://github.com/franciszver/clarif-eye/issues/52)).
- **Pause**: pressing Pause stops the audio instantly.
- **Heading navigation**: Caps Lock + Space, then pressing H repeatedly,
  moves through six headings in order, ending at "Honest operational
  notes."
- **Audio playback**: the audio plays at all. This had regressed to
  complete silence in an earlier build and is now fixed.

This was one screen reader (Narrator), one platform (Windows), one browser,
one session. It is not a multi-platform certification. No other claim in
this document is "screen-reader tested." That phrase is not used here for
anything beyond the checklist above, because it has not been confirmed for
anything beyond the checklist above.

#### Fixes confirmed by this pass

The checklist above confirmed five defects that this document previously
listed as open, with a fix in place but unverified by a human. Each is
recorded here with the reasoning behind its fix, kept for anyone debugging
something similar later.

- **Audio talked over the announcement
  ([#47](https://github.com/franciszver/clarif-eye/issues/47)), now
  fixed.** The synthesized
  audio used to begin at the same moment the screen reader was still
  announcing the completion status, making both hard to hear. The fix: the
  with-audio completion announcement is now short ("Description ready.")
  instead of "Description ready. Audio is playing; the text is below too.",
  since audio being available at all is the primary signal and a long
  announcement both duplicates it and collides with it. The gap before
  audio starts is created by the server, not the browser: `autoplay` stays
  on (see the entry below for why an earlier attempt that turned it off
  broke playback entirely) and the handler that streams results to the UI
  yields the completion status and text first, then waits about 1.8
  seconds before yielding the audio path, so Gradio does not mount the
  autoplaying player until a screen reader has had time to finish the
  short announcement. If a browser blocks autoplay anyway, the audio
  control stays visible and reachable so the user can press play manually;
  the status wording no longer claims audio "is playing" as a fact, since
  that may not be true. The human pass above confirms the two do not
  collide: "Description ready." finishes before audio starts.
- **Images announced as "graphic"
  ([#48](https://github.com/franciszver/clarif-eye/issues/48)), now
  fixed.** A real screen-reader
  pass (owner, Narrator on Windows) had found every image on the page
  announcing as an unlabelled "graphic", repeatedly, once per image, while
  navigating ("graphic, graphic, graphic"), covering both Gradio's own
  chrome (footer logo, "Use via API" logo, button glyphs) and the user's
  own uploaded photo. The fix: the aria-live shim now classifies every
  `img`/`svg`/`[role="img"]` it finds. The uploaded photo preview
  (identified structurally, it's the `<img>` inside the photo-input
  component's own container, never a string/URL match) is given a real
  accessible name ("The photo you submitted"). Everything else, Gradio's
  chrome, is marked decorative and handled per element type, since a
  screen reader continuing to announce "graphic" meant the node was still
  present in the accessibility tree, not just unnamed: `<img>` gets
  `alt=""` plus `aria-hidden="true"`; inline `<svg>` gets
  `aria-hidden="true"` plus `focusable="false"` (`alt=""` does nothing on
  `<svg>`, it only applies to `<img>`, so setting it alone would have
  changed nothing); any `[role="img"]` element gets `aria-hidden="true"`.
  Decorative nodes are also taken out of the tab order if they were
  somehow focusable. The human pass above confirms tabbing through the
  page announces no "graphic" anywhere.
- **Pressing Play collided with the control announcement
  ([#52](https://github.com/franciszver/clarif-eye/issues/52)), now
  fixed.** A real screen-reader pass (developer, Narrator on Windows) had
  found that pressing the audio widget's own Play button, to replay the
  description or because the browser had blocked the automatic playback
  from [#47](https://github.com/franciszver/clarif-eye/issues/47), made
  the screen reader announce the control's activation/state change at the
  same instant the audio started, so the opening seconds of the
  description (often the most important part) were lost under that
  announcement. This was distinct from #47: #47 was this
  app's own status text colliding with automatic playback; #52 is the
  screen reader's own announcement, which this app cannot detect or
  suppress, colliding with a user gesture. The fix: the same aria-live
  shim now delays the actual start of any user-initiated play by about a
  second (`USER_PLAY_DELAY_MS`), separately from the ~1.8s
  automatic-playback delay from #47 (`AUDIO_PLAY_DELAY_MS`); the two
  delays are structured so a user gesture arriving while the automatic
  attempt is still pending does not stack into a longer wait. Playback is
  never started early and then paused again (that would produce an
  audible stutter); the real start is simply deferred. Pausing is not
  delayed. The human pass above confirms pressing Play lets the
  announcement finish before audio starts, and the audio plays from its
  first word.
- **Audio never played at all, now fixed.** This was a regression from
  [#47](https://github.com/franciszver/clarif-eye/issues/47)/[#52](https://github.com/franciszver/clarif-eye/issues/52)'s
  first implementation, then a second, deeper regression in the attempt to
  fix it. After #47 and #52 shipped, the developer reported "the spoken
  description doesn't start": audio never played, period, not just with
  imperfect timing. The first diagnosis (an `audioEl.src` truthiness
  check inside the aria-live shim's `apply()` that a JS property
  assignment never triggers a `MutationObserver` for) led to a fix that
  swapped that check for a real `loadeddata` event listener on the
  `<audio>` element, with `audioEl.preload` forced to `"auto"`. **That fix
  also passed its own full test suite while audio was still completely
  broken in the browser**: it fixed the symptom it could see (a
  `MutationObserver` never firing) but not the actual cause. A
  real-browser check on this branch found: run status
  "Description ready.", a correct download link to a real `.mp3` in
  `#audio-output` (so audio *was* generated), but the `<audio>` element
  had `preload="auto"` (the shim did run) with `src` absent, `readyState`
  0, and zero `loadeddata`/`play`/`error` events ever firing. The reason:
  Gradio only assigns the `<audio>` element a source at all when its own
  `autoplay` prop is `true`. #47 had turned `autoplay` off so the app's
  own JS could control when playback started; with it off, Gradio never
  assigned a source in the first place, so there was no `loadeddata` (or
  any other media-load event) for any JS listener to ever catch. The
  event-based approach was a dead end, not a bug to patch further.
  **Corrected fix:** `autoplay=True` is restored on the `gr.Audio`
  component, the only thing that makes Gradio assign a source and play it
  at all, and the announcement-vs-audio gap that #47 introduced
  `autoplay=False` to create is now produced without any client-side JS:
  `handle_submit_staged` (a generator) yields the completion status and
  the description text first, with no audio path, then sleeps
  `AUDIO_PLAY_DELAY_MS` and yields again with the audio path, so Gradio
  only mounts the (autoplaying) player once the status announcement has
  had time to be spoken. The `loadeddata` listener, the `preload` forcing,
  and the `deferredPlaySrc`/`a11yAutoplayPending` bookkeeping have all
  been removed as dead code. #52's user-gesture play delay is unchanged:
  it wraps any JS-initiated call to `audioEl.play()`, which is still
  exactly how Gradio's own Play button (re-)triggers playback, so pressing
  Play still gets the delay; native browser autoplay does not go through
  that wrapped method, so nothing about it changed. Pausing remains
  immediate, as before. The human pass above confirms audio plays again,
  and that the completion announcement and audio no longer collide.
- **Camera photos came back mirrored
  ([#59](https://github.com/franciszver/clarif-eye/issues/59)), now
  fixed.** The developer, using
  the camera to photograph a bill, label, or sign, had found the resulting
  image reversed. Gradio's `gr.Image` webcam capture mirrors by default
  (`WebcamOptions.mirror=True`), a setting meant for selfies. Reversed
  text is not merely harder to read; the vision model cannot read it at
  all, so the description it produced was confidently wrong rather than
  merely poor, and every downstream step (density scoring, number
  verification, the spoken description) then operated on that wrong
  reading. The fix: the `gr.Image` call in `build_interface` now passes
  `webcam_options=gr.WebcamOptions(mirror=False)`, since this app has no
  selfie use case; every photo is of something the user is not looking
  at, so mirroring has no upside and only breaks text. This is checked by
  an automated test that inspects the constructed component's
  `webcam_options.mirror` value, a constructor-argument check only. The
  human pass above confirms the captured photo is no longer mirrored:
  photographed text reads correctly.

## What is explicitly NOT verified

- **A full keyboard-only pass** (using the app start to finish with no
  mouse, outside the checklist above) has not been human-verified.
- **Other screen readers** - this was checked with Narrator only; behavior
  on VoiceOver, TalkBack, JAWS, or others is unknown.
- **Other platforms** - this was checked on Windows only; behavior on
  macOS, iOS, Android, or Linux is unknown.
- **Other browsers** - the browser used has not been varied; other
  browser/screen-reader combinations are unknown.
- **Mobile** has not been checked at all, on any platform.
- **Render's cold-start loading page.** After 15 minutes with no traffic,
  the app sleeps; waking it back up takes about a minute, and Render shows
  its own loading page during that time. That page belongs to Render, not
  this app, so whether it announces itself to a screen reader is outside
  this app's control and has not been checked.

## Known defects

No known accessibility defects are currently outstanding.

## What testing did and did not catch

Every accessibility defect found in this project so far was found by a
person, not by the automated test suite, and every one of them passed that
suite while the app was still broken:

| Defect | Test suite status while broken |
| --- | --- |
| Live region was inert | 455 tests passing, 12 of them accessibility tests |
| Description not keyboard reachable | 455 passing |
| Audio talked over the announcement | 458 passing |
| Images announced as "graphic" | 474 passing |
| Play button collision | 479 passing |
| Audio never played at all | 484 passing |
| Camera mirrored, text unreadable | 484 passing |

The tests prove the code does what it says. They do not prove it does what
it is for. That gap is why the human passes in this document exist, and
why this document distinguishes machine-verified from human-verified
instead of calling either one "tested."

## Known limitations a user should know

- A request can take roughly **21-31 seconds** end to end (measured
  separately, see
  [#17](https://github.com/franciszver/clarif-eye/issues/17)); the
  progress announcement tells you this is expected, but there is no
  finer-grained progress indicator.
- **Audio may be unavailable.** If speech synthesis fails, the app falls
  back to showing (and announcing) the description as text - the text is
  always the reliable fallback, never a silent failure.
