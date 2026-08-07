---
name: plain-language
description: Strip AI-slop patterns out of any text this project ships. Use when writing or editing user-facing copy, docs, README, PR descriptions, issue text, or code comments that a reader will see. Adapted from a personal cover-letter writing skill.
---

# Plain language

Text this project ships is read by two audiences who both punish padding: a blind user
listening to it through a screen reader, and a hiring manager reading a public portfolio
repo. Slop costs more here than usual. A screen reader speaks every filler word at the
same speed as the useful ones.

Source of the tells: `references/ai-tells.md`, copied from a personal cover-letter writing
skill on 2026-08-06. Treat the word lists as perishable and the constructions as the
durable core. Refresh from the source project rather than editing the copy. Run
`tests/test_plain_language.py::test_skill_files_carry_no_private_references` after any
refresh; it catches dangling references and local filesystem paths, but not an unknown
private proper noun, so read the diff by eye for private context too.

## Where this applies

Everything a person reads:

- UI copy: status messages, labels, the intro text, the how-it-works section
- `docs/` and `README.md`
- PR descriptions and issue bodies
- Code comments that explain reasoning

Not: variable names, test names, log lines nobody reads.

## The rules, in priority order

### 1. No em dashes

The owner flagged em-dash prose as reading like AI (rule tightened 2026-08-04). Default
to zero. A comma, a full stop, or a parenthetical does the job.

This is the single most likely thing to slip, because em dashes feel like good writing.

### 2. Constructions to strike

These have a much longer half-life than vocabulary, so check them first.

- **Negative parallelism.** "It's not X, it's Y." Delete the construction and state the
  thing. This is the most durable AI tell on record, named independently by three
  sources across three years.
- **Rule of three.** "fast, simple and reliable." Real writing has lists of two and
  four. Concrete factual enumerations are fine; decorative triplets are not.
- **"Not only X, but also Y."** Split it or drop half.
- **Participle pile-ups.** Trailing clauses starting *highlighting, showcasing,
  reflecting, ensuring, underscoring, demonstrating*. Almost always deletable.
- **Connective chains.** *Moreover, Furthermore, Additionally* stacked across paragraphs.
- **Compulsive summary.** A closing paragraph starting *Overall* or *In conclusion* that
  restates what was just said. Cut it.
- **Staccato runs.** Three or four short sentences in a row. This is the current-generation
  default and it reads as generated.

### 3. Vocabulary to strike

Grep for these. Every hit is deleted or replaced with a plain word.

*delve, intricate, tapestry, pivotal, underscore, landscape, foster, testament, enhance,
crucial, robust, seamless, comprehensive, leverage (verb), navigate (figurative), realm,
showcase, spearhead, vital, essential, myriad, plethora, resonate, unlock, elevate,
transformative, cutting-edge, state-of-the-art, dynamic, synergy, boasts, commendable,
surpass, primarily, meticulous.*

Also the 2026 plain-monosyllable register, where the tell is collocation and density
rather than the word: *quietly (building/transforming), a shift in, this matters because,
shapes how, lands, actually (as filler), real (value/impact), earn (trust/the right to),
the work, hold space, compound, send the signal.*

### 4. Formatting tells

- No bold-stemmed bullets: every bullet opening with a **bolded label:** then text.
- No emoji in headings.
- Uniform bullet lengths that form a visual rectangle.
- Rhetorical colons: two-word fragment, colon, payoff.

### 5. Product-copy rules specific to this app

- **Never claim something the code cannot guarantee.** "Audio is playing" was shipped and
  was false whenever a browser blocked playback. Say what is true in every branch.
- **Shorter is an accessibility feature, not a style preference.** Every word in a status
  message is spoken aloud before the user gets the actual answer.
- **Say what was verified and how.** Do not write "tested" where the honest word is
  "machine-verified". See `docs/ACCESSIBILITY.md` for the split this project uses.

## The test

Read it aloud. If you would not say it to someone standing in front of you, rewrite it.

Do not chase AI detectors. They are unreliable in both directions and will flag ordinary
careful writing. If a detector and the read-aloud test disagree, the read-aloud test wins.

## Applying it to a diff

1. Grep the changed text for the vocabulary list and for ` — `.
2. Read each changed sentence and ask: does this report a fact, or perform an attitude?
   Facts stay.
3. For any paragraph, ask whether it restates its own opening claim and adds nothing. If
   so, cut it.
