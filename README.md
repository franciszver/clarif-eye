# Clarif-Eye

[![CI](https://github.com/franciszver/clarif-eye/actions/workflows/ci.yml/badge.svg)](https://github.com/franciszver/clarif-eye/actions/workflows/ci.yml)

Try it live: [https://clarif-eye.onrender.com](https://clarif-eye.onrender.com).
It runs on a free hosting tier, so if it has been idle the first request
can take about 60 seconds to wake it up; see [Deployment](#deployment) for
why.

Clarif-Eye turns a photo into a spoken description, for people who cannot
see the photo they are describing. Point a camera at a bill, a label, or a
document, and the app reads the text, describes the scene, and speaks the
result back to you. On documents with dense numbers, such as an itemized
bill, it does extra work to check that every amount it plans to say aloud
actually traces back to the photographed text before speaking it.

## A demo, honestly described

This is a portfolio demo, not a product. It runs entirely on free-tier
models from [OpenRouter](https://openrouter.ai/), which keeps the running
cost at zero but means responses are slower and less reliable than a paid
model would be. A request on the closer-look path (dense documents that
need number verification) has measured 21 to 31 seconds end to end. The
app tells you up front to expect up to about 30 seconds.

See [docs/ACCESSIBILITY.md](docs/ACCESSIBILITY.md) for what has actually
been verified about the screen-reader experience, and what has not.

## Running it locally

Requires Python 3.11 or later.

```bash
pip install -e .
```

The app calls OpenRouter, so it needs an API key in the environment:

```bash
export OPENROUTER_API_KEY=your-key-here
python app.py
```

This starts a local Gradio server and prints its URL.

For development setup (installing test dependencies, running the test
suite), see [CONTRIBUTING.md](CONTRIBUTING.md).

## How it works

1. You take or upload a photo.
2. A vision-language model reads any text in the photo and separately
   describes the scene (what it is, its layout).
3. A router decides, from that text alone, whether a quick description is
   enough or the document is dense enough to need a closer look (an
   itemized bill, a prescription label, a form with numbers on it). This
   decision is plain Python, no model or network call, scoring things like
   digit density, currency amounts, and document keywords.
4. If a quick description is enough, the photographed text and scene
   description are turned directly into the spoken script.
5. If a closer look is needed, the app first does a web search related to
   what was photographed, then a stronger text-reasoning model writes the
   script from the photographed text, the scene description, and whatever
   the search turned up.
6. On that closer-look path, before anything is spoken, every number in the
   drafted script is checked against the photographed text. If a number
   doesn't trace back to what the camera actually saw, the app stops and
   asks: it reads out the description it wrote, says which number it could
   not check, and offers two buttons - hear it anyway, or take a new photo.
   Nothing is read aloud as fact until you choose, and this is the only
   thing the app ever stops to ask you about.
7. The final script is converted to speech.

The pipeline is built with [LangGraph](https://github.com/langchain-ai/langgraph)'s
`StateGraph`. Every model call goes through one of two roles, "eyes" (reads
the photo) and "brain" (writes the closer-look description), each an
ordered ladder of free models tried in turn if an earlier one fails. If
speech synthesis fails, the app still shows the description as text rather
than failing silently.

The in-app "How this works" panel (visible once the app is running)
carries the same explanation, kept in sync with the code by a test. For
more detail, including what a real screen-reader pass did and did not
confirm, see [docs/ACCESSIBILITY.md](docs/ACCESSIBILITY.md). For how the
external systems this app depends on (the model API, search, text-to-speech)
are checked against real calls rather than only mocks, see
[docs/SCENARIOS.md](docs/SCENARIOS.md).

## Deployment

Clarif-Eye runs on Render's free tier, configured in
[render.yaml](render.yaml). Two things to know before you rely on the live
URL:

- **The app sleeps after 15 minutes with no traffic.** Waking it back up
  takes about a minute, during which Render shows its own loading page.
  This is on top of the 21 to 31 second pipeline latency noted above; a
  cold visit can take close to two minutes end to end.
- **That loading page is Render's, not this app's.** Whether it announces
  itself to a screen reader has not been checked, because it is outside
  this app's control. A blind user may spend that first minute on a page
  nobody here has verified.

The host supplies `OPENROUTER_API_KEY` as an environment variable in its
own dashboard. `.env` is never deployed; it exists only for running the
app locally.

## License

MIT, see [LICENSE](LICENSE).
