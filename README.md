# Clarif-Eye

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
   doesn't trace back to what the camera actually saw, the app reports that
   the result could not be verified rather than risk reading a wrong amount
   or date aloud.
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

## License

MIT, see [LICENSE](LICENSE).
