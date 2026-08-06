"""Research node logic (issue #10 / P2.1): web-lookup for the research path.

This is the FIRST stage of the RESEARCH path (research -> analysis), run for
DENSE documents that dynamic_router sends the slow way. It looks up external
context for the primary subject found in the photo and hands it to
analysis.py as scraper_data - purely OPTIONAL supporting context, never
something the pipeline depends on to keep moving.

QUERY DERIVATION (no LLM call - this must stay cheap and fast)
----------------------------------------------------------------
The search query is built with plain string logic, not a model call: an
extra ladder round-trip here would burn latency budget the pipeline can't
spare (see client.ROLE_TIMEOUTS - the brain call after this one already has
up to 45s to run). ocr_output is preferred as the query source because it is
the photographed document's OWN words - the exact product name, medication
name, or document title a search engine can match against - whereas
scene_context is model-generated prose describing the photo (router.py made
the same call for the same reason: see its module docstring). scene_context
is used only as a fallback when ocr_output is empty. The chosen source is
truncated to its first QUERY_MAX_WORDS words to keep the query a short,
literal search string rather than a full paragraph.

INJECTABLE NETWORK SEAMS
--------------------------
Two seams, both injectable so tests never touch the network:
  - `searcher`: the search backend. Must expose `.text(query, max_results=N)`
    returning a list of dicts with an "href" key - the shape ddgs.DDGS()
    already provides. Defaults to a real DDGS(), constructed lazily.
  - `client`: an httpx.Client-like object used to fetch the top search
    result's page. Defaults to a real httpx.Client(), constructed lazily and
    closed in a `finally` if this function owns it (same pattern as
    vision.run_vision/synth.run_fast_synth's OpenRouterClient handling).
There is no separate "fetcher" abstraction beyond `client` - fetching is a
single client.stream() call (see _fetch_and_extract), so a third seam would
be indirection with nothing to inject differently.

WHICH ddgs PACKAGE
--------------------
`duckduckgo-search` (the older PyPI name) is deprecated in favour of `ddgs`,
which imports cleanly in this environment - `import ddgs` succeeds, so this
module uses `ddgs.DDGS` (see pyproject.toml).

FAILURE BEHAVIOR - THIS NODE MUST NEVER BREAK THE PIPELINE
--------------------------------------------------------------
scraper_data is optional context; analysis.py already treats "" as "no
external context available" and proceeds using ocr_output/scene_context
alone (see analysis.py's module docstring). So every failure mode here -
no query derivable, no search results, a search backend raising, a fetch
timeout, an HTTP 4xx/5xx, a non-HTML content type (PDF/image/etc), an
oversized page, malformed HTML, or any other unexpected exception - all
collapse to the SAME outcome: `{"scraper_data": ""}`. Nothing here may raise
into the graph. KeyboardInterrupt/SystemExit (BaseException, not Exception)
are never caught, same contract as vision.py/synth.py/analysis.py.

Two hazards get explicit bounds, since this runs against arbitrary sites
picked by a search result, on a shared Hugging Face Space:
  - `_FETCH_TIMEOUT_SECONDS` bounds the page fetch.
  - `_MAX_PAGE_BYTES` bounds how much of the response body is ever read;
    exceeding it aborts the fetch entirely rather than truncating and
    returning a partial page (see _fetch_and_extract).

THE #10 CONTRACT DECISION
----------------------------
scraper_data == "" currently means BOTH "fast path, research never ran" and
"research ran and found nothing usable". This module deliberately does NOT
disambiguate the two with a sentinel value threaded through scraper_data.
Reasons:
  1. analysis.py already documents (and tests) that "" means exactly one
     thing to it either way: "no external context available - proceed using
     ocr_output/scene_context alone, and do not hedge or leave the script
     contentless just because the scrape is empty." Its behaviour for the
     two causes is identical by design, so there is no decision downstream
     that actually depends on telling them apart.
  2. The state schema is capped at exactly 7 keys (state.py) - a sentinel
     folded into scraper_data itself would only be safe if every consumer
     checked for it structurally (the vision.is_degraded_scene pattern this
     issue was pointed at). Introducing that constant+predicate pair here
     with no consumer that needs to branch on it is exactly the kind of
     speculative machinery CLAUDE.md's Simplicity First asks to avoid, and
     it adds a NEW failure class this project has already had to fix twice:
     a rewording or accidental substring match silently breaking detection,
     or a caller that forgets to check the predicate and reads the sentinel
     text aloud as if it were real scraped content.
  3. A hedge like "no information was found" spoken into a script the model
     might paraphrase is worse than the model simply not mentioning web
     context at all - which is exactly what an empty "" already produces
     via analysis.py's existing prompt construction (the "Additional
     context from a web lookup" section is omitted entirely when
     scraper_data is falsy - see analysis._build_messages).
Collapsing both cases to "" therefore loses no information any consumer
acts on, and keeps the contract exactly as simple as it needs to be.

SECURITY - scraped text is UNTRUSTED
---------------------------------------
scraper_data returned here is attacker-influenceable web content (whoever
controls the top search result controls this string). This module does not
sanitise or censor it beyond the size cap - analysis.py is responsible for
treating it as untrusted, and does: analysis._build_messages wraps it with
prompting.fence_untrusted before it ever reaches a prompt (see analysis.py
line ~176), the same fencing already applied to ocr_output.
"""

import re

import httpx

QUERY_MAX_WORDS = 12

# Bounded fetch: this runs against an arbitrary site picked by a search
# result, on a shared Hugging Face Space - an unbounded fetch there is a
# real hazard, not a hypothetical one.
_FETCH_TIMEOUT_SECONDS = 8.0
_MAX_PAGE_BYTES = 300_000

# Only the single top result is ever used - "fetch ONE page" (no crawling,
# no retry across multiple results), so there is nothing useful in asking
# the search backend for more than one.
_MAX_SEARCH_RESULTS = 1

# Extracted text is capped independently of analysis.py's own
# _SCRAPER_DATA_CAP (4000 chars) - this keeps the state value itself a sane
# size even before it reaches analysis.py's prompt-construction cap, rather
# than relying on a downstream module to bound an unbounded upstream value.
_EXTRACTED_TEXT_MAX_CHARS = 4000

_HREF_SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)


def _default_searcher():
    """Factory for the real search backend. Called lazily (never at import time)."""
    from ddgs import DDGS

    return DDGS()


def _default_client():
    """Factory for the real HTTP client. Called lazily (never at import time)."""
    return httpx.Client()


def _derive_query(ocr_output, scene_context):
    """Build a short, literal search query from ocr_output (or scene_context).

    See module docstring "QUERY DERIVATION" for why ocr_output is preferred
    and no LLM call is used. Returns "" if neither input has any text.
    """
    ocr_output = (ocr_output or "").strip()
    source = ocr_output if ocr_output else (scene_context or "").strip()
    if not source:
        return ""
    return " ".join(source.split()[:QUERY_MAX_WORDS])


def _search_top_result(searcher, query):
    """Return the first usable http(s) href from `searcher`'s results, or None.

    A missing/blank/non-http(s) href on a given result is skipped in favour
    of the next one, but no additional network round-trip happens - the
    search backend was already asked for at most _MAX_SEARCH_RESULTS.
    """
    results = searcher.text(query, max_results=_MAX_SEARCH_RESULTS)
    for result in results or []:
        href = result.get("href") if isinstance(result, dict) else None
        if isinstance(href, str) and _HREF_SCHEME_RE.match(href.strip()):
            return href.strip()
    return None


def _decode(content_type, raw_bytes):
    """Decode `raw_bytes` using the charset in `content_type`, defensively.

    Falls back to utf-8 with lossy replacement if the declared charset is
    missing, unknown, or wrong - a malformed/mislabelled page must degrade
    into readable-but-imperfect text, never raise.
    """
    charset = "utf-8"
    lowered = (content_type or "").lower()
    if "charset=" in lowered:
        charset = lowered.split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
    try:
        return raw_bytes.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return raw_bytes.decode("utf-8", errors="replace")


def _extract_readable_text(html):
    """Extract and normalise readable text from `html`, or None if there is none.

    Script/style/noscript content is dropped before extraction - it is
    never meant to be read as prose. Collapsed to a single run of
    whitespace-separated text and capped at _EXTRACTED_TEXT_MAX_CHARS (word
    boundary, with a visible truncation marker, mirroring
    analysis._cap_scraper_data's approach).
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True)).strip()
    if not text:
        return None
    if len(text) > _EXTRACTED_TEXT_MAX_CHARS:
        text = text[:_EXTRACTED_TEXT_MAX_CHARS].rsplit(" ", 1)[0] + " [truncated]"
    return text


def _fetch_and_extract(client, url):
    """Fetch `url` (bounded by timeout and size) and return extracted text, or None.

    Streamed rather than fetched in one shot so the size cap is enforced
    DURING download, not after the fact - a page that blows the cap is
    abandoned as soon as it's detected, not downloaded in full and then
    discarded. The content-type check happens before any body bytes are
    read at all, so a non-HTML response (PDF, image, ...) costs nothing
    beyond the response headers.
    """
    with client.stream("GET", url, timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True) as response:
        if response.status_code >= 400:
            return None
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.lower():
            return None
        chunks = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > _MAX_PAGE_BYTES:
                return None
            chunks.append(chunk)
        html = _decode(content_type, b"".join(chunks))
    return _extract_readable_text(html)


def run_research(ocr_output, scene_context, searcher=None, client=None):
    """Look up external context for the document's subject; return a research_node state update.

    Always returns {"scraper_data": <str>} - never raises (except
    KeyboardInterrupt/SystemExit). `searcher`/`client` are injectable (tests
    pass fakes); when omitted, real defaults are constructed lazily. A
    client built here (not injected) is closed in a `finally`; an injected
    client is owned by the caller and never closed here - same pattern as
    vision.run_vision/synth.run_fast_synth's OpenRouterClient handling.
    """
    query = _derive_query(ocr_output, scene_context)
    if not query:
        return {"scraper_data": ""}

    owns_client = client is None
    try:
        if searcher is None:
            searcher = _default_searcher()
        if owns_client:
            client = _default_client()

        try:
            url = _search_top_result(searcher, query)
        except Exception:
            return {"scraper_data": ""}
        if url is None:
            return {"scraper_data": ""}

        try:
            text = _fetch_and_extract(client, url)
        except Exception:
            return {"scraper_data": ""}
        if text is None:
            return {"scraper_data": ""}

        return {"scraper_data": text}
    except Exception:
        # Contract (module docstring): no raw exception may escape into the
        # graph, including a failure constructing the default searcher/
        # client. Never swallows KeyboardInterrupt/SystemExit, which derive
        # from BaseException, not Exception.
        return {"scraper_data": ""}
    finally:
        if owns_client and client is not None:
            client.close()
