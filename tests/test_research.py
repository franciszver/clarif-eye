"""Tests for the research node (issue #10 / P2.1): web-lookup for the research path.

No network calls: the search backend (`searcher`) and the HTTP client used
to fetch the top result (`client`) are both injectable, same pattern as
vision.py/synth.py's `client` seam. Page fetches are exercised with a REAL
httpx.Client wired to httpx.MockTransport (same technique test_client.py
uses for OpenRouterClient) so the actual streaming/size/content-type logic
runs, not a hand-rolled fake.

Covers: query derivation, every failure path (no results, search raising,
timeout, 404, 500, non-HTML content-type, oversized page, malformed HTML,
unexpected exception) degrading to scraper_data == "" rather than raising,
KeyboardInterrupt propagating, the happy path, and client/searcher lifecycle.
"""

import socket

import httpx
import pytest

from clarif_eye import research
from clarif_eye.graph import build_graph, research_node
from clarif_eye.client import CompletionResult
from clarif_eye.research import _derive_query, run_research
from clarif_eye.state import make_initial_state

from tests.test_graph import DEEP_TRACE
from tests._stream_helpers import drain_stream_collecting_trace


# --- Fakes -------------------------------------------------------------


class FakeSearcher:
    def __init__(self, results=None, exc=None):
        self.results = results if results is not None else []
        self.exc = exc
        self.calls = []

    def text(self, query, **kwargs):
        self.calls.append({"query": query, "kwargs": kwargs})
        if self.exc is not None:
            raise self.exc
        return self.results


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def html_response(status_code=200, body="<html><body><p>Hello world</p></body></html>", content_type="text/html"):
    return httpx.Response(status_code, headers={"content-type": content_type}, text=body)


def fake_getaddrinfo(hostname_to_ips):
    """Build a socket.getaddrinfo replacement backed by a fixed hostname->IPs map.

    No real DNS is ever performed by the SSRF tests below - every hostname
    they use is normalised (rstrip('.').lower()) and looked up here.
    """

    def _fake(host, *args, **kwargs):
        ips = hostname_to_ips.get(host)
        if ips is None:
            raise socket.gaierror(f"no fake DNS entry for {host!r}")
        results = []
        for ip in ips:
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            sockaddr = (ip, 0, 0, 0) if family == socket.AF_INET6 else (ip, 0)
            results.append((family, socket.SOCK_STREAM, 6, "", sockaddr))
        return results

    return _fake


def redirect_response(location, status_code=302):
    return httpx.Response(status_code, headers={"location": location})


# --- Query derivation -----------------------------------------------------


def test_derive_query_uses_ocr_output_when_present():
    query = _derive_query("Acme Widget Pro 3000", "a product box on a table")
    assert query == "Acme Widget Pro 3000"


def test_derive_query_falls_back_to_scene_context_when_ocr_empty():
    query = _derive_query("", "a red bicycle leaning against a brick wall")
    assert query == "a red bicycle leaning against a brick wall"


def test_derive_query_returns_empty_when_both_empty():
    assert _derive_query("", "") == ""
    assert _derive_query(None, None) == ""


def test_derive_query_truncates_to_a_word_limit():
    long_ocr = " ".join(f"word{i}" for i in range(50))
    query = _derive_query(long_ocr, "")
    assert query != long_ocr
    assert len(query.split()) < 50
    assert query == " ".join(long_ocr.split()[: len(query.split())])


# --- No query derivable: search must not even be attempted -----------------


def test_run_research_skips_search_when_no_query_can_be_derived():
    searcher = FakeSearcher(results=[{"href": "https://example.com"}])

    result = run_research("", "", searcher=searcher)

    assert result == {"scraper_data": ""}
    assert searcher.calls == []


# --- Search-level failures --------------------------------------------------


def test_no_search_results_degrades_to_empty_scraper_data():
    searcher = FakeSearcher(results=[])

    result = run_research("some product", "a scene", searcher=searcher)

    assert result == {"scraper_data": ""}


def test_search_raising_degrades_to_empty_scraper_data():
    searcher = FakeSearcher(exc=RuntimeError("ddgs backend exploded"))

    result = run_research("some product", "a scene", searcher=searcher)

    assert result == {"scraper_data": ""}


def test_search_result_missing_usable_href_degrades_to_empty():
    searcher = FakeSearcher(results=[{"title": "no href here"}, {"href": None}, {"href": "not-a-url"}])

    result = run_research("some product", "a scene", searcher=searcher)

    assert result == {"scraper_data": ""}


# --- Fetch-level failures ----------------------------------------------------


def test_fetch_timeout_degrades_to_empty_scraper_data():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    searcher = FakeSearcher(results=[{"href": "https://example.com/page"}])
    client = make_client(handler)

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    assert result == {"scraper_data": ""}


def test_fetch_404_degrades_to_empty_scraper_data():
    def handler(request):
        return html_response(status_code=404, body="not found")

    searcher = FakeSearcher(results=[{"href": "https://example.com/missing"}])
    client = make_client(handler)

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    assert result == {"scraper_data": ""}


def test_fetch_500_degrades_to_empty_scraper_data():
    def handler(request):
        return html_response(status_code=500, body="server error")

    searcher = FakeSearcher(results=[{"href": "https://example.com/broken"}])
    client = make_client(handler)

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    assert result == {"scraper_data": ""}


def test_non_html_content_type_degrades_to_empty_scraper_data():
    def handler(request):
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4 fake pdf bytes")

    searcher = FakeSearcher(results=[{"href": "https://example.com/doc.pdf"}])
    client = make_client(handler)

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    assert result == {"scraper_data": ""}


def test_oversized_page_degrades_to_empty_scraper_data():
    huge_body = "<html><body>" + ("x " * 500_000) + "</body></html>"

    def handler(request):
        return html_response(body=huge_body)

    searcher = FakeSearcher(results=[{"href": "https://example.com/huge"}])
    client = make_client(handler)

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    # Enforced cap: a page bigger than the size limit is dropped entirely,
    # not truncated and partially returned.
    assert result == {"scraper_data": ""}


def test_malformed_html_does_not_raise(monkeypatch):
    def handler(request):
        return html_response(body="<html><body>unterminated <div><p>oops")

    def _boom_extract(html):
        raise ValueError("parser choked")

    monkeypatch.setattr(research, "_extract_readable_text", _boom_extract)

    searcher = FakeSearcher(results=[{"href": "https://example.com/bad"}])
    client = make_client(handler)

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    assert result == {"scraper_data": ""}


def test_unexpected_exception_during_search_degrades_without_raising():
    searcher = FakeSearcher(exc=ValueError("weird failure"))

    result = run_research("some product", "a scene", searcher=searcher)

    assert result == {"scraper_data": ""}


def test_unexpected_exception_during_fetch_degrades_without_raising():
    def handler(request):
        raise ConnectionError("network is unreachable")

    searcher = FakeSearcher(results=[{"href": "https://example.com/page"}])
    client = make_client(handler)

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    assert result == {"scraper_data": ""}


# --- KeyboardInterrupt / SystemExit must NOT be swallowed -------------------


def test_keyboard_interrupt_from_searcher_is_not_swallowed():
    searcher = FakeSearcher(exc=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        run_research("some product", "a scene", searcher=searcher)


def test_keyboard_interrupt_from_client_is_not_swallowed():
    def handler(request):
        raise KeyboardInterrupt()

    searcher = FakeSearcher(results=[{"href": "https://example.com/page"}])
    client = make_client(handler)

    with pytest.raises(KeyboardInterrupt):
        run_research("some product", "a scene", searcher=searcher, client=client)


def test_system_exit_from_searcher_is_not_swallowed():
    searcher = FakeSearcher(exc=SystemExit())

    with pytest.raises(SystemExit):
        run_research("some product", "a scene", searcher=searcher)


# --- Happy path ---------------------------------------------------------------


def test_happy_path_returns_extracted_scraper_data():
    def handler(request):
        return html_response(
            body=(
                "<html><head><style>body{color:red}</style></head>"
                "<body><script>evil()</script>"
                "<h1>Acme Widget Pro 3000</h1>"
                "<p>The Acme Widget Pro 3000 is a bestselling gadget "
                "released in 2024 with a five year warranty.</p>"
                "</body></html>"
            )
        )

    searcher = FakeSearcher(results=[{"href": "https://example.com/widget", "title": "Acme Widget Pro 3000"}])
    client = make_client(handler)

    result = run_research("Acme Widget Pro 3000", "a product box on a table", searcher=searcher, client=client)

    assert searcher.calls[0]["query"] == "Acme Widget Pro 3000"
    assert "Acme Widget Pro 3000" in result["scraper_data"]
    assert "bestselling gadget" in result["scraper_data"]
    # Script/style content must not leak into the extracted text.
    assert "evil()" not in result["scraper_data"]
    assert "color:red" not in result["scraper_data"]


def test_happy_path_falls_back_to_empty_body_html_yields_empty_scraper_data():
    def handler(request):
        return html_response(body="<html><head></head><body></body></html>")

    searcher = FakeSearcher(results=[{"href": "https://example.com/blank"}])
    client = make_client(handler)

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    assert result == {"scraper_data": ""}


# --- Client lifecycle --------------------------------------------------------


def test_self_constructed_client_is_closed(monkeypatch):
    def handler(request):
        return html_response()

    fake_client = make_client(handler)
    monkeypatch.setattr(research, "_default_client", lambda: fake_client)
    searcher = FakeSearcher(results=[{"href": "https://example.com/page"}])

    run_research("some product", "a scene", searcher=searcher)

    assert fake_client.is_closed


def test_injected_client_is_not_closed():
    def handler(request):
        return html_response()

    client = make_client(handler)
    searcher = FakeSearcher(results=[{"href": "https://example.com/page"}])

    run_research("some product", "a scene", searcher=searcher, client=client)

    assert not client.is_closed


# --- research_node: client/searcher injection, graph-facing wrapper --------
#
# research_node is a node of the DEEP-PATH CHILD GRAPH since issue #84 / P9.5
# (clarif_eye.deep_path), so the state it is handed uses the child's own key
# names - see the same note in tests/test_analysis.py.


def test_research_node_accepts_explicit_injected_searcher_and_client():
    def handler(request):
        return html_response()

    client = make_client(handler)
    searcher = FakeSearcher(results=[{"href": "https://example.com/page"}])
    state = {"document_text": "some product", "scene_description": "a scene"}

    result = research_node(state, searcher=searcher, client=client)

    assert result["scraper_data"] != ""


def test_research_node_accepts_searcher_and_client_via_config_configurable():
    def handler(request):
        return html_response()

    client = make_client(handler)
    searcher = FakeSearcher(results=[{"href": "https://example.com/page"}])
    state = {"document_text": "some product", "scene_description": "a scene"}

    result = research_node(
        state, config={"configurable": {"searcher": searcher, "research_client": client}}
    )

    assert result["scraper_data"] != ""


def test_research_node_degrades_gracefully_with_no_injected_seams_and_no_query():
    # No searcher/client injected AND no query derivable (empty state) -
    # must short-circuit before ever constructing a real DDGS()/httpx.Client().
    state = {"document_text": "", "scene_description": ""}

    result = research_node(state)

    assert result == {"scraper_data": ""}


# --- Full compiled graph: research path end to end, with fakes -------------


def test_full_compiled_graph_runs_end_to_end_on_research_path_with_fakes():
    long_ocr_text = " ".join(["invoice", "total", "$1234.56"] * 20)  # trips complexity heuristic

    class FakeVisionAnalysisClient:
        def __init__(self):
            self.calls = []

        def complete(self, role, messages, **params):
            self.calls.append(role)
            if len(self.calls) == 1:
                return CompletionResult(
                    content=f"OCR_TEXT: {long_ocr_text}\nSCENE: a dense invoice document",
                    model="fake-eyes",
                )
            return CompletionResult(
                content="The image shows an invoice with a total amount due.",
                model="fake-brain",
            )

        def close(self):
            pass

    def handler(request):
        return html_response(
            body="<html><body><p>Background info about invoices and totals.</p></body></html>"
        )

    searcher = FakeSearcher(results=[{"href": "https://example.com/invoices"}])
    fetch_client = make_client(handler)
    vision_client = FakeVisionAnalysisClient()

    graph = build_graph()
    state = make_initial_state("base64data")
    config = {
        "configurable": {
            "client": vision_client,
            "searcher": searcher,
            "research_client": fetch_client,
        }
    }

    result, trace = drain_stream_collecting_trace(graph, state, config)

    # "deep_path" is the parent's own node completing AFTER the two child
    # nodes it contains (issue #84 / P9.5) - the trace helper streams with
    # subgraphs=True so the child's nodes stay visible, then the node that
    # holds them reports its own completion. Nothing about the route changed.
    assert trace == DEEP_TRACE
    assert result["scraper_data"] != ""
    assert "Background info" in result["scraper_data"]
    assert result["final_output"] != ""


# --- Contract decision (#10, revised by #81/P9.2): scraper_data now
# distinguishes "never ran" (None) from "ran and found nothing" ("") -
# this module's OWN behavior (what run_research returns) is unchanged;
# only the value it starts from before it ever runs is now different from
# the value it produces when it runs and comes up empty. See state.py's
# ClarifEyeState.scraper_data comment for the full rationale: analysis.py
# still treats both as "no external context available" and proceeds
# identically either way, so no consumer's BEHAVIOR depends on this
# distinction - it exists so callers that inspect state directly (a future
# feature, or a human debugging a run) aren't left guessing.


def test_scraper_data_never_ran_is_none_ran_and_found_nothing_is_empty_string():
    not_applicable = make_initial_state("data")["scraper_data"]

    searcher = FakeSearcher(results=[])
    ran_and_found_nothing = run_research("some product", "a scene", searcher=searcher)["scraper_data"]

    assert not_applicable is None
    assert ran_and_found_nothing == ""
    assert not_applicable != ran_and_found_nothing
    assert isinstance(ran_and_found_nothing, str)


# --- SSRF hardening ------------------------------------------------------
#
# The fetched URL comes from a DuckDuckGo result for a query derived from
# ocr_output - attacker-controlled by photographing arbitrary text. No real
# DNS or network access happens in any of these: socket.getaddrinfo is
# monkeypatched to a fixed fake map, and the HTTP client is always an
# httpx.MockTransport.
#
# IMPORTANT: the handler records calls into a list rather than raising
# AssertionError on an unexpected request. run_research wraps the fetch in
# `except Exception`, so a raised AssertionError would be silently
# swallowed and degrade to the exact same {"scraper_data": ""} as a correct
# rejection - masking the vulnerability instead of proving it's fixed (this
# was caught during CHECK F: the first version of these tests still
# "passed" against the vulnerable code). Asserting on the recorded call
# list is what actually distinguishes "rejected before fetch" from "fetch
# attempted and its result happened to be discarded".


def no_fetch_client():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return html_response()

    client = make_client(handler)
    client.ssrf_test_calls = calls
    return client


@pytest.mark.parametrize(
    "url,hostname,ips",
    [
        ("http://127.0.0.1/", "127.0.0.1", ["127.0.0.1"]),
        ("http://localhost/", "localhost", ["127.0.0.1"]),
        ("http://169.254.169.254/latest/meta-data/", "169.254.169.254", ["169.254.169.254"]),
        ("http://10.0.0.5/", "10.0.0.5", ["10.0.0.5"]),
        ("http://192.168.1.1/", "192.168.1.1", ["192.168.1.1"]),
        ("http://172.16.0.1/", "172.16.0.1", ["172.16.0.1"]),
        ("http://[::1]/", "::1", ["::1"]),
        ("http://[fd00::1]/", "fd00::1", ["fd00::1"]),
    ],
)
def test_ssrf_blocks_addresses_in_blocked_ranges(monkeypatch, url, hostname, ips):
    monkeypatch.setattr(research.socket, "getaddrinfo", fake_getaddrinfo({hostname: ips}))
    searcher = FakeSearcher(results=[{"href": url}])
    client = no_fetch_client()

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    assert result == {"scraper_data": ""}
    assert client.ssrf_test_calls == []


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/",
        "data:text/html,x",
    ],
)
def test_ssrf_blocks_non_http_schemes(url):
    # These never even reach _fetch_and_extract: _search_top_result already
    # filters hrefs to http(s)-only, so the search "finds nothing usable".
    # Directly exercise the scheme check too, since that's the actual new
    # security control (defence in depth, not reliance on the search filter).
    assert research._is_safe_url(url) is False

    searcher = FakeSearcher(results=[{"href": url}])
    client = no_fetch_client()

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    assert result == {"scraper_data": ""}
    assert client.ssrf_test_calls == []


def test_ssrf_blocks_uppercase_localhost_with_trailing_dot(monkeypatch):
    monkeypatch.setattr(research.socket, "getaddrinfo", fake_getaddrinfo({"localhost": ["127.0.0.1"]}))
    searcher = FakeSearcher(results=[{"href": "http://LOCALHOST./"}])
    client = no_fetch_client()

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    assert result == {"scraper_data": ""}
    assert client.ssrf_test_calls == []


def test_ssrf_blocks_ip_literal_with_trailing_dot_and_port(monkeypatch):
    monkeypatch.setattr(research.socket, "getaddrinfo", fake_getaddrinfo({"127.0.0.1": ["127.0.0.1"]}))
    searcher = FakeSearcher(results=[{"href": "http://127.0.0.1.:80/"}])
    client = no_fetch_client()

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    assert result == {"scraper_data": ""}
    assert client.ssrf_test_calls == []


def test_ssrf_blocks_public_url_redirecting_to_metadata_endpoint(monkeypatch):
    monkeypatch.setattr(
        research.socket,
        "getaddrinfo",
        fake_getaddrinfo({"example.com": ["93.184.216.34"], "169.254.169.254": ["169.254.169.254"]}),
    )
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.host == "example.com":
            return redirect_response("http://169.254.169.254/latest/meta-data/")
        return html_response()

    searcher = FakeSearcher(results=[{"href": "http://example.com/redirect-me"}])
    client = make_client(handler)

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    assert result == {"scraper_data": ""}
    # The redirect target must never actually be requested - blocked at the
    # hop, before the second fetch.
    assert calls == ["http://example.com/redirect-me"]


def test_ssrf_blocks_redirect_chain_longer_than_hop_cap(monkeypatch):
    monkeypatch.setattr(research.socket, "getaddrinfo", fake_getaddrinfo({"example.com": ["93.184.216.34"]}))
    calls = []

    def handler(request):
        calls.append(str(request.url))
        # Always redirect to a new path on the same (safe) host - a chain
        # longer than research._MAX_REDIRECT_HOPS.
        n = len(calls)
        return redirect_response(f"http://example.com/hop-{n + 1}")

    searcher = FakeSearcher(results=[{"href": "http://example.com/hop-1"}])
    client = make_client(handler)

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    assert result == {"scraper_data": ""}
    # initial fetch + _MAX_REDIRECT_HOPS redirects = the bounded number of
    # attempts; the chain never runs away unbounded.
    assert len(calls) == research._MAX_REDIRECT_HOPS + 1


def test_benign_single_redirect_still_succeeds(monkeypatch):
    monkeypatch.setattr(research.socket, "getaddrinfo", fake_getaddrinfo({"example.com": ["93.184.216.34"]}))

    def handler(request):
        if str(request.url) == "http://example.com/old-page":
            return redirect_response("http://example.com/new-page")
        return html_response(body="<html><body><p>The real content lives here.</p></body></html>")

    searcher = FakeSearcher(results=[{"href": "http://example.com/old-page"}])
    client = make_client(handler)

    result = run_research("some product", "a scene", searcher=searcher, client=client)

    assert "The real content lives here." in result["scraper_data"]
