"""MANUAL-ONLY full-pipeline latency/accuracy benchmark for Clarif-Eye
(issue P6.1 / #17).

A live measurement of the real Gradio UI found a research-path request
taking 99.0s end to end, and a follow-up single-sample probe of the
analysis node alone found scraper_data capped at 4000 chars costing
82.6s vs 7.6s with none, for near-identical output. Both numbers are
SINGLE SAMPLES against a free-tier OpenRouter backend whose queueing is
highly variable - a repeat sweep in the same investigation saw
scraper-cap=500 come back slower (53.4s) than scraper-cap=1000 (12.7s),
which is noise, not a real relationship between cap size and latency.
Single samples proved nothing then and prove nothing here either.

A follow-up n=5 sweep (median of 5 runs per configuration, this script,
production verifier) confirmed exactly that: the 99.0s figure was an
OUTLIER, not a typical run - the measured median for the research path is
~21-31s across cap=0/1000/4000 (min/max spans 19-60s, queue noise
dominates), and accuracy was verified 5/5 at every cap tested, including
cap=0. See analysis.py's _SCRAPER_DATA_CAP comment for the corrected
justification. The lesson from the single-sample era stands regardless:
single samples against this noisy free-tier backend prove nothing -
always sweep n>=5 and report median/min/max.

This script exists so that question can be answered properly: it drives
the REAL compiled graph (clarif_eye.graph.build_graph()) end to end -
real OpenRouter calls for vision/fast_synth/analysis, a real DuckDuckGo
search + page fetch for research, real TTS synthesis - and, for EVERY
configuration swept, takes at least MIN_RUNS (default 5, enforced)
samples and reports the MEDIAN plus MIN/MAX, never a single number.

ACCURACY is scored with the PRODUCTION verifier,
clarif_eye.analysis._numbers_verified, run against the graph's own
returned state (ocr_output/scene_context/scraper_data/final_output) -
never literal substring matching. A model that spells "$104.95" as "one
hundred four dollars and ninety five cents", or reformats a date, is
CORRECT (see D17/D18) - a substring check would misjudge that as a
failure, exactly the mistake this project has already made twice.

This script makes REAL network calls (OpenRouter, DuckDuckGo, an
arbitrary web page, edge-tts/gTTS) using OPENROUTER_API_KEY from the
environment. It is NOT part of the pytest suite and must NOT be run by
an automated agent - only by a human, or an orchestrator that has
explicitly decided to spend real API calls and real wall-clock time
(this can take many minutes: n>=5 runs, times however many
configurations are swept, times however long a research-path request
takes).

WHAT IT SWEEPS
----------------
--scraper-caps accepts a comma-separated list of scraper_data_cap values
(config["configurable"]["scraper_data_cap"], see analysis.py) and the
script runs the full --runs sample count at EACH value, against the SAME
research-path photo, so the with/without-context question (and the
4000-vs-smaller question) gets an actual n>=5 median comparison instead
of another single sample.

--pipeline-budget sets config["configurable"]["deadline"]
(time.monotonic() + budget) for every run, defaulting to
clarif_eye.graph.DEFAULT_PIPELINE_BUDGET_SECONDS - so this script can also
be used to check how often a given budget actually gets hit for a given
photo/cap combination (a run whose deadline_hit flag is True still
"succeeds", just in degraded form - see PER-STAGE TIMING below).

PER-STAGE TIMING
-------------------
graph.stream(state, config=config, stream_mode="updates") (issue #80 /
P9.1) yields one dict per node the MOMENT it completes, keyed by node
name - so this script times each stage by stamping time.monotonic() as
each chunk arrives, giving vision/research/analysis/fast_synth/tts timings
from a single instrumented run, not separate re-runs. A completion
timestamp is the opposite of the old entry-based trace this replaced: a
node's own duration is the gap SINCE the previous completion (or `start`,
the wall-clock moment right before the run began, for the first node) -
see _stage_durations below.

Usage:
    OPENROUTER_API_KEY=... python scripts/benchmark_pipeline.py \\
        --fast-image path/to/toxic_label.jpg \\
        --research-image path/to/utility_bill.jpg \\
        --runs 5 --scraper-caps 500,1000,2000,4000
"""

import argparse
import base64
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from clarif_eye.analysis import _numbers_verified
from clarif_eye.client import OpenRouterClient, OpenRouterError
from clarif_eye.graph import DEFAULT_PIPELINE_BUDGET_SECONDS, build_graph
from clarif_eye.state import make_initial_state
from clarif_eye.tts import DEFAULT_PROVIDER_CHAIN

MIN_RUNS = 5


@dataclass
class RunResult:
    """One full graph.invoke() - total latency, per-stage latency, and an
    accuracy verdict scored with the production verifier."""

    label: str
    run_index: int
    total_s: float
    stage_s: dict = field(default_factory=dict)
    verified: bool = True
    final_output_len: int = 0
    deadline_hit: bool = False


def _stage_durations(completions, start):
    """Turn a list of (node_name, completion_timestamp) - each stamped the
    instant that node's stream_mode="updates" chunk arrived, i.e. when it
    FINISHED, never when it started (langgraph's stream API tells you
    completion, not start - see clarif_eye.graph's module docstring) -
    into {node_name: duration_seconds}.

    A node's own duration is the gap SINCE the previous node's completion
    (or `start`, the wall-clock moment right before the graph began, for
    the first node) - not the gap to the NEXT completion, which would
    attribute node i's duration to node i+1. Completion timestamps already
    include the last node's own finish time, so - unlike the old
    entry-based trace this replaced - no separate `end` value is needed to
    measure it.
    """
    durations = {}
    previous = start
    for node_name, timestamp in completions:
        durations[node_name] = timestamp - previous
        previous = timestamp
    return durations


def _load_image_b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def build_resources():
    """Construct the real, shared, injectable seams once - same shape as
    clarif_eye.ui.build_resources, reused here so a benchmark run exercises
    the exact same object graph production does."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key or not api_key.strip():
        raise SystemExit("OPENROUTER_API_KEY is required and must not be blank")
    try:
        client = OpenRouterClient()
    except OpenRouterError as exc:
        raise SystemExit(f"could not construct OpenRouterClient: {exc}") from exc

    from ddgs import DDGS

    searcher = DDGS()

    import httpx

    research_client = httpx.Client()

    tts_providers = [factory() for factory in DEFAULT_PROVIDER_CHAIN]

    return client, searcher, research_client, tts_providers


def run_once(graph, image_b64, *, label, run_index, client, searcher, research_client, tts_providers,
             scraper_data_cap, pipeline_budget_seconds):
    """Run the compiled graph once end to end, instrumented for stage timing."""
    state = make_initial_state(image_b64)
    configurable = {
        "client": client,
        "searcher": searcher,
        "research_client": research_client,
        "tts_providers": tts_providers,
        "deadline": time.monotonic() + pipeline_budget_seconds,
    }
    if scraper_data_cap is not None:
        configurable["scraper_data_cap"] = scraper_data_cap

    start = time.monotonic()
    result = dict(state)
    completions = []
    for chunk in graph.stream(state, config={"configurable": configurable}, stream_mode="updates"):
        for node_name, update in chunk.items():
            result.update(update)
            completions.append((node_name, time.monotonic()))
    end = time.monotonic()

    verified = _numbers_verified(
        result.get("final_output", ""),
        result.get("ocr_output", ""),
        result.get("scene_context", ""),
        result.get("scraper_data", ""),
    )

    return RunResult(
        label=label,
        run_index=run_index,
        total_s=end - start,
        stage_s=_stage_durations(completions, start),
        verified=verified,
        final_output_len=len(result.get("final_output", "")),
        deadline_hit=time.monotonic() >= configurable["deadline"],
    )


def summarize(results, label):
    """Median/min/max total latency (and per-stage) for one configuration's
    n>=len(results) runs - never a single number, per the module docstring."""
    subset = [r for r in results if r.label == label]
    totals = [r.total_s for r in subset]
    stage_names = sorted({name for r in subset for name in r.stage_s})
    print(f"\n=== {label} (n={len(subset)}) ===")
    print(f"  total:  median={statistics.median(totals):.1f}s  min={min(totals):.1f}s  max={max(totals):.1f}s")
    for name in stage_names:
        values = [r.stage_s[name] for r in subset if name in r.stage_s]
        if not values:
            continue
        print(
            f"  {name:<12} median={statistics.median(values):.1f}s  "
            f"min={min(values):.1f}s  max={max(values):.1f}s  (n={len(values)})"
        )
    verified_count = sum(1 for r in subset if r.verified)
    deadline_hits = sum(1 for r in subset if r.deadline_hit)
    print(f"  verified: {verified_count}/{len(subset)}    deadline_hit: {deadline_hits}/{len(subset)}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast-image", type=Path, help="Fast-path photo (e.g. a simple label)")
    parser.add_argument("--research-image", type=Path, help="Research-path photo (e.g. a dense bill)")
    parser.add_argument("--runs", type=int, default=MIN_RUNS, help=f"Samples per configuration (>= {MIN_RUNS})")
    parser.add_argument(
        "--scraper-caps",
        type=str,
        default=None,
        help="Comma-separated scraper_data_cap values to sweep on --research-image, e.g. 500,1000,2000,4000",
    )
    parser.add_argument("--pipeline-budget", type=float, default=DEFAULT_PIPELINE_BUDGET_SECONDS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.runs < MIN_RUNS:
        raise SystemExit(f"--runs must be >= {MIN_RUNS} (free-tier latency is too noisy for fewer samples)")
    if not args.fast_image and not args.research_image:
        raise SystemExit("at least one of --fast-image / --research-image is required")

    client, searcher, research_client, tts_providers = build_resources()
    graph = build_graph()
    results = []

    try:
        if args.fast_image:
            image_b64 = _load_image_b64(args.fast_image)
            for i in range(args.runs):
                results.append(
                    run_once(
                        graph,
                        image_b64,
                        label="fast",
                        run_index=i,
                        client=client,
                        searcher=searcher,
                        research_client=research_client,
                        tts_providers=tts_providers,
                        scraper_data_cap=None,
                        pipeline_budget_seconds=args.pipeline_budget,
                    )
                )

        if args.research_image:
            image_b64 = _load_image_b64(args.research_image)
            caps = (
                [int(c.strip()) for c in args.scraper_caps.split(",")]
                if args.scraper_caps
                else [None]
            )
            for cap in caps:
                label = "research" if cap is None else f"research (cap={cap})"
                for i in range(args.runs):
                    results.append(
                        run_once(
                            graph,
                            image_b64,
                            label=label,
                            run_index=i,
                            client=client,
                            searcher=searcher,
                            research_client=research_client,
                            tts_providers=tts_providers,
                            scraper_data_cap=cap,
                            pipeline_budget_seconds=args.pipeline_budget,
                        )
                    )
    finally:
        client.close()
        research_client.close()

    for label in dict.fromkeys(r.label for r in results):
        summarize(results, label)


if __name__ == "__main__":
    main()
