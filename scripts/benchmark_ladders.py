"""MANUAL-ONLY reusable ladder benchmark for Clarif-Eye (issue P1.6 / #9).

Free OpenRouter models churn monthly, so re-measuring the "eyes" and
"brain" ladders needs to be one command, not an ad-hoc probe rebuilt from
scratch each time. This script makes REAL network calls to OpenRouter
using OPENROUTER_API_KEY from the environment. It is NOT part of the
pytest suite (tests must stay offline, see tests/test_benchmark_script.py
for the offline drift-guard tests) and must NOT be run by an automated
agent - only by a human, or an orchestrator that has explicitly decided to
spend real API calls.

Model IDs are NEVER hardcoded here: every rung benchmarked comes from
clarif_eye.registry.load_registry(), so this script cannot silently drift
from src/clarif_eye/config/models.toml.

Brain replies are scored with the PRODUCTION verifier,
clarif_eye.analysis._numbers_verified, not ad-hoc string matching. An
earlier ad-hoc probe (predating this script) misjudged a correct reply
that spelled amounts out in words ("one hundred four dollars and ninety
five cents" instead of "$104.95") as a failure, because it was matching
literal digit substrings instead of using the real verifier. Reusing the
production function is the only way this script's verdicts stay trustworthy.

Timeout: httpx is used directly with a generous 90s timeout (see
_TIMEOUT_SECONDS below). This is deliberately NOT clarif_eye.client's
ROLE_TIMEOUTS budget - the point of this script is to MEASURE how long a
rung actually takes, including slow outliers, not to enforce or emulate
the production latency budget.

Usage:
    OPENROUTER_API_KEY=... python scripts/benchmark_ladders.py \\
        --role all --runs 2 --image path/to/photo.jpg
"""

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from clarif_eye.analysis import _build_messages as _brain_messages
from clarif_eye.analysis import _numbers_verified
from clarif_eye.client import API_BASE_URL, OpenRouterClient
from clarif_eye.registry import load_registry
from clarif_eye.vision import _build_messages as _eyes_messages

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
BRAIN_FIXTURE_PATH = FIXTURES_DIR / "vision_reply_parsed.json"

# Generous on purpose - see module docstring. Not a production budget.
_TIMEOUT_SECONDS = 90.0


@dataclass
class RunResult:
    """One rung's one run - latency, HTTP status, reply length, and (brain
    only) the production-verifier verdict."""

    role: str
    model: str
    run_index: int
    latency_s: float
    status_code: int | None
    reply_len: int
    success: bool
    verified: bool | None = None
    error: str | None = None


def build_plan(registry, role):
    """Return the ordered list of (role, model) rungs to benchmark.

    Always derived from `registry.ladder(...)` - never a hardcoded model
    list - so this script can never drift from src/clarif_eye/config/
    models.toml. `role` is "eyes", "brain", or "all".
    """
    roles = ("eyes", "brain") if role == "all" else (role,)
    plan = []
    for r in roles:
        for model in registry.ladder(r):
            plan.append((r, model))
    return plan


def load_brain_fixture(path=BRAIN_FIXTURE_PATH):
    """Load the recorded vision fixture's ocr_output/scene_context for the brain ladder."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["ocr_output"], data["scene_context"]


def score_brain_reply(reply, ocr_output, scene_context):
    """Score a brain reply with the PRODUCTION verifier, not ad-hoc string matching.

    See module docstring: an earlier ad-hoc probe misjudged a correct
    spelled-out-amounts reply because it matched literal digit substrings.
    Only clarif_eye.analysis._numbers_verified is trusted for a verdict.
    """
    return _numbers_verified(reply, ocr_output, scene_context, "")


def build_messages(role, *, image_b64=None, ocr_output=None, scene_context=None):
    """Build the production-shaped request body for `role`.

    Reuses the real node's own message builder (vision._build_messages /
    analysis._build_messages) so the benchmarked request is byte-for-byte
    what production sends, never a re-derived approximation.
    """
    if role == "eyes":
        return _eyes_messages(image_b64)
    return _brain_messages(ocr_output, scene_context, "")


def build_http_client(timeout=_TIMEOUT_SECONDS):
    """Construct the httpx client used to send benchmark requests. I/O only - never called at import time."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key or not api_key.strip():
        raise SystemExit("OPENROUTER_API_KEY is required and must not be blank")
    return httpx.Client(
        base_url=API_BASE_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
    )


def run_rung(http_client, role, model, run_index, messages, *, ocr_output=None, scene_context=None):
    """Send one production-shaped request for one rung and record the outcome."""
    start = time.monotonic()
    status_code = None
    content = None
    error = None
    try:
        response = http_client.post("/chat/completions", json={"model": model, "messages": messages})
        status_code = response.status_code
        if status_code == 200:
            result = OpenRouterClient._parse_success(response, model)
            content = result.content if result is not None else None
    except httpx.HTTPError as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_s = time.monotonic() - start

    verified = None
    if role == "brain" and content is not None:
        verified = score_brain_reply(content, ocr_output, scene_context)

    return RunResult(
        role=role,
        model=model,
        run_index=run_index,
        latency_s=latency_s,
        status_code=status_code,
        reply_len=len(content) if content else 0,
        success=content is not None,
        verified=verified,
        error=error,
    )


def any_rung_never_succeeded(results, plan):
    """True if some (role, model) rung in `plan` had zero successful runs."""
    for role, model in plan:
        rung_results = [r for r in results if r.role == role and r.model == model]
        if not any(r.success for r in rung_results):
            return True
    return False


def print_summary(results):
    header = f"{'role':<6} {'model':<50} {'run':>3} {'latency_s':>9} {'status':>6} {'reply_len':>9} {'ok':>5} {'verified':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.role:<6} {r.model:<50} {r.run_index:>3} {r.latency_s:>9.2f} "
            f"{str(r.status_code):>6} {r.reply_len:>9} {str(r.success):>5} {str(r.verified):>8}"
            + (f"  [{r.error}]" if r.error else "")
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("eyes", "brain", "all"), default="all")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--image", type=Path, help="Image path, required if --role includes eyes")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    registry = load_registry()
    plan = build_plan(registry, args.role)

    image_b64 = None
    if args.role in ("eyes", "all"):
        if args.image is None:
            raise SystemExit("--image is required when benchmarking the eyes ladder")
        image_b64 = base64.b64encode(args.image.read_bytes()).decode("ascii")

    ocr_output = scene_context = None
    if args.role in ("brain", "all"):
        ocr_output, scene_context = load_brain_fixture()

    http_client = build_http_client()
    results = []
    try:
        for role, model in plan:
            messages = build_messages(
                role, image_b64=image_b64, ocr_output=ocr_output, scene_context=scene_context
            )
            for run_index in range(args.runs):
                results.append(
                    run_rung(
                        http_client,
                        role,
                        model,
                        run_index,
                        messages,
                        ocr_output=ocr_output,
                        scene_context=scene_context,
                    )
                )
    finally:
        http_client.close()

    print_summary(results)

    if any_rung_never_succeeded(results, plan):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
