"""Offline drift-guard tests for scripts/benchmark_ladders.py (issue P1.6 / #9).

These tests import the script's helpers WITHOUT calling main() and WITHOUT
any network access, and assert two things a reusable benchmark script must
guarantee:

1. The model list it would benchmark comes from clarif_eye.registry.
   load_registry() - never a hardcoded list - so the script cannot
   silently drift from src/clarif_eye/config/models.toml.
2. Brain replies are scored via the PRODUCTION verifier,
   clarif_eye.analysis._numbers_verified, not ad-hoc string matching (see
   the script's module docstring for why that matters).
"""

import scripts.benchmark_ladders as benchmark_ladders
from clarif_eye.registry import ModelRegistry


class FakeRegistry:
    """Stand-in registry whose ladder() is trivially distinguishable from the real config."""

    def __init__(self, ladders):
        self._ladders = ladders

    def ladder(self, role):
        return self._ladders[role]


def test_build_plan_uses_registry_for_eyes():
    """The eyes plan must be exactly what the injected registry reports - the drift guard."""
    fake_registry = FakeRegistry({"eyes": ("fake/eyes-model-a:free", "fake/eyes-model-b:free")})

    plan = benchmark_ladders.build_plan(fake_registry, "eyes")

    assert plan == [
        ("eyes", "fake/eyes-model-a:free"),
        ("eyes", "fake/eyes-model-b:free"),
    ]


def test_build_plan_uses_registry_for_brain():
    fake_registry = FakeRegistry({"brain": ("fake/brain-model-a:free",)})

    plan = benchmark_ladders.build_plan(fake_registry, "brain")

    assert plan == [("brain", "fake/brain-model-a:free")]


def test_build_plan_for_all_covers_both_roles_in_registry_order():
    fake_registry = FakeRegistry(
        {
            "eyes": ("fake/eyes-model:free",),
            "brain": ("fake/brain-model:free",),
        }
    )

    plan = benchmark_ladders.build_plan(fake_registry, "all")

    assert plan == [("eyes", "fake/eyes-model:free"), ("brain", "fake/brain-model:free")]


def test_build_plan_reflects_the_real_registry_shape():
    """Sanity check against a real ModelRegistry (not just the fake), still no network."""
    real_registry = ModelRegistry(
        {
            "eyes": ["real/eyes-model:free"],
            "brain": ["real/brain-model:free"],
        }
    )

    plan = benchmark_ladders.build_plan(real_registry, "all")

    assert plan == [("eyes", "real/eyes-model:free"), ("brain", "real/brain-model:free")]


def test_score_brain_reply_calls_production_verifier(monkeypatch):
    """The drift guard for scoring: score_brain_reply must call analysis._numbers_verified."""
    calls = []

    def fake_verifier(spoken_output, ocr_output, scene_context, scraper_data):
        calls.append((spoken_output, ocr_output, scene_context, scraper_data))
        return "sentinel-verdict"

    monkeypatch.setattr(benchmark_ladders, "_numbers_verified", fake_verifier)

    result = benchmark_ladders.score_brain_reply("reply text", "ocr text", "scene text")

    assert result == "sentinel-verdict"
    assert calls == [("reply text", "ocr text", "scene text", "")]


def test_run_rung_scores_brain_replies_with_the_production_verifier(monkeypatch):
    """End-to-end (still offline): run_rung must route brain scoring through _numbers_verified."""
    calls = []

    def fake_verifier(spoken_output, ocr_output, scene_context, scraper_data):
        calls.append(spoken_output)
        return True

    monkeypatch.setattr(benchmark_ladders, "_numbers_verified", fake_verifier)

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "a reply"}}]}

    class FakeHttpClient:
        def post(self, path, json):
            return FakeResponse()

    result = benchmark_ladders.run_rung(
        FakeHttpClient(),
        "brain",
        "fake/brain-model:free",
        0,
        messages=[{"role": "user", "content": "hi"}],
        ocr_output="ocr",
        scene_context="scene",
    )

    assert calls == ["a reply"]
    assert result.verified is True
    assert result.success is True
    assert result.reply_len == len("a reply")


def test_any_rung_never_succeeded_true_when_a_rung_has_no_success():
    plan = [("eyes", "model-a:free"), ("eyes", "model-b:free")]
    results = [
        benchmark_ladders.RunResult(
            role="eyes", model="model-a:free", run_index=0, latency_s=1.0,
            status_code=200, reply_len=10, success=True,
        ),
        benchmark_ladders.RunResult(
            role="eyes", model="model-b:free", run_index=0, latency_s=1.0,
            status_code=500, reply_len=0, success=False,
        ),
    ]

    assert benchmark_ladders.any_rung_never_succeeded(results, plan) is True


def test_any_rung_never_succeeded_false_when_every_rung_has_a_success():
    plan = [("eyes", "model-a:free")]
    results = [
        benchmark_ladders.RunResult(
            role="eyes", model="model-a:free", run_index=0, latency_s=1.0,
            status_code=500, reply_len=0, success=False,
        ),
        benchmark_ladders.RunResult(
            role="eyes", model="model-a:free", run_index=1, latency_s=1.0,
            status_code=200, reply_len=10, success=True,
        ),
    ]

    assert benchmark_ladders.any_rung_never_succeeded(results, plan) is False
