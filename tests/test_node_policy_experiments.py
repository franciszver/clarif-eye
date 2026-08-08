"""Offline drift-guard tests for the P9.6 node-policy experiments (issue #85).

Each policy (retry_policy, timeout, cache_policy, error_handler) has one
experiment script under scripts/ that builds a minimal langgraph graph
exercising that policy against a controllable fake node, runs entirely
offline (no live API calls), and prints a VERDICT. This test does NOT
re-run the experiments (they print to stdout, not a return value worth
importing) - it pins the two things that make the verdicts trustworthy
rather than merely asserted:

1. Every experiment script imports cleanly (catches syntax/import errors
   the way tests/test_scripts_import.py does for the rest of scripts/).
2. Its recorded output file exists under scripts/experiments/ and names a
   verdict, so "adopt / keep ours / hybrid" is reproducible evidence on
   disk, not a claim made in a PR description.
"""

from pathlib import Path

import scripts.experiment_cache_policy as experiment_cache_policy
import scripts.experiment_error_handler as experiment_error_handler
import scripts.experiment_retry_policy as experiment_retry_policy
import scripts.experiment_timeout as experiment_timeout

REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"

# (module, recorded-output filename)
EXPERIMENTS = [
    (experiment_retry_policy, "retry_policy.txt"),
    (experiment_timeout, "timeout.txt"),
    (experiment_cache_policy, "cache_policy.txt"),
    (experiment_error_handler, "error_handler.txt"),
]


def test_every_experiment_script_defines_main():
    """Each script must expose a callable main() - the entry point the
    recorded output under scripts/experiments/ was produced by."""
    for module, _filename in EXPERIMENTS:
        assert callable(module.main), f"{module.__name__} has no callable main()"


def test_every_experiment_output_file_exists_and_names_a_verdict():
    verdicts = ("ADOPT", "KEEP OURS", "HYBRID")
    for _module, filename in EXPERIMENTS:
        path = EXPERIMENTS_DIR / filename
        assert path.exists(), f"missing recorded output: {path}"
        text = path.read_text(encoding="utf-8")
        assert "VERDICT" in text, f"{path} does not record a VERDICT"
        assert any(v in text for v in verdicts), (
            f"{path} does not name one of {verdicts}"
        )


def test_every_experiment_output_file_records_the_command_that_produced_it():
    """The recorded output must start with the command line used to produce
    it, so the verdict is reproducible rather than asserted (issue #85)."""
    for _module, filename in EXPERIMENTS:
        path = EXPERIMENTS_DIR / filename
        text = path.read_text(encoding="utf-8")
        first_line = text.splitlines()[0]
        assert "python scripts/experiment_" in first_line, (
            f"{path} does not open with the command that produced it: {first_line!r}"
        )
