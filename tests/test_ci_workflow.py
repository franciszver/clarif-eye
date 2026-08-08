"""Failing-first checks for the GitHub Actions CI workflow (issue #21 /
P8.1). CI must run lint and the test suite against committed fixtures with
NO live model calls and NO API key, so this file checks the workflow file
itself: it exists, is valid YAML, pins the same Python version render.yaml
pins, scans full git history (not just the last commit) for the secret
scan step, and never references a secret or API key. A second check
confirms the README carries a CI badge linking to the workflow. No network
call and no server launched here.
"""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def test_ci_workflow_exists_and_is_valid_yaml():
    assert WORKFLOW_PATH.exists(), ".github/workflows/ci.yml is missing"

    with open(WORKFLOW_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data.get("jobs"), "ci.yml declares no jobs"


def test_ci_workflow_pins_python_3_11():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "3.11" in text, "ci.yml must pin Python 3.11, matching render.yaml"


def test_ci_workflow_checks_out_full_history_for_the_secret_scan():
    with open(WORKFLOW_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    found_full_checkout = False
    for job in data["jobs"].values():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/checkout"):
                if step.get("with", {}).get("fetch-depth") == 0:
                    found_full_checkout = True

    assert found_full_checkout, (
        "ci.yml must check out with fetch-depth: 0, or scan_history.py "
        "only scans the last commit and passes meaninglessly"
    )


def test_ci_workflow_runs_the_secret_scan():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "scan_history.py" in text


def test_ci_workflow_runs_pytest():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "pytest" in text


def test_ci_workflow_runs_ruff():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "ruff" in text


def test_ci_workflow_never_references_a_secret_or_api_key():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY" not in text
    # The deploy job legitimately references secrets.RENDER_DEPLOY_HOOK (the
    # hook URL itself is never in this file); no other secret is referenced.
    secret_refs = re.findall(r"secrets\.(\w+)", text)
    assert secret_refs == ["RENDER_DEPLOY_HOOK"]


def test_ci_workflow_has_a_deploy_job_gated_to_main_pushes_after_test():
    with open(WORKFLOW_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    deploy = data["jobs"].get("deploy")
    assert deploy, "ci.yml must declare a deploy job"

    needs = deploy.get("needs")
    assert needs == "test" or needs == ["test"]

    condition = deploy.get("if", "")
    assert "refs/heads/main" in condition
    assert "github.event_name == 'push'" in condition

    steps = deploy.get("steps", [])
    assert steps, "deploy job must have at least one step"
    deploy_step = steps[0]
    run_text = deploy_step.get("run", "")
    assert "curl" in run_text
    assert "-fsS" in run_text
    assert "--max-time" in run_text
    assert "$RENDER_DEPLOY_HOOK" in run_text

    env = deploy_step.get("env", {})
    assert env.get("RENDER_DEPLOY_HOOK") == "${{ secrets.RENDER_DEPLOY_HOOK }}"


def test_ci_workflow_has_no_continue_on_error():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "continue-on-error" not in text


def test_readme_has_a_ci_badge_linking_to_the_workflow():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "workflows/ci.yml/badge.svg" in text
    assert "actions/workflows/ci.yml" in text
