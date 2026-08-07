"""MANUAL-ONLY full-history secret and personal-email scan (issue #19 / P7.1).

Scans every blob reachable from every ref (`git rev-list --objects --all`)
for credential patterns, and every commit's author/committer email for
anything other than a GitHub noreply address, before this repository stops
being private. Makes no network call; reads only the local git object
database via `git cat-file` and `git log`.

Usage:
    python scripts/scan_history.py

Exits 1 and prints every finding if anything is found. Exits 0 and prints
the object/commit counts scanned if clean.
"""

import re
import subprocess

SECRET_PATTERNS = [
    ("OpenRouter key", re.compile(r"sk-or-v1-[A-Za-z0-9]+")),
    ("Hugging Face token", re.compile(r"hf_[A-Za-z0-9]+")),
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]+")),
    ("AWS access key ID", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("Private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("Non-empty OPENROUTER_API_KEY", re.compile(r"^OPENROUTER_API_KEY=\S+", re.MULTILINE)),
]

# Commits are authored with a GitHub noreply address, never a personal one.
# `noreply@github.com` is GitHub's own committer address for merges made
# through the GitHub web UI, not a personal address.
ALLOWED_EMAILS_RE = re.compile(r"(@users\.noreply\.github\.com$|^noreply@github\.com$)")


def run(args):
    return subprocess.run(args, capture_output=True, check=True)


def scan_blobs():
    """Greps the content of every blob reachable from any ref. Returns a
    list of (object_sha, pattern_name, snippet) findings and the number of
    blobs scanned."""
    objects = run(["git", "rev-list", "--objects", "--all"]).stdout.decode("utf-8", "replace")
    shas = [line.split()[0] for line in objects.splitlines() if line.strip()]

    check_input = "\n".join(shas).encode()
    proc = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
        input=check_input,
        capture_output=True,
        check=True,
    )
    blob_shas = [
        line.split()[0]
        for line in proc.stdout.decode("utf-8", "replace").splitlines()
        if line.endswith(" blob")
    ]

    findings = []
    for sha in blob_shas:
        content = run(["git", "cat-file", "-p", sha]).stdout.decode("utf-8", "replace")
        for name, pattern in SECRET_PATTERNS:
            match = pattern.search(content)
            if match:
                findings.append((sha, name, match.group(0)[:40]))
    return findings, len(blob_shas)


def scan_commit_emails():
    """Returns (bad_emails, all_emails) across every commit on every ref."""
    log = run(["git", "log", "--all", "--format=%H %ae %ce"]).stdout.decode("utf-8", "replace")
    lines = [line for line in log.splitlines() if line.strip()]
    all_emails = set()
    bad = []
    for line in lines:
        commit, author_email, committer_email = line.split(" ", 2)
        all_emails.add(author_email)
        all_emails.add(committer_email)
        if not ALLOWED_EMAILS_RE.search(author_email):
            bad.append((commit, "author", author_email))
        if not ALLOWED_EMAILS_RE.search(committer_email):
            bad.append((commit, "committer", committer_email))
    return bad, all_emails, len(lines)


def main():
    secret_findings, blob_count = scan_blobs()
    bad_emails, all_emails, commit_count = scan_commit_emails()

    print(f"Scanned {blob_count} blobs across all refs for credential patterns.")
    print(f"Scanned {commit_count} commits across all refs for author/committer email.")
    print(f"Distinct commit emails found: {sorted(all_emails)}")

    ok = True
    if secret_findings:
        ok = False
        print("\nCredential findings:")
        for sha, name, snippet in secret_findings:
            print(f"  blob {sha}: {name} matched {snippet!r}")
    else:
        print("No credential patterns found in any blob.")

    if bad_emails:
        ok = False
        print("\nNon-noreply commit emails found:")
        for commit, role, email in bad_emails:
            print(f"  {commit} {role}: {email}")
    else:
        print("All commit author/committer emails are GitHub noreply addresses.")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
