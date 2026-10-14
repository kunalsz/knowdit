#!/usr/bin/env python3
"""Build a learnable Code4rena dataset from scraped markdown reports, then
learn it in space-bounded batches.

The knowdit C4 loader (`knowdit_project::C4PairedProjectData::from_dataset_dir`)
keys every file on a NUMERIC u32 contest id, not the report slug:

    <dataset>/audits/<id>.json    contest metadata (contestId + title)
    <dataset>/contracts/<id>/     Solidity sources (walked for **/*.sol)
    <dataset>/reports/<id>.md     markdown audit report (optional)

Report files, however, are named by slug (e.g. `2023-07-basin.md`). This script
bridges the two:

  Phase 1 (once)   parse every report -> assign a numeric id -> write all
                   audits/<id>.json and copy all reports/<id>.md.
  Phase 2 (batched) per batch: clone the contest repo(s) into contracts/<id>/,
                   run `knowdit learn c4 --c4-ids <batch>`, then delete the
                   cloned sources so only one batch lives on disk at a time.

Usage:
    scripts/build_c4_dataset.py --reports-dir /path/to/reports/code4rena \\
        --dataset-dir code4rena --database-url sqlite:///kg.db \\
        --knowdit-bin ./target/release/knowdit --batch-size 10

    # Phase 1 only, no writes:
    scripts/build_c4_dataset.py --prepare-only --dry-run

    # Prepare + clone a batch, stop before learning:
    scripts/build_c4_dataset.py --skip-learn --batch-size 2

Git runs non-interactively (no credential/host-key prompts) with a timeout, so
a repo that needs auth is skipped instead of hanging the batch. Every contest
that is not learned is listed in `<dataset>/skipped.txt` for manual handling.

A contest listed in `--skip-ids` (or found in `<dataset>/skip-ids.txt`) is
never cloned or learned, and is recorded as `skipped:manual`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_REPORTS_DIR = "/home/zeit/Desktop/reports/code4rena"
DEFAULT_DATASET_DIR = "code4rena"

# Directory names the C4 loader drops from a project's scope (mirrors
# `DEFAULT_VENDORED_EXCLUDES` in crates/knowdit-project/src/data.rs). A clone
# whose Solidity all lives under one of these is effectively empty to the
# loader, so we detect that and skip it rather than learning nothing.
EXCLUDED_DIRS = {
    "node_modules", "lib", "build", "artifacts", "cache",
    "test", "mock", "deploy", "script", "target", ".git",
}

# Languages that mean "this contest is not Solidity" when Solidity is absent.
NON_SOLIDITY_LANGUAGES = ("Move", "Cairo", "Rust", "Golang", "Go", "Vyper", "TypeScript", "Solana")

SKIPPED_FILENAME = "skipped.txt"
# Optional file of ids/slugs to skip, one per line (`#` comments allowed).
SKIP_LIST_FILENAME = "skip-ids.txt"

# Budget for the top-level `git clone`. Submodules get their own budget below,
# so this only has to cover the repo itself.
DEFAULT_CLONE_TIMEOUT = 900
# Budget for `git submodule update`, which is best-effort and must never sink
# an otherwise-good checkout. Kept separate so a heavy `lib/` tree (excluded
# from learning anyway) cannot consume the clone's budget.
DEFAULT_SUBMODULE_TIMEOUT = 300

# Written into `contracts/<id>/` only after a clone has fully completed (all
# URLs attempted and in-scope Solidity present). Its presence is what licenses
# REUSING an existing tree instead of re-cloning; without it a partial tree
# left behind by a timeout must not be trusted.
CLONE_SENTINEL = ".knowdit-clone-complete"

# Learn-phase defaults. `learn c4`'s own defaults are merge-concurrency=1 and
# context-window-utilization=0.4; a C4 bulk run is dominated by the SERIAL
# merge phase (extract concurrency does not touch it), so these raise merge
# parallelism and prompt size for throughput. Both are overridable per run.
DEFAULT_MERGE_CONCURRENCY = 4
DEFAULT_CONTEXT_WINDOW_UTILIZATION = 0.7
# Set from --clone-timeout / --submodule-timeout in main(); read by git().
CLONE_TIMEOUT = DEFAULT_CLONE_TIMEOUT
SUBMODULE_TIMEOUT = DEFAULT_SUBMODULE_TIMEOUT

# Every git invocation runs with these so a missing key, an unknown host key,
# or a credential prompt can never block on stdin — the process fails instead,
# and the contest is recorded in skipped.txt for manual handling.
GIT_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "echo",
    "GCM_INTERACTIVE": "never",
    "GIT_SSH_COMMAND": (
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=no "
        "-o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
    ),
}

# Submodules in C4 mirrors often declare SSH URLs (`git@github.com:org/repo`)
# even though the repos are public. Rewriting them to HTTPS lets the anonymous
# clone succeed; private/nonexistent ones still fail and get skipped.
GIT_SSH_REWRITES = [
    "-c", "url.https://github.com/.insteadOf=git@github.com:",
    "-c", "url.https://github.com/.insteadOf=ssh://git@github.com/",
]

GITHUB_RE = re.compile(r"https?://github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
# `code-423n4/<name>-findings` appears throughout a report even when the Scope
# section forgot to link the contest repo; stripping the suffix recovers it.
FINDINGS_REPO_RE = re.compile(r"code-423n4/([A-Za-z0-9._-]+?)-findings")
REPOSITORY_LINE_RE = re.compile(r"\*\s*Repository:\s*<([^>]+)>")
COMMIT_RE = re.compile(r"\*\s*Commit hash:\s*`([0-9a-fA-F]{7,40})`")
SCOPE_HEADING_RE = re.compile(r"^#\s+Scope\s*$", re.MULTILINE)
NEXT_HEADING_RE = re.compile(r"^#\s+\S", re.MULTILINE)
# C4 phrases this two ways, and BOTH must match:
#   "written in the Solidity programming language"        (singular, with "the")
#   "written in Rust and C++ programming languages"       (plural, no "the")
# Matching only the singular form let `2025-09-monad` (Rust/C++) through as
# Solidity, producing a project with no in-scope `.sol`.
WRITTEN_IN_RE = re.compile(
    r"written in (?:the )?([^.]*?) programming languages?", re.IGNORECASE
)
SLUG_RE = re.compile(r"^(\d{4})-(\d{2})-(.+)$")


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

@dataclass
class ReportRecord:
    """Everything we can recover from one markdown report."""

    slug: str
    id: int
    title: str
    repo_urls: list[str] = field(default_factory=list)
    commit: str | None = None
    status: str = "ready"
    error: str | None = None

    @property
    def primary_url(self) -> str | None:
        return self.repo_urls[0] if self.repo_urls else None


def scope_section(text: str) -> str:
    """The `# Scope` section body, or the whole document when absent."""
    match = SCOPE_HEADING_RE.search(text)
    if match is None:
        return text
    rest = text[match.end():]
    nxt = NEXT_HEADING_RE.search(rest)
    return rest[: nxt.start()] if nxt else rest


def parse_github_url(url: str) -> tuple[str, str] | None:
    """`(org, repo)` from a GitHub URL, ignoring any /tree|/blob path tail."""
    match = GITHUB_RE.search(url)
    if match is None:
        return None
    org, repo = match.group(1), match.group(2)
    return org, repo


def scope_repo_links(scope: str) -> tuple[list[str], list[str]]:
    """`(direct, anchored)` GitHub repo links in the Scope section.

    A link is *anchored* when its URL carries a `#fragment` (e.g.
    `https://github.com/code-423n4/2021-12-maple#audit-scope`) — that points at
    a README section, not the repo root, so it is not a usable clone target.
    Both lists hold normalized, de-duplicated `https://github.com/<org>/<repo>`.
    """
    direct: list[str] = []
    anchored: list[str] = []
    seen_direct: set[str] = set()
    seen_anchored: set[str] = set()

    def add(raw: str) -> None:
        parsed = parse_github_url(raw)
        if parsed is None:
            return
        org, repo = parsed
        if repo.endswith("-findings") or repo.endswith(".wiki"):
            return
        url = f"https://github.com/{org}/{repo}"
        if "#" in raw:
            if url not in seen_anchored:
                seen_anchored.add(url)
                anchored.append(url)
        elif url not in seen_direct:
            seen_direct.add(url)
            direct.append(url)

    for _text, url in MARKDOWN_LINK_RE.findall(scope):
        add(url)
    for url in REPOSITORY_LINE_RE.findall(scope):
        add(url)
    return direct, anchored


def collect_repo_urls(scope: str, slug: str, text: str) -> list[str]:
    """Ordered, de-duplicated clone URLs for a contest.

    Code4rena mirrors (`code-423n4/<name>`) win: they are snapshots pinned at
    the audited commit. Upstream repos from `* Repository: <...>` lines and
    explicit `/tree/<sha>` links are the fallback for the handful of old
    contests with no mirror. Finding-issue links and wiki links are dropped.

    Anchored (`#fragment`) URLs are rejected — they refer to a README section,
    not the repo — and callers skip the contest outright when that leaves no
    direct link (see [`scope_repo_links`] + [`parse_report`]).

    `text` is the full report: when the Scope section has no usable link, the
    ubiquitous `code-423n4/<name>-findings` issue URLs still reveal the mirror.
    """
    mirrors: list[str] = []
    upstreams: list[str] = []

    for _text, url in MARKDOWN_LINK_RE.findall(scope):
        if "#" in url:
            continue
        parsed = parse_github_url(url)
        if parsed is None:
            continue
        org, repo = parsed
        if repo.endswith("-findings") or repo.endswith(".wiki"):
            continue
        if org == "code-423n4":
            mirrors.append(f"https://github.com/{org}/{repo}")
        elif "/tree/" in url or "/blob/" in url:
            upstreams.append(f"https://github.com/{org}/{repo}")

    for url in REPOSITORY_LINE_RE.findall(scope):
        if "#" in url:
            continue
        parsed = parse_github_url(url)
        if parsed is None:
            continue
        org, repo = parsed
        if repo.endswith("-findings"):
            continue
        target = mirrors if org == "code-423n4" else upstreams
        target.append(f"https://github.com/{org}/{repo}")

    # Last resort: recover the mirror name from `<name>-findings` issue links
    # anywhere in the report (covers old reports whose Scope text omitted it).
    # Restrict to the slug's own month prefix — reports cross-reference other
    # contests' findings, and those mirrors must not be cloned here.
    if not mirrors and not upstreams:
        month_prefix = "-".join(slug.split("-")[:2]) + "-"
        for name in FINDINGS_REPO_RE.findall(text):
            if name.lower().startswith(month_prefix.lower()):
                mirrors.append(f"https://github.com/code-423n4/{name}")

    ordered: list[str] = []
    seen: set[str] = set()

    # Prefer a mirror whose name matches the slug or at least its month prefix,
    # so mitigation-review / follow-up repos mentioned in Scope don't win.
    month_prefix = "-".join(slug.split("-")[:2]) + "-"
    ranked = sorted(
        mirrors,
        key=lambda u: (
            0 if u.rsplit("/", 1)[-1] == slug
            else 1 if u.rsplit("/", 1)[-1].startswith(month_prefix)
            else 2
        ),
    )
    for url in ranked + upstreams:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def primary_link_text(scope: str, primary_url: str, slug: str) -> str:
    """Link text for the chosen mirror, for use as the project title."""
    for text, url in MARKDOWN_LINK_RE.findall(scope):
        parsed = parse_github_url(url)
        if parsed is None:
            continue
        if f"https://github.com/{parsed[0]}/{parsed[1]}" == primary_url:
            return clean_title(text, slug)
    return clean_title("", slug)


def clean_title(text: str, slug: str) -> str:
    """Strip C4 boilerplate from a Scope link label; fall back to the slug."""
    title = text.strip().replace("`", "")
    title = re.sub(r"\s+repository$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^C4\s+", "", title)
    title = re.sub(r"\s+code contest$", "", title)
    title = re.sub(r"\s+contest$", "", title)
    title = re.sub(r"\s+audit$", "", title)
    title = re.sub(r"\s+code$", "", title)
    title = title.strip()
    if title:
        return title
    # No usable link label (e.g. recovered from `-findings` issue URLs):
    # humanize the slug so the KG project name isn't a bare `2022-12-stealth-project`.
    humanized = SLUG_RE.sub(lambda m: m.group(3), slug).replace("-", " ")
    return humanized.title()


def is_solidity_contest(scope: str) -> bool:
    """False only when the Scope names a non-Solidity language and no Solidity.

    Mixed contests (e.g. "Solidity ... as well as ... Vyper") stay in because
    Solidity is present; the C4 loader only ever reads `.sol` anyway.
    """
    match = WRITTEN_IN_RE.search(scope)
    if match is None:
        return True
    languages = match.group(1)
    if re.search(r"solidity", languages, re.IGNORECASE):
        return True
    return not any(re.search(lang, languages, re.IGNORECASE) for lang in NON_SOLIDITY_LANGUAGES)


def parse_report(path: Path, contest_id: int) -> ReportRecord:
    text = path.read_text(encoding="utf-8", errors="replace")
    slug = path.stem
    scope = scope_section(text)
    record = ReportRecord(slug=slug, id=contest_id, title=clean_title("", slug))

    if not is_solidity_contest(scope):
        record.status = "skipped:non-solidity"
        record.error = "scope is not Solidity (C4 loader reads .sol only)"
        return record

    direct, anchored = scope_repo_links(scope)
    if anchored and not direct:
        # The only repo-shaped link is a README anchor such as
        # `https://github.com/code-423n4/2021-12-maple#audit-scope`, which
        # points at a section rather than the repo root. Refuse to guess.
        record.repo_urls = anchored
        record.status = "skipped:anchored-link"
        record.error = (
            "scope links only an anchored README URL, not a repo root: "
            f"{anchored[0]}"
        )
        return record

    record.repo_urls = collect_repo_urls(scope, slug, text)
    if not record.repo_urls:
        record.status = "skipped:no-repo"
        record.error = "no GitHub repo link found in the report"
        return record

    commit = COMMIT_RE.search(text)
    record.commit = commit.group(1) if commit else None
    record.title = primary_link_text(scope, record.repo_urls[0], slug)
    return record


# ---------------------------------------------------------------------------
# Manual skip list
# ---------------------------------------------------------------------------

def parse_skip_ids(values: list[str], dataset_dir: Path) -> set[int]:
    """Numeric ids to skip, from `--skip-ids` and `<dataset>/skip-ids.txt`.

    Accepts bare ids (`20250902`), and `slug`/`id-slug`-style entries are
    resolved later against parsed records (see [`apply_skip_list`]) because a
    slug can't be turned into an id before the reports are parsed.
    """
    ids: set[int] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            if token.isdigit():
                ids.add(int(token))
            else:
                # A slug was passed on the CLI; encode it for the later pass.
                ids.add(_slug_sentinel(token))
    path = dataset_dir / SKIP_LIST_FILENAME
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.isdigit():
                ids.add(int(line))
            else:
                ids.add(_slug_sentinel(line))
    return ids


def _slug_sentinel(slug: str) -> int:
    """Pack a slug into an int that cannot collide with a real contest id.

    Real ids are `<YYYYMM><NN>` (8 digits). Hashing a slug into the negative
    range keeps both kinds in one set without ambiguity.
    """
    return -abs(hash(slug)) - 1


def apply_skip_list(records: list[ReportRecord], skip: set[int]) -> int:
    """Mark skipped records. Returns how many were newly skipped."""
    if not skip:
        return 0

    def wanted(record: ReportRecord) -> bool:
        if record.id in skip:
            return True
        if _slug_sentinel(record.slug) in skip:
            return True
        # Allow `20230902-2025-09-monad` or a bare slug suffix to match.
        for token in skip:
            if token >= 0:
                continue
            if any(
                _slug_sentinel(part) == token
                for part in (record.slug, f"{record.id}-{record.slug}")
            ):
                return True
        return False

    count = 0
    for record in records:
        if record.status == "ready" and wanted(record):
            record.status = "skipped:manual"
            record.error = "listed in --skip-ids / skip-ids.txt"
            count += 1
    return count


# ---------------------------------------------------------------------------
# Id assignment
# ---------------------------------------------------------------------------

def assign_ids(slugs: list[str]) -> dict[str, int]:
    """`slug -> <YYYYMM><NN>` where NN is the slug's alphabetical index in its
    month (01-based). Deterministic, unique, traceable to the report month.
    """
    by_month: dict[str, list[str]] = {}
    for slug in slugs:
        match = SLUG_RE.match(slug)
        month = f"{match.group(1)}{match.group(2)}" if match else "000000"
        by_month.setdefault(month, []).append(slug)

    ids: dict[str, int] = {}
    for month, members in by_month.items():
        for index, slug in enumerate(sorted(members), start=1):
            ids[slug] = int(f"{month}{index:02d}")
    return ids


# ---------------------------------------------------------------------------
# Dataset materialization (phase 1)
# ---------------------------------------------------------------------------

def audit_payload(record: ReportRecord) -> str:
    payload = {
        "contestId": record.id,
        "title": record.title,
        "slug": record.slug,
    }
    return json.dumps(payload, indent=2) + "\n"


def check_collisions(audits_dir: Path, records: list[ReportRecord]) -> None:
    """Abort before writing anything when an existing `<id>.json` disagrees.

    The synthesized id scheme can collide with a pre-existing dataset — e.g.
    the legacy date-named `code4rena/audits/20230703.json` vs `2023-07-axelar`.
    Detecting this up front keeps a conflict from leaving a half-written
    dataset behind.
    """
    conflicts = []
    for record in records:
        if record.status != "ready":
            continue
        path = audits_dir / f"{record.id}.json"
        if path.exists() and path.read_text(encoding="utf-8", errors="replace") != audit_payload(record):
            conflicts.append((record, path))
    if not conflicts:
        return
    lines = [
        "Refusing to write: existing audits/<id>.json files disagree with the "
        "synthesized id scheme.",
        "This usually means a pre-existing dataset (e.g. the legacy date-named "
        "code4rena/audits/20230703.json) collides with it.",
        "Re-run with --force to overwrite, or point --dataset-dir at a clean directory.",
        "",
    ]
    for record, path in conflicts:
        lines.append(f"  {path.name}: exists, wanted slug {record.slug} (id {record.id})")
    raise SystemExit("\n".join(lines))


def write_audit_json(path: Path, record: ReportRecord, force: bool) -> str:
    """Write `audits/<id>.json`. Returns 'written' | 'kept'."""
    desired = audit_payload(record)
    if path.exists() and path.read_text(encoding="utf-8", errors="replace") == desired:
        return "kept"
    path.write_text(desired, encoding="utf-8")
    return "written"


def materialize(dataset_dir: Path, reports_dir: Path, records: list[ReportRecord],
                force: bool, dry_run: bool) -> dict[str, int]:
    """Write every audits/<id>.json + reports/<id>.md. Returns write counters."""
    audits_dir = dataset_dir / "audits"
    out_reports_dir = dataset_dir / "reports"
    counters = {"audit_written": 0, "audit_kept": 0, "report_copied": 0, "report_kept": 0}

    if not dry_run:
        audits_dir.mkdir(parents=True, exist_ok=True)
        out_reports_dir.mkdir(parents=True, exist_ok=True)
        if not force:
            check_collisions(audits_dir, records)

    for record in records:
        if record.status != "ready":
            continue
        audit_path = audits_dir / f"{record.id}.json"
        report_path = out_reports_dir / f"{record.id}.md"

        if dry_run:
            continue

        outcome = write_audit_json(audit_path, record, force)
        counters["audit_written" if outcome == "written" else "audit_kept"] += 1

        if report_path.exists() and not force:
            counters["report_kept"] += 1
        else:
            shutil.copyfile(reports_dir / f"{record.slug}.md", report_path)
            counters["report_copied"] += 1

    return counters


# ---------------------------------------------------------------------------
# Clone / verify (phase 2)
# ---------------------------------------------------------------------------

def has_solidity(root: Path) -> bool:
    """True when any `.sol` file exists outside the loader's excluded dirs."""
    if not root.is_dir():
        return False
    for path in root.rglob("*.sol"):
        if path.is_file() and not any(part in EXCLUDED_DIRS for part in path.parts):
            return True
    return False


def run(argv: list[str], cwd: Path | None = None, env: dict | None = None,
        timeout: int | None = None, stdin_devnull: bool = False) -> bool:
    """Run a command, streaming output. True on exit 0.

    `stdin_devnull` + `env` are used for git so an auth/host-key prompt fails
    fast instead of hanging the batch; `timeout` backstops a wedged network op.
    """
    print(f"\n  $ {' '.join(argv)}" + (f"\n    (cwd: {cwd})" if cwd else ""), flush=True)
    stdin = subprocess.DEVNULL if stdin_devnull else None
    try:
        result = subprocess.run(
            [str(a) for a in argv],
            cwd=str(cwd) if cwd else None,
            env=env,
            stdin=stdin,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        print(f"  ! {exc}", flush=True)
        return False
    except subprocess.TimeoutExpired:
        print(f"  ! timed out after {timeout}s", flush=True)
        return False
    return result.returncode == 0


def classify_git_failure(output: str) -> str:
    """Short, human reason for a git failure, for skipped.txt.

    GitHub deliberately conflates "private" and "does not exist" (and, with
    prompts disabled, reports both as an auth error), so those collapse into
    one label telling the user to check the repo by hand.
    """
    lowered = output.lower()
    if "permission denied (publickey)" in lowered or "could not read from remote repository" in lowered:
        return "auth failed (needs SSH key / credentials)"
    if (
        "authentication failed" in lowered
        or "could not read username" in lowered
        or "invalid username or token" in lowered
        or "password authentication is not supported" in lowered
    ):
        return "repo inaccessible (private/deleted, or needs auth)"
    if "repository not found" in lowered or "not found" in lowered:
        return "repo inaccessible (private/deleted, or needs auth)"
    if "timed out" in lowered:
        return "timed out"
    if "could not resolve host" in lowered or "unable to access" in lowered:
        return "network/host unreachable"
    return "git error"


def git(argv: list[str], cwd: Path | None = None, timeout: int | None = None,
        capture: bool = False) -> tuple[bool, str]:
    """Run git non-interactively with SSH->HTTPS rewrites. See [`GIT_ENV`].

    Returns `(ok, combined_output)`. With `capture=False` output streams live;
    with `capture=True` it is collected and echoed so the caller can classify
    the failure (auth vs. dead repo) for skipped.txt.
    """
    env = dict(os.environ)
    env.update(GIT_ENV)
    argv = ["git", *GIT_SSH_REWRITES, *argv]
    print(f"\n  $ {' '.join(argv)}" + (f"\n    (cwd: {cwd})" if cwd else ""), flush=True)

    if not capture:
        return run(argv, cwd=cwd, env=env,
                   timeout=timeout or CLONE_TIMEOUT, stdin_devnull=True), ""

    try:
        result = subprocess.run(
            [str(a) for a in argv],
            cwd=str(cwd) if cwd else None,
            env=env,
            stdin=subprocess.DEVNULL,
            timeout=timeout or CLONE_TIMEOUT,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        print(f"  ! {exc}", flush=True)
        return False, str(exc)
    except subprocess.TimeoutExpired:
        print(f"  ! timed out after {timeout or CLONE_TIMEOUT}s", flush=True)
        return False, "timed out"

    output = (result.stdout or "") + (result.stderr or "")
    if output.strip():
        print(output.rstrip(), flush=True)
    return result.returncode == 0, output


def _update_submodules(dest: Path) -> None:
    """Best-effort `git submodule update`, on its OWN timeout.

    Deliberately separate from the top-level clone. `--recurse-submodules` on
    the clone makes one timeout cover both phases, and a heavy `lib/` tree
    easily eats the whole budget: `2023-07-basin` fetched its sources fine,
    then spent >600s on 5 `lib/` submodules and was reported as a failure even
    though the checkout was complete.

    Submodules genuinely matter for one shape — a wrapper mirror whose scope
    lives at the top level of a submodule (e.g. `2022-06-canto-v2`) — so we
    still fetch them; they are just no longer allowed to sink the clone.
    Everything under `lib/` is excluded from learning regardless.
    """
    if not (dest / ".gitmodules").is_file():
        return
    ok, _ = git(
        ["submodule", "update", "--init", "--recursive", "--depth", "1"],
        cwd=dest, capture=True, timeout=SUBMODULE_TIMEOUT,
    )
    if not ok:
        print(
            "    . submodule fetch incomplete (best-effort; in-scope source "
            "is unaffected unless this contest keeps its scope in a submodule)",
            flush=True,
        )


def _clone_shallow(url: str, dest: Path) -> tuple[bool, str]:
    """Shallow-clone the top-level repo, then fetch submodules best-effort.

    Two phases on two budgets, so a slow submodule cannot fail a good checkout.
    Returns `(ok, reason)`; `reason` is a short human string on failure.
    """
    ok, output = _clone_phase(url, dest)
    if not ok:
        return False, classify_git_failure(output)
    _update_submodules(dest)
    return True, ""


def _clone_phase(url: str, dest: Path) -> tuple[bool, str]:
    """Just the `git clone --depth 1` — no submodules. Returns `(ok, output)`."""
    return git(["clone", "-q", "--depth", "1", url, str(dest)], capture=True)


def clone_repo(url: str, dest: Path, commit: str | None, is_mirror: bool) -> tuple[bool, str]:
    """Shallow-clone `url` into `dest`. Returns `(ok, reason)`.

    Code4rena mirrors are already pinned at the audited commit, so a plain
    `--depth 1` clone is correct. Upstream repos are fetched at `commit` when
    one is known, falling back to a plain clone when the sha isn't fetchable.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)

    if commit and not is_mirror:
        dest.mkdir(parents=True)
        ok = (
            git(["init", "-q", str(dest)])[0]
            and git(["remote", "add", "origin", url], cwd=dest)[0]
            and git(["fetch", "-q", "--depth", "1", "origin", commit], cwd=dest)[0]
            and git(["checkout", "-q", "FETCH_HEAD"], cwd=dest)[0]
        )
        if ok:
            _update_submodules(dest)
            return True, ""
        print(f"  ! commit {commit[:12]} not fetchable; retrying default branch", flush=True)
        shutil.rmtree(dest, ignore_errors=True)

    return _clone_shallow(url, dest)


def clone_contest(record: ReportRecord, contracts_dir: Path) -> tuple[bool, str | None]:
    """Clone repo(s) for one contest under `contracts/<id>/`.

    Returns `(ok, error)`. Each repo lands in its own subdirectory so
    multi-repo contests don't collide; the loader walks the tree recursively.

    URLs are tried in order (mirror first) until one contributes Solidity.
    Some mirrors are README-only wrappers (e.g. `2022-06-badger`) or hold their
    scope as a submodule that no longer resolves; in those cases we fall
    through to the upstream repo named in the report. Every additional URL is
    still cloned when it resolves, since multi-repo contests need all of them.
    """
    dest_root = contracts_dir / str(record.id)
    sentinel = dest_root / CLONE_SENTINEL
    if sentinel.is_file() and has_solidity(dest_root):
        # A PREVIOUS run completed this clone. Reuse it — re-downloading is the
        # slow part of a resume. The sentinel, not the mere presence of `.sol`,
        # is what licenses reuse: a tree left by a timeout has sources but may
        # be missing URLs, so it is wiped and redone below.
        return True, None
    if dest_root.exists():
        # Partial/interrupted tree (or one predating the sentinel): start clean
        # so a stale checkout cannot be silently trusted.
        shutil.rmtree(dest_root, ignore_errors=True)

    mirrors = [u for u in record.repo_urls if u.startswith("https://github.com/code-423n4/")]
    upstreams = [u for u in record.repo_urls if u not in mirrors]
    cloned_any = False
    reasons: list[str] = []

    def try_clone(urls: list[str]) -> bool:
        """Clone every URL; True when any succeeded."""
        nonlocal cloned_any
        any_ok = False
        for url in urls:
            parsed = parse_github_url(url)
            if parsed is None:
                continue
            org, repo = parsed
            ok, reason = clone_repo(url, dest_root / repo, record.commit, org == "code-423n4")
            if ok:
                cloned_any = True
                any_ok = True
            else:
                reasons.append(f"{url}: {reason}")
                print(f"  ! clone failed ({reason}): {url}", flush=True)
        return any_ok

    # The C4 mirror aggregates the whole scope, so one that yields Solidity is
    # sufficient. Upstreams are provenance — only needed when no mirror worked
    # (README-only wrapper or a dead submodule), and then we want ALL of them
    # because the scope can span several repos.
    complete = (try_clone(mirrors) and has_solidity(dest_root)) or (
        bool(upstreams) and try_clone(upstreams) and has_solidity(dest_root)
    )
    if complete:
        # Every URL was attempted and in-scope source is present: safe to mark
        # complete, so a later resume reuses this tree instead of re-cloning.
        try:
            sentinel.write_text(f"{record.id}\n{record.slug}\n", encoding="utf-8")
        except OSError as exc:
            print(f"  ! could not write clone sentinel: {exc}", flush=True)
        return True, None

    if not cloned_any:
        detail = "; ".join(reasons) if reasons else "no cloneable URL"
        return False, f"all clones failed ({detail})"
    shutil.rmtree(dest_root, ignore_errors=True)
    return False, "no .sol files after excludes"


# ---------------------------------------------------------------------------
# Learn (phase 2)
# ---------------------------------------------------------------------------

def learn_batch(ids: list[int], args: argparse.Namespace) -> bool:
    # `--database-url` is a flag on the `learn` parent command, so it must come
    # BEFORE the `c4` subcommand (``learn --database-url <URL> c4 ...``).
    argv = [
        args.knowdit_bin,
        "learn",
        "--database-url", args.database_url,
        "c4",
        "--c4-dir", str(args.dataset_dir),
        "--c4-ids", ",".join(str(i) for i in ids),
        "--concurrency", str(args.concurrency),
        "--merge-concurrency", str(args.merge_concurrency),
        "--context-window-utilization", str(args.context_window_utilization),
    ]
    if args.link_per_batch:
        argv.append("--link")
    return run(argv)


def learn_link_all(args: argparse.Namespace) -> bool:
    return run([
        args.knowdit_bin,
        "learn",
        "--database-url", args.database_url,
        "link",
    ])


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def save_manifest(dataset_dir: Path, records: list[ReportRecord]) -> None:
    path = dataset_dir / "manifest.json"
    payload = [asdict(r) for r in sorted(records, key=lambda r: r.id)]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def apply_manifest_resume(dataset_dir: Path, records: list[ReportRecord]) -> int:
    """Carry permanently-settled outcomes forward from a previous manifest.

    Only statuses that will NOT change on a retry are carried:
      * `learned`            — done
      * `skipped:non-solidity` / `skipped:no-repo` / `skipped:anchored-link`
                             — decided from the report text, deterministic
      * `skipped:manual`     — the user asked for it

    Deliberately NOT carried: `skipped:clone-failed`, `skipped:empty`,
    `failed`. Those can be transient (a network blip, a learn error), so the
    contest is retried on the next run. knowdit also skips already-completed
    `platform_id`s, but without this the script would re-clone every repo.
    """
    path = dataset_dir / "manifest.json"
    if not path.is_file():
        return 0
    try:
        previous = {entry["id"]: entry for entry in json.loads(path.read_text())}
    except (json.JSONDecodeError, KeyError, TypeError):
        return 0

    carried = {
        "learned",
        "skipped:non-solidity",
        "skipped:no-repo",
        "skipped:anchored-link",
        "skipped:manual",
    }
    resumed = 0
    for record in records:
        prior = previous.get(record.id)
        if prior is None or prior.get("slug") != record.slug:
            continue
        if prior.get("status") in carried:
            record.status = prior["status"]
            record.error = prior.get("error")
            resumed += 1
    return resumed


def cleanup_failed_contracts(contracts_dir: Path, records: list[ReportRecord],
                             active_ids: set[int]) -> int:
    """Delete `contracts/<id>/` for contests that settled as failures.

    Without this, every `skipped:clone-failed` / `skipped:empty` tree is left
    on disk forever (only a SUCCESSFUL learn triggers cleanup in the batch
    loop), so a long run accumulates tens of MB of dead checkouts. They are
    re-cloned on the next run anyway — the manifest does not carry
    `clone-failed` forward — so there is nothing worth keeping.

    `active_ids` guards the in-flight batch: never delete a tree the current
    batch may still be reading.

    Returns the number of directories removed.
    """
    if not contracts_dir.is_dir():
        return 0
    failed = {
        r.id for r in records
        if r.status in ("skipped:clone-failed", "skipped:empty", "failed")
    }
    removed = 0
    for path in sorted(contracts_dir.iterdir()):
        if not path.is_dir() or not path.name.isdigit():
            continue
        if int(path.name) not in failed or int(path.name) in active_ids:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed


def write_skipped_file(dataset_dir: Path, records: list[ReportRecord]) -> None:
    """Write `skipped.txt`: every contest that did NOT get learned, with the
    reason and the clone URLs, so it can be handled by hand.

    Covers auth/host-key failures (e.g. an SSH submodule with no key), dead
    repos, wrapper mirrors with no Solidity, non-Solidity contests, reports
    with no recoverable repo link, and anything in the manual skip list.
    """
    skipped = [r for r in records if r.status not in ("learned", "ready")]
    path = dataset_dir / SKIPPED_FILENAME
    if not skipped:
        if path.exists():
            path.unlink()
        return

    lines = [
        "# Contests NOT learned — handle these manually.",
        "# Each entry: id  slug  status  reason  (clone URLs)",
        "",
    ]
    for record in sorted(skipped, key=lambda r: r.id):
        lines.append(f"{record.id}\t{record.slug}\t{record.status}\t{record.error or ''}")
        for url in record.repo_urls:
            lines.append(f"\t\t{url}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(records: list[ReportRecord]) -> None:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    print("\nReport summary:")
    for status in sorted(counts):
        print(f"  {status:24} {counts[status]}")
    print(f"  {'TOTAL':24} {len(records)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_c4_dataset.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reports-dir", type=Path, default=Path(DEFAULT_REPORTS_DIR),
                        help=f"Scraped markdown reports (default: {DEFAULT_REPORTS_DIR}).")
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR),
                        help="Target dataset dir with audits/ contracts/ reports/ "
                             f"(default: {DEFAULT_DATASET_DIR}).")
    parser.add_argument("--database-url", default=None,
                        help="Historical KG URL for `knowdit learn` (sqlite://, mysql://, "
                             "postgres://). Required unless --prepare-only/--skip-learn.")
    parser.add_argument("--knowdit-bin", default="./target/release/knowdit",
                        help="Path to the knowdit binary (default: ./target/release/knowdit).")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Contests cloned+learned per batch (default: 10).")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Passed through to `learn c4 --concurrency`: how many "
                             "projects categorize+extract at once (default: 1).")
    parser.add_argument("--merge-concurrency", type=int, default=DEFAULT_MERGE_CONCURRENCY,
                        help="Passed through to `learn c4 --merge-concurrency`: merge "
                             "agents running in parallel. Merge is the serial "
                             "bottleneck (extract concurrency cannot touch it), so "
                             f"raising this is the main throughput lever (default: "
                             f"{DEFAULT_MERGE_CONCURRENCY}). Decisions aggregate by "
                             "name, so this changes speed, not results.")
    parser.add_argument("--context-window-utilization", type=float,
                        default=DEFAULT_CONTEXT_WINDOW_UTILIZATION,
                        help="Passed through to `learn c4 --context-window-utilization`: "
                             "fraction (0,1] of the model window a single "
                             "categorize/extract/merge prompt may fill. Higher = bigger "
                             "prompts, fewer chunks (default: "
                             f"{DEFAULT_CONTEXT_WINDOW_UTILIZATION}).")
    parser.add_argument("--clone-timeout", type=int, default=DEFAULT_CLONE_TIMEOUT,
                        help="Timeout in seconds for the top-level `git clone` "
                             f"(default: {DEFAULT_CLONE_TIMEOUT}). A hung fetch is aborted "
                             "and the contest is skipped, not retried forever.")
    parser.add_argument("--submodule-timeout", type=int, default=DEFAULT_SUBMODULE_TIMEOUT,
                        help="Timeout in seconds for `git submodule update`, which runs "
                             "BEST-EFFORT after a successful clone "
                             f"(default: {DEFAULT_SUBMODULE_TIMEOUT}). Submodules rarely "
                             "matter (scope under `lib/` is excluded anyway), so a slow "
                             "one no longer fails an otherwise-good checkout.")
    parser.add_argument("--skip-ids", action="append", default=[], metavar="IDS",
                        help="Contest ids and/or slugs to skip entirely (never cloned or "
                             "learned, recorded as `skipped:manual`). Repeatable and "
                             "comma-separated, e.g. `--skip-ids 20250902` or "
                             "`--skip-ids 2025-09-monad,2024-05-canto`. A "
                             f"`<dataset>/{SKIP_LIST_FILENAME}` file is read too.")
    parser.add_argument("--link-per-batch", action="store_true",
                        help="Pass --link to every `learn c4` invocation.")
    parser.add_argument("--link-at-end", action="store_true",
                        help="Run one `learn link` after all batches finish.")
    parser.add_argument("--keep-contracts", action="store_true",
                        help="Do not delete contracts/<id>/ after a successful learn.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Phase 1 only: write audits/ + reports/, no clones.")
    parser.add_argument("--skip-clone", action="store_true",
                        help="Skip cloning (assumes contracts/ is already populated).")
    parser.add_argument("--skip-learn", action="store_true",
                        help="Prepare + clone, then stop (implies --keep-contracts).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing audits/reports files and ignore resume state.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore manifest.json; re-clone and re-learn every ready contest.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and print the plan without writing anything.")
    return parser


def validate_numeric_args(args: argparse.Namespace) -> None:
    """Reject out-of-range knobs up front — always, including under --dry-run.

    knowdit clamps some of these, but a silently-clamped value would make the
    run behave differently from what was asked for, so fail loudly instead.
    """
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.merge_concurrency < 1:
        raise SystemExit("--merge-concurrency must be at least 1")
    if not 0.0 < args.context_window_utilization <= 1.0:
        raise SystemExit("--context-window-utilization must be in (0, 1]")
    if args.clone_timeout < 1:
        raise SystemExit("--clone-timeout must be at least 1 second")
    if args.submodule_timeout < 1:
        raise SystemExit("--submodule-timeout must be at least 1 second")


def check_preconditions(args: argparse.Namespace, records: list[ReportRecord]) -> None:
    if not args.reports_dir.is_dir():
        raise SystemExit(f"reports dir not found: {args.reports_dir}")

    if args.dry_run or args.prepare_only:
        return

    if not args.skip_clone and shutil.which("git") is None:
        raise SystemExit("git not found on $PATH (needed to clone repos)")

    if not args.skip_learn:
        if not args.database_url:
            raise SystemExit("--database-url is required to learn (or pass --skip-learn)")
        if not Path(args.knowdit_bin).exists():
            raise SystemExit(f"knowdit binary not found: {args.knowdit_bin}")

    ready = [r for r in records if r.status == "ready"]
    if not ready:
        raise SystemExit("no ready contests to process")


def main(argv: list[str] | None = None) -> int:
    global CLONE_TIMEOUT, SUBMODULE_TIMEOUT
    args = build_parser().parse_args(argv)
    args.dataset_dir = args.dataset_dir.resolve()
    args.reports_dir = args.reports_dir.resolve()
    CLONE_TIMEOUT = args.clone_timeout
    SUBMODULE_TIMEOUT = args.submodule_timeout

    report_paths = sorted(args.reports_dir.glob("*.md"))
    if not report_paths:
        raise SystemExit(f"no .md reports under {args.reports_dir}")

    ids = assign_ids([p.stem for p in report_paths])
    records = [parse_report(path, ids[path.stem]) for path in report_paths]

    if not args.no_resume and not args.force:
        resumed = apply_manifest_resume(args.dataset_dir, records)
        if resumed:
            print(f"Resuming: {resumed} contest(s) already settled in manifest.json "
                  "(use --no-resume or --force to redo them).")

    skip = parse_skip_ids(args.skip_ids, args.dataset_dir)
    if skip:
        n = apply_skip_list(records, skip)
        if n:
            print(f"Skipping {n} contest(s) from --skip-ids / {SKIP_LIST_FILENAME}.")

    validate_numeric_args(args)

    print_summary(records)
    print(f"\nDataset dir: {args.dataset_dir}")
    print(f"Learn flags: --concurrency {args.concurrency} "
          f"--merge-concurrency {args.merge_concurrency} "
          f"--context-window-utilization {args.context_window_utilization}")
    if args.dry_run:
        print("\n[dry-run] no files written.")
        return 0

    check_preconditions(args, records)

    counters = materialize(args.dataset_dir, args.reports_dir, records,
                           args.force, args.dry_run)
    print(
        f"\nPhase 1: wrote {counters['audit_written']} audits "
        f"({counters['audit_kept']} kept), copied {counters['report_copied']} reports "
        f"({counters['report_kept']} kept)."
    )
    save_manifest(args.dataset_dir, records)

    if args.prepare_only:
        print("--prepare-only: stopping before clone.")
        return 0

    contracts_dir = args.dataset_dir / "contracts"
    pending = [r for r in records if r.status == "ready"]
    batches = [pending[i:i + args.batch_size] for i in range(0, len(pending), args.batch_size)]
    print(f"\nPhase 2: {len(batches)} batch(es) of up to {args.batch_size}.")

    skipped_notes: list[str] = []
    for index, batch in enumerate(batches, start=1):
        print(f"\n=== Batch {index}/{len(batches)}: {len(batch)} contest(s) ===", flush=True)

        if not args.skip_clone:
            cloned: list[ReportRecord] = []
            for record in batch:
                print(f"\n[clone] {record.id} ({record.slug}) <- {record.primary_url}")
                ok, error = clone_contest(record, contracts_dir)
                if ok:
                    cloned.append(record)
                else:
                    # A clone that needs credentials / points at a dead repo is
                    # not a script failure — skip the contest and record it for
                    # manual handling instead of aborting the run.
                    empty = bool(error) and "no .sol files" in error
                    record.status = "skipped:empty" if empty else "skipped:clone-failed"
                    record.error = error
                    skipped_notes.append(f"{record.id} {record.slug}: {error}")
                    print(f"  ! skipped: {error}", flush=True)
            batch = cloned

        if not batch:
            save_manifest(args.dataset_dir, records)
            write_skipped_file(args.dataset_dir, records)
            cleanup_failed_contracts(contracts_dir, records, set())
            continue

        if args.skip_learn:
            print("\n--skip-learn: cloned one batch and stopping; contracts/ left in place.")
            save_manifest(args.dataset_dir, records)
            break

        ids_to_learn = [r.id for r in batch]
        if learn_batch(ids_to_learn, args):
            for record in batch:
                record.status = "learned"
                record.error = None
            if not args.keep_contracts:
                for record in batch:
                    shutil.rmtree(contracts_dir / str(record.id), ignore_errors=True)
            print(f"\n[batch {index}] learned {len(ids_to_learn)} contest(s).", flush=True)
        else:
            for record in batch:
                record.status = "failed"
                record.error = "learn exited non-zero"
            skipped_notes.append(f"batch {index} (ids {ids_to_learn}): learn failed")
            print(f"\n[batch {index}] learn FAILED; keeping contracts/ for retry.", flush=True)

        save_manifest(args.dataset_dir, records)
        write_skipped_file(args.dataset_dir, records)
        # Reclaim trees from earlier settled failures. The current batch's ids
        # are excluded: on a learn failure its contracts are kept on purpose.
        reclaimed = cleanup_failed_contracts(
            contracts_dir, records, set(ids_to_learn) if not args.keep_contracts else set()
        )
        if reclaimed:
            print(f"[batch {index}] reclaimed {reclaimed} dead contract dir(s).", flush=True)

    if args.link_at_end and not args.skip_learn:
        print("\n=== Final link pass ===", flush=True)
        if not learn_link_all(args):
            skipped_notes.append("final learn link failed")

    save_manifest(args.dataset_dir, records)
    write_skipped_file(args.dataset_dir, records)

    print("\nDone.")
    print_summary(records)
    skipped_path = args.dataset_dir / SKIPPED_FILENAME
    if skipped_path.exists():
        print(f"\nContests needing manual attention: {skipped_path}")
    if skipped_notes:
        print("\nNot learned:")
        for note in skipped_notes:
            print(f"  - {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
