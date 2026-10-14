#!/usr/bin/env python3
"""Build a learnable dataset from Sherlock contest reports, then learn it in
space-bounded batches — via the `learn c4` pipeline.

Why the C4 layout
-----------------
`learn c4` is the only ingestion path that wires BOTH source code -> semantics
AND report -> findings. `learn sherlock` expects a scraper's
`metadata/ source/ reports/` tree which we do not have, and `feed` attaches no
source at all (its semantics are prose-only, with `_narrative` placeholder
function refs). So the Sherlock reports are laid out in the C4 shape the loader
already understands:

    <dataset>/audits/<id>.json    contest metadata (contestId + title)
    <dataset>/contracts/<id>/     Solidity sources (walked for **/*.sol)
    <dataset>/reports/<id>.md     the markdown report

The id is the report's own `- **Contest ID:** N` value (1..1279, all distinct),
so nothing has to be synthesized. Because the id becomes the KG platform id
`c4-<id>`, these projects are intentionally indistinguishable from genuine C4
ones in the database.

Report shapes (all 228 handled)
-------------------------------
Full block (188 reports):
    ## Scope
    - **Repo:** `Merit-Circle/dutch-nft`
    - **Branch:** `main`
    - **Commit:** `822c3c546227ded78e791cf5b0d2addc59dd0214`
    - **GitHub:** https://github.com/Merit-Circle/dutch-nft/tree/822c3c546...
    - **Files in scope:**
      - `src/MeritDutchAuction.sol` (137 nSLOC)

Bare form (40 older reports):
    ## Scope
    - https://github.com/sherlock-audit/2022-08-sentiment

Clone strategy
--------------
1. `https://github.com/sherlock-audit/<filename-stem>` — the packaged mirror.
   It nests every in-scope repo at the audited commit, so it is the most
   complete single source when it exists.
2. If that fails or yields no Solidity: the repo(s) named in the report (35
   contests are multi-repo: 2-5 repos), shallow at that block's `**Commit:**`.
3. If that also fails: recorded in `skipped_sherlock.txt`.

Only Solidity contests are learned: a report whose `Files in scope:` list is
non-empty and contains no `.sol` is skipped. Mixed scopes (`.sol` + `.move` /
`.go` / `.vy`) are kept — the loader reads `.sol` only.

Usage:
    scripts/build_sherlock_dataset.py --dry-run
    scripts/build_sherlock_dataset.py --prepare-only
    scripts/build_sherlock_dataset.py --batch-size 10 \\
        --database-url sqlite:///contest_kg.db?mode=rwc
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

DEFAULT_REPORTS_DIR = "/home/zeit/Desktop/reports/sherlock/solidity-reports-md"
DEFAULT_DATASET_DIR = "sherlock"

# Directory names the C4 loader drops from a project's scope (mirrors
# `DEFAULT_VENDORED_EXCLUDES` in crates/knowdit-project/src/data.rs).
EXCLUDED_DIRS = {
    "node_modules", "lib", "build", "artifacts", "cache",
    "test", "mock", "deploy", "script", "target", ".git",
}

SKIPPED_FILENAME = "skipped_sherlock.txt"
SKIP_LIST_FILENAME = "skip-ids.txt"

DEFAULT_CLONE_TIMEOUT = 900
DEFAULT_SUBMODULE_TIMEOUT = 300

# Written into `contracts/<id>/` only after a clone fully completed. Its
# presence is what licenses REUSING a tree instead of re-cloning; a partial
# tree left by a timeout has sources but must not be trusted.
CLONE_SENTINEL = ".knowdit-clone-complete"

DEFAULT_MERGE_CONCURRENCY = 4
DEFAULT_CONTEXT_WINDOW_UTILIZATION = 0.7
CLONE_TIMEOUT = DEFAULT_CLONE_TIMEOUT
SUBMODULE_TIMEOUT = DEFAULT_SUBMODULE_TIMEOUT

GIT_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "echo",
    "GCM_INTERACTIVE": "never",
    "GIT_SSH_COMMAND": (
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=no "
        "-o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
    ),
}

GIT_SSH_REWRITES = [
    "-c", "url.https://github.com/.insteadOf=git@github.com:",
    "-c", "url.https://github.com/.insteadOf=ssh://git@github.com:",
]

# `## Scope` .. next `##` heading.
SCOPE_HEADING_RE = re.compile(r"^##\s+Scope\s*$", re.MULTILINE)
NEXT_H2_RE = re.compile(r"^##\s+\S", re.MULTILINE)
TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
CONTEST_ID_RE = re.compile(r"\*\*Contest ID:\*\*\s*(\d+)", re.IGNORECASE)
REPO_LINE_RE = re.compile(r"\*\*Repo:\*\*\s*`([^`]+)`")
COMMIT_LINE_RE = re.compile(r"\*\*Commit:\*\*\s*`([0-9a-fA-F]{7,40})`")
GITHUB_LINE_RE = re.compile(r"\*\*GitHub:\*\*\s*(\S+)")
SCOPE_FILE_RE = re.compile(r"^\s*[-*]\s*`([^`]+)`")
BARE_GITHUB_RE = re.compile(r"^\s*[-*]\s*(https?://github\.com/\S+)")
GITHUB_URL_RE = re.compile(r"https?://github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+)")


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

@dataclass
class RepoBlock:
    """One `**Repo:**` ... `**Files in scope:**` group in the Scope section."""

    url: str
    commit: str | None = None
    files: list[str] = field(default_factory=list)


@dataclass
class ReportRecord:
    slug: str
    id: int
    title: str
    repos: list[str] = field(default_factory=list)
    commits: dict[str, str | None] = field(default_factory=dict)
    scope_files: list[str] = field(default_factory=list)
    status: str = "ready"
    error: str | None = None

    @property
    def primary_url(self) -> str | None:
        return self.repos[0] if self.repos else None

    def commit_for(self, url: str) -> str | None:
        return self.commits.get(url)


def scope_section(text: str) -> str:
    match = SCOPE_HEADING_RE.search(text)
    if match is None:
        return ""
    rest = text[match.end():]
    nxt = NEXT_H2_RE.search(rest)
    return rest[: nxt.start()] if nxt else rest


def github_repo(url: str) -> tuple[str, str] | None:
    """`(org, repo)` from a GitHub URL, ignoring `/tree|/blob` tails."""
    match = GITHUB_URL_RE.search(url)
    if match is None:
        return None
    return match.group(1), match.group(2)


def normalize_repo_url(url: str) -> str | None:
    """`https://github.com/<org>/<repo>` — the cloneable root, no branch path."""
    parsed = github_repo(url)
    if parsed is None:
        return None
    org, repo = parsed
    if repo.endswith("-judging"):
        # `sherlock-audit/<contest>-judging` is the judge's repo, not scope.
        return None
    return f"https://github.com/{org}/{repo}"


def parse_scope(scope: str) -> tuple[list[RepoBlock], list[str]]:
    """`(repo_blocks, bare_urls)` from a Scope section.

    Walks the lines so each `**Commit:**` binds to the `**Repo:**` above it and
    each indented `` - `path` `` file binds to the GitHub line above it.
    """
    blocks: list[RepoBlock] = []
    bare: list[str] = []
    current: RepoBlock | None = None
    pending_commit: str | None = None

    for line in scope.splitlines():
        repo_match = REPO_LINE_RE.search(line)
        if repo_match:
            current = RepoBlock(url="", files=[])
            blocks.append(current)
            pending_commit = None
            continue

        commit_match = COMMIT_LINE_RE.search(line)
        if commit_match:
            pending_commit = commit_match.group(1)
            if current is not None:
                current.commit = pending_commit
            continue

        github_match = GITHUB_LINE_RE.search(line)
        if github_match:
            url = normalize_repo_url(github_match.group(1))
            if url is None:
                continue
            if current is None:
                current = RepoBlock(url=url, files=[])
                blocks.append(current)
            else:
                current.url = url
            if current.commit is None and pending_commit:
                current.commit = pending_commit
            continue

        bare_match = BARE_GITHUB_RE.match(line)
        if bare_match:
            url = normalize_repo_url(bare_match.group(1))
            if url is not None:
                bare.append(url)
            continue

        file_match = SCOPE_FILE_RE.match(line)
        if file_match and current is not None:
            current.files.append(file_match.group(1))

    return blocks, bare


def is_solidity_scope(files: list[str]) -> bool:
    """False only when files are listed and NONE is Solidity.

    Mixed scopes (`.sol` beside `.move`/`.go`/`.vy`) stay in: the C4 loader
    reads `**/*.sol` only, so the non-Solidity entries are simply ignored.
    """
    if not files:
        return True
    return any(f.lower().endswith(".sol") for f in files)


def parse_report(path: Path) -> ReportRecord:
    text = path.read_text(encoding="utf-8", errors="replace")
    slug = path.stem
    scope = scope_section(text)

    id_match = CONTEST_ID_RE.search(text)
    if id_match is None:
        return ReportRecord(slug=slug, id=0, title=slug,
                            status="skipped:no-id",
                            error="no `- **Contest ID:** N` line in the report")
    contest_id = int(id_match.group(1))

    title_match = TITLE_RE.search(text)
    title = title_match.group(1).strip() if title_match else slug

    blocks, bare = parse_scope(scope)

    repos: list[str] = []
    commits: dict[str, str | None] = {}
    scope_files: list[str] = []
    for block in blocks:
        if not block.url or block.url in commits:
            continue
        repos.append(block.url)
        commits[block.url] = block.commit
        scope_files.extend(block.files)
    for url in bare:
        if url not in commits:
            repos.append(url)
            commits[url] = None

    # Fallback: the packaged mirror named after the report file.
    mirror = f"https://github.com/sherlock-audit/{slug}"
    record = ReportRecord(slug=slug, id=contest_id, title=title,
                          repos=repos, commits=commits, scope_files=scope_files)

    if not repos:
        # Bare/missing scope: fall back to the packaged mirror. The commit is
        # unknown, so the default branch is used; the post-clone `.sol` check
        # is what ultimately decides.
        record.repos = [mirror]
        record.commits = {mirror: None}
        record.error = "no repo in Scope; using sherlock-audit mirror"
        if not is_solidity_scope(scope_files):
            record.status = "skipped:non-solidity"
            record.error = "scope lists files, none Solidity"
        return record

    if not is_solidity_scope(scope_files):
        record.status = "skipped:non-solidity"
        record.error = "scope lists files, none Solidity"
        return record

    return record


# ---------------------------------------------------------------------------
# Manual skip list
# ---------------------------------------------------------------------------

def _slug_sentinel(slug: str) -> int:
    """Pack a slug into a negative int so both ids and slugs fit one set."""
    return -abs(hash(slug)) - 1


def parse_skip_ids(values: list[str], dataset_dir: Path) -> set[int]:
    """Ids/slugs to skip, from `--skip-ids` and `<dataset>/skip-ids.txt`."""
    tokens: set[int] = set()
    raw: list[str] = []
    for value in values:
        raw.extend(t.strip() for t in value.split(",") if t.strip())
    path = dataset_dir / SKIP_LIST_FILENAME
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                raw.append(line)
    for token in raw:
        tokens.add(int(token) if token.isdigit() else _slug_sentinel(token))
    return tokens


def apply_skip_list(records: list[ReportRecord], skip: set[int]) -> int:
    if not skip:
        return 0
    count = 0
    for record in records:
        if record.status != "ready":
            continue
        if record.id in skip or _slug_sentinel(record.slug) in skip:
            record.status = "skipped:manual"
            record.error = "listed in --skip-ids / skip-ids.txt"
            count += 1
    return count


# ---------------------------------------------------------------------------
# Dataset materialization (phase 1)
# ---------------------------------------------------------------------------

def audit_payload(record: ReportRecord) -> str:
    return json.dumps(
        {"contestId": record.id, "title": record.title, "slug": record.slug},
        indent=2,
    ) + "\n"


def check_collisions(audits_dir: Path, records: list[ReportRecord]) -> None:
    """Abort before writing when an existing `<id>.json` disagrees or two
    reports claim the same Contest ID."""
    seen: dict[int, str] = {}
    for record in records:
        if record.status != "ready":
            continue
        if record.id in seen:
            raise SystemExit(
                f"duplicate Contest ID {record.id}: {seen[record.id]} and "
                f"{record.slug}. Resolve before running."
            )
        seen[record.id] = record.slug

    conflicts = [
        (r, audits_dir / f"{r.id}.json")
        for r in records
        if r.status == "ready"
        and (audits_dir / f"{r.id}.json").is_file()
        and (audits_dir / f"{r.id}.json").read_text(encoding="utf-8", errors="replace")
        != audit_payload(r)
    ]
    if not conflicts:
        return
    lines = [
        "Refusing to write: existing audits/<id>.json files disagree with the "
        "parsed Contest IDs.",
        "Re-run with --force to overwrite, or point --dataset-dir at a clean directory.",
        "",
    ]
    for record, path in conflicts:
        lines.append(f"  {path.name}: exists, wanted slug {record.slug}")
    raise SystemExit("\n".join(lines))


def materialize(dataset_dir: Path, reports_dir: Path, records: list[ReportRecord],
                force: bool, dry_run: bool) -> dict[str, int]:
    audits_dir = dataset_dir / "audits"
    out_reports = dataset_dir / "reports"
    counters = {"audit_written": 0, "audit_kept": 0, "report_copied": 0, "report_kept": 0}

    if not dry_run:
        audits_dir.mkdir(parents=True, exist_ok=True)
        out_reports.mkdir(parents=True, exist_ok=True)
        if not force:
            check_collisions(audits_dir, records)

    for record in records:
        if record.status != "ready" or dry_run:
            continue
        audit_path = audits_dir / f"{record.id}.json"
        report_path = out_reports / f"{record.id}.md"

        desired = audit_payload(record)
        if audit_path.is_file() and audit_path.read_text(
            encoding="utf-8", errors="replace"
        ) == desired:
            counters["audit_kept"] += 1
        else:
            audit_path.write_text(desired, encoding="utf-8")
            counters["audit_written"] += 1

        if report_path.is_file() and not force:
            counters["report_kept"] += 1
        else:
            shutil.copyfile(reports_dir / f"{record.slug}.md", report_path)
            counters["report_copied"] += 1

    return counters


# ---------------------------------------------------------------------------
# Clone / verify (phase 2)
# ---------------------------------------------------------------------------

def has_solidity(root: Path) -> bool:
    """True when any `.sol` exists outside the loader's excluded dirs."""
    if not root.is_dir():
        return False
    for path in root.rglob("*.sol"):
        if path.is_file() and not any(part in EXCLUDED_DIRS for part in path.parts):
            return True
    return False


def run(argv: list[str], cwd: Path | None = None, env: dict | None = None,
        timeout: int | None = None, stdin_devnull: bool = False,
        announce: bool = True) -> bool:
    if announce:
        print(f"\n  $ {' '.join(argv)}" + (f"\n    (cwd: {cwd})" if cwd else ""),
              flush=True)
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
    lowered = output.lower()
    if "permission denied (publickey)" in lowered or "could not read from remote repository" in lowered:
        return "auth failed (needs SSH key / credentials)"
    if (
        "authentication failed" in lowered
        or "could not read username" in lowered
        or "invalid username or token" in lowered
        or "password authentication is not supported" in lowered
        or "repository not found" in lowered
    ):
        return "repo inaccessible (private/deleted, or needs auth)"
    if "couldn't find remote ref" in lowered or "not our ref" in lowered:
        return "commit not fetchable"
    if "timed out" in lowered:
        return "timed out"
    if "could not resolve host" in lowered or "unable to access" in lowered:
        return "network/host unreachable"
    return "git error"


def git(argv: list[str], cwd: Path | None = None, timeout: int | None = None,
        capture: bool = False) -> tuple[bool, str]:
    env = dict(os.environ)
    env.update(GIT_ENV)
    argv = ["git", *GIT_SSH_REWRITES, *argv]
    print(f"\n  $ {' '.join(argv)}" + (f"\n    (cwd: {cwd})" if cwd else ""), flush=True)

    if not capture:
        # `run` would print this line again; the header is already emitted above.
        return run(argv, cwd=cwd, env=env,
                   timeout=timeout or CLONE_TIMEOUT, stdin_devnull=True,
                   announce=False), ""

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
    """Best-effort `git submodule update` on its own budget.

    Kept separate from the clone's timeout so a heavy `lib/` tree (excluded from
    learning anyway) can never sink an otherwise-good checkout.
    """
    if not (dest / ".gitmodules").is_file():
        return
    ok, _ = git(["submodule", "update", "--init", "--recursive", "--depth", "1"],
                cwd=dest, capture=True, timeout=SUBMODULE_TIMEOUT)
    if not ok:
        print("    . submodule fetch incomplete (best-effort)", flush=True)


def clone_repo(url: str, dest: Path, commit: str | None) -> tuple[bool, str]:
    """Clone `url` into `dest`, preferring the pinned `commit`.

    These are ordinary upstream repos (not C4 mirrors), so the audited commit is
    fetched explicitly. If the sha is not fetchable we fall back to a plain
    shallow clone of the default branch — the report's scope was taken at that
    commit, but a slightly newer tree is far better than losing the contest
    entirely; the `.sol` check still gates acceptance.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)

    if commit:
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

    ok, output = git(["clone", "-q", "--depth", "1", url, str(dest)], capture=True)
    if not ok:
        return False, classify_git_failure(output)
    _update_submodules(dest)
    return True, ""


def clone_contest(record: ReportRecord, contracts_dir: Path) -> tuple[bool, str | None]:
    """Clone every repo for one contest under `contracts/<id>/`.

    Priority order:
      1. `https://github.com/sherlock-audit/<slug>` — the packaged mirror, which
         nests every in-scope repo and is therefore complete by construction.
      2. The repo(s) named in the report at their pinned commits.
      3. Neither works -> the contest is reported as a clone failure and lands
         in `skipped_sherlock.txt`.

    Returns `(ok, error)`; the sentinel is written only on full success.
    """
    dest_root = contracts_dir / str(record.id)
    sentinel = dest_root / CLONE_SENTINEL
    if sentinel.is_file() and has_solidity(dest_root):
        return True, None
    if dest_root.exists():
        shutil.rmtree(dest_root, ignore_errors=True)

    reasons: list[str] = []
    cloned_any = False

    def attempt(urls: list[str]) -> bool:
        nonlocal cloned_any
        any_ok = False
        for url in urls:
            parsed = github_repo(url)
            if parsed is None:
                continue
            org, repo = parsed
            commit = record.commit_for(url)
            ok, reason = clone_repo(url, dest_root / f"{org}-{repo}", commit)
            if ok:
                cloned_any = True
                any_ok = True
            else:
                reasons.append(f"{url}: {reason}")
                print(f"  ! clone failed ({reason}): {url}", flush=True)
        return any_ok

    mirror = f"https://github.com/sherlock-audit/{record.slug}"
    primary = [u for u in record.repos if u != mirror]

    # 1. Packaged mirror first: it nests every in-scope repo at the audited
    #    commit, so when it exists it is the most complete single source.
    complete = attempt([mirror]) and has_solidity(dest_root)
    # 2. Otherwise the repo(s) named in the report, at their pinned commits.
    if not complete and primary:
        print(f"  . mirror unavailable; falling back to the report's repo(s)", flush=True)
        complete = attempt(primary) and has_solidity(dest_root)

    if complete:
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
    return run([args.knowdit_bin, "learn",
                "--database-url", args.database_url, "link"])


# ---------------------------------------------------------------------------
# Manifest / reporting
# ---------------------------------------------------------------------------

def save_manifest(dataset_dir: Path, records: list[ReportRecord]) -> None:
    payload = [asdict(r) for r in sorted(records, key=lambda r: r.id)]
    (dataset_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def apply_manifest_resume(dataset_dir: Path, records: list[ReportRecord]) -> int:
    """Carry permanently-settled outcomes forward.

    Only deterministic decisions are carried: `learned`, the text-derived
    skips (`non-solidity`, `no-id`, `manual`). Transient failures
    (`clone-failed`, `empty`, `failed`) are NOT carried, so a network blip or a
    learn error is retried on the next run.
    """
    path = dataset_dir / "manifest.json"
    if not path.is_file():
        return 0
    try:
        previous = {int(e["id"]): e for e in json.loads(path.read_text())}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 0

    carried = {"learned", "skipped:non-solidity", "skipped:no-id", "skipped:manual"}
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

    Only a SUCCESSFUL learn cleans up in the batch loop, so without this every
    failed clone accumulates on disk. `active_ids` protects the in-flight batch.
    """
    if not contracts_dir.is_dir():
        return 0
    failed = {r.id for r in records
              if r.status in ("skipped:clone-failed", "skipped:empty", "failed")}
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
    """Write `skipped_sherlock.txt`: contests not learned, with reason + repos."""
    skipped = [r for r in records if r.status not in ("learned", "ready")]
    path = dataset_dir / SKIPPED_FILENAME
    if not skipped:
        if path.exists():
            path.unlink()
        return

    lines = [
        "# Sherlock contests NOT learned — handle these manually.",
        "# Each entry: contest_id  slug  status  reason  (repo URLs)",
        "",
    ]
    for record in sorted(skipped, key=lambda r: (r.id, r.slug)):
        lines.append(
            f"{record.id}\t{record.slug}\t{record.status}\t{record.error or ''}"
        )
        for url in record.repos:
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
        prog="build_sherlock_dataset.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reports-dir", type=Path, default=Path(DEFAULT_REPORTS_DIR),
                        help=f"Sherlock report directory (default: {DEFAULT_REPORTS_DIR}).")
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR),
                        help="Target C4-layout dataset dir with audits/ contracts/ "
                             f"reports/ (default: {DEFAULT_DATASET_DIR}).")
    parser.add_argument("--database-url", default=None,
                        help="Historical KG URL for `knowdit learn` (sqlite://, mysql://, "
                             "postgres://). Required unless --prepare-only/--skip-learn.")
    parser.add_argument("--knowdit-bin", default="./target/release/knowdit",
                        help="Path to the knowdit binary (default: ./target/release/knowdit).")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Contests cloned+learned per batch (default: 10).")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Passed through to `learn c4 --concurrency` (default: 1).")
    parser.add_argument("--merge-concurrency", type=int, default=DEFAULT_MERGE_CONCURRENCY,
                        help="Passed through to `learn c4 --merge-concurrency`: merge "
                             "agents in parallel. Merge is the serial bottleneck, so "
                             f"raising this is the main throughput lever (default: "
                             f"{DEFAULT_MERGE_CONCURRENCY}).")
    parser.add_argument("--context-window-utilization", type=float,
                        default=DEFAULT_CONTEXT_WINDOW_UTILIZATION,
                        help="Passed through to `learn c4 --context-window-utilization` "
                             f"(default: {DEFAULT_CONTEXT_WINDOW_UTILIZATION}).")
    parser.add_argument("--clone-timeout", type=int, default=DEFAULT_CLONE_TIMEOUT,
                        help=f"Timeout for the top-level `git clone` (default: "
                             f"{DEFAULT_CLONE_TIMEOUT}s).")
    parser.add_argument("--submodule-timeout", type=int, default=DEFAULT_SUBMODULE_TIMEOUT,
                        help="Timeout for best-effort `git submodule update` "
                             f"(default: {DEFAULT_SUBMODULE_TIMEOUT}s).")
    parser.add_argument("--skip-ids", action="append", default=[], metavar="IDS",
                        help="Contest ids and/or slugs to skip entirely (recorded as "
                             "`skipped:manual`). Repeatable / comma-separated. A "
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
                        help="Overwrite existing audits/reports and ignore resume state.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore manifest.json; re-clone and re-learn every ready contest.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and print the plan without writing anything.")
    return parser


def validate_numeric_args(args: argparse.Namespace) -> None:
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
    if not [r for r in records if r.status == "ready"]:
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

    records = [parse_report(path) for path in report_paths]

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
    print(f"\nPhase 1: wrote {counters['audit_written']} audits "
          f"({counters['audit_kept']} kept), copied {counters['report_copied']} reports "
          f"({counters['report_kept']} kept).")
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
