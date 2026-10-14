#!/usr/bin/env python3
"""Build a learnable dataset from iam0x52 audit reports, then learn it in
space-bounded batches — via the `learn c4` pipeline.

Why the C4 layout
-----------------
`learn c4` is the one ingestion path that wires BOTH source code -> semantics
AND report -> findings, and it keys every project on a NUMERIC u32 id:

    <dataset>/audits/<id>.json    contest metadata (contestId + title)
    <dataset>/contracts/<id>/     Solidity sources (walked for **/*.sol)
    <dataset>/reports/<id>.md     the markdown report

so iam0x52 reports — which carry no id — are laid out in that shape. The id
becomes the KG platform id `c4-<id>` downstream, exactly as the other builders
do, so the datasets are indistinguishable in the database.

Report -> repo mapping
----------------------
Each report is a flat markdown file with a `Scope` section that names the repo
and the exact reviewed commit:

    ## <br/> Scope

    Repo: [Blueberry-staking](https://github.com/Blueberryfi/blueberry-staking/)

    Review Hash: [f45c0d](https://github.com/Blueberryfi/blueberry-staking/blob/efaf7fc690e38914ba475d5ac61a4d0bd3f45c0d)

The Review Hash link's URL is the authority: it carries both the repo AND the
full 40-char commit sha (the visible label is only a 7-char abbreviation). The
URL comes in several shapes — `/blob/<sha>/`, `/tree/<sha>`, `/commit/<sha>`,
even `/pull/<n>/commits/<sha>` — all of which the sha extraction handles.

A report can scope several repos (a `Repo:`/`Review Hash:` pair each). A report
whose scope is `N/A` (deployed-contracts-only reviews, no repo) is recorded as
`skipped:no-scope` — there is nothing to clone — and listed in
`skipped_iam0x52.txt`.

Clone strategy
--------------
Every scoped repo is cloned at its reviewed commit with

    git clone --revision=<sha> --depth=1 <url> <dest>

into `contracts/<id>/<org>-<repo>/`. An unfetchable sha (private/deleted repo)
falls back to a plain shallow clone of the default branch; the post-clone
`.sol` check still gates acceptance. Engagements with no fetchable repo, or that
yield no in-scope `.sol`, are recorded in `skipped_iam0x52.txt`.

Usage:
    scripts/build_iam0x52_dataset.py --dry-run
    scripts/build_iam0x52_dataset.py --prepare-only
    scripts/build_iam0x52_dataset.py --skip-learn --batch-size 5
    scripts/build_iam0x52_dataset.py --batch-size 5 \\
        --database-url sqlite:///contest_kg.db
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

DEFAULT_REPORTS_DIR = "/home/zeit/Desktop/reports/iam0x52"
DEFAULT_DATASET_DIR = "iam0x52"

# Directory names the C4 loader drops from a project's scope (mirrors
# `DEFAULT_VENDORED_EXCLUDES` in crates/knowdit-project/src/data.rs).
EXCLUDED_DIRS = {
    "node_modules", "lib", "build", "artifacts", "cache",
    "test", "mock", "deploy", "script", "target", ".git",
}

SKIPPED_FILENAME = "skipped_iam0x52.txt"
SKIP_LIST_FILENAME = "skip-ids.txt"

DEFAULT_CLONE_TIMEOUT = 900
DEFAULT_SUBMODULE_TIMEOUT = 300

# Written into `contracts/<id>/` only after a clone fully completed. Its
# presence is what licenses REUSING a tree instead of re-cloning.
CLONE_SENTINEL = ".knowdit-clone-complete"

DEFAULT_MERGE_CONCURRENCY = 4
DEFAULT_CONTEXT_WINDOW_UTILIZATION = 0.7
CLONE_TIMEOUT = DEFAULT_CLONE_TIMEOUT
SUBMODULE_TIMEOUT = DEFAULT_SUBMODULE_TIMEOUT

# iam0x52 reports have no numeric id, so ids are synthesized. C4 ids are u32 and
# top out around 20_000_000 in practice; the other synthetic datasets use
# 90_000_000+ (Cantina competitions), 91_000_000+ (Spearbit), 92_000_000+
# (Cantina reviews), and 93_000_000+ (Immunefi). This reserved 94_000_000+ band
# keeps the six synthetic datasets apart and clear of any genuine contest id.
ID_BASE = 94_000_000

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
    "-c", "url.https://github.com/.insteadOf=ssh://git@github.com/",
]

# Scope section: `## <br/> Scope` up to the next `##` heading.
SCOPE_HEADING_RE = re.compile(r"^##\s*<br/>\s*Scope\s*$", re.MULTILINE)
NEXT_H2_RE = re.compile(r"^##\s", re.MULTILINE)
TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
REPO_LINE_RE = re.compile(r"^Repo:\s*(.*)$", re.MULTILINE)
REVIEW_HASH_LINE_RE = re.compile(r"^Review Hash:\s*(.*)$", re.MULTILINE)
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
GITHUB_URL_RE = re.compile(r"https?://github\.com/+([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+)")
FULL_SHA_RE = re.compile(r"\b([0-9a-fA-F]{40})\b")


# ---------------------------------------------------------------------------
# Scope extraction
# ---------------------------------------------------------------------------

@dataclass
class ScopeRepo:
    url: str
    ref: str | None


@dataclass
class ReportRecord:
    slug: str
    id: int
    title: str
    scope: list[ScopeRepo] = field(default_factory=list)
    status: str = "ready"
    error: str | None = None

    @property
    def primary_url(self) -> str | None:
        return self.scope[0].url if self.scope else None


def scope_section(text: str) -> str:
    match = SCOPE_HEADING_RE.search(text)
    if match is None:
        return ""
    rest = text[match.end():]
    nxt = NEXT_H2_RE.search(rest)
    return rest[: nxt.start()] if nxt else rest


def _first_link_url(line: str) -> str | None:
    """First markdown-link target on a `Repo:` / `Review Hash:` line."""
    match = MARKDOWN_LINK_RE.search(line)
    return match.group(1) if match else None


def parse_scope(scope: str) -> list[ScopeRepo]:
    """`(repo, reviewed commit)` pairs from a report's Scope section.

    Walks the lines so each `Review Hash:` binds to the `Repo:` above it. The
    review-hash URL is authoritative — it carries both the repo and the full
    40-char sha — so a pair is emitted only when that URL names a GitHub repo;
    the sha is taken from the same URL when present.
    """
    scope_repos: list[ScopeRepo] = []
    seen: set[tuple[str, str | None]] = set()

    for line in scope.splitlines():
        if not REVIEW_HASH_LINE_RE.match(line):
            continue
        url = _first_link_url(line)
        if url is None:
            continue
        repo_match = GITHUB_URL_RE.search(url)
        if repo_match is None:
            continue
        org, repo = repo_match.group(1), repo_match.group(2)
        repo = repo[:-4] if repo.endswith(".git") else repo
        canonical = f"https://github.com/{org}/{repo}"
        sha_match = FULL_SHA_RE.search(url)
        ref = sha_match.group(1) if sha_match else None
        key = (canonical, ref)
        if key in seen:
            continue
        seen.add(key)
        scope_repos.append(ScopeRepo(url=canonical, ref=ref))

    return scope_repos


def parse_report(path: Path, contest_id: int) -> ReportRecord:
    text = path.read_text(encoding="utf-8", errors="replace")
    slug = path.stem
    title_match = TITLE_RE.search(text)
    title = title_match.group(1).strip() if title_match else slug

    scope = parse_scope(scope_section(text))
    if not scope:
        return ReportRecord(slug=slug, id=contest_id, title=title,
                            status="skipped:no-scope",
                            error="no GitHub repo + review hash in the Scope section")
    return ReportRecord(slug=slug, id=contest_id, title=title, scope=scope)


# ---------------------------------------------------------------------------
# Manual skip list
# ---------------------------------------------------------------------------

def _slug_sentinel(slug: str) -> int:
    return -abs(hash(slug)) - 1


def parse_skip_ids(values: list[str], dataset_dir: Path) -> set[int]:
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

def assign_ids(slugs: list[str]) -> dict[str, int]:
    """`slug -> ID_BASE + 1-based index`, in sorted-slug order (deterministic)."""
    return {slug: ID_BASE + index for index, slug in enumerate(sorted(slugs), start=1)}


def audit_payload(record: ReportRecord) -> str:
    return json.dumps(
        {"contestId": record.id, "title": record.title, "slug": record.slug},
        indent=2,
    ) + "\n"


def check_collisions(audits_dir: Path, records: list[ReportRecord]) -> None:
    seen: dict[int, str] = {}
    for record in records:
        if record.status != "ready":
            continue
        if record.id in seen:
            raise SystemExit(
                f"duplicate id {record.id}: {seen[record.id]} and {record.slug}."
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
        "synthesized ids.",
        "Re-run with --force to overwrite, or point --dataset-dir at a clean directory.",
        "",
    ]
    for record, path in conflicts:
        lines.append(f"  {path.name}: exists, wanted slug {record.slug}")
    raise SystemExit("\n".join(lines))


def materialize(reports_dir: Path, dataset_dir: Path, records: list[ReportRecord],
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
        return "ref not fetchable"
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
    """Best-effort `git submodule update` on its own budget."""
    if not (dest / ".gitmodules").is_file():
        return
    ok, _ = git(["submodule", "update", "--init", "--recursive", "--depth", "1"],
                cwd=dest, capture=True, timeout=SUBMODULE_TIMEOUT)
    if not ok:
        print("    . submodule fetch incomplete (best-effort)", flush=True)


def repo_dir_name(url: str) -> str:
    match = GITHUB_URL_RE.search(url)
    if match is None:
        return "repo"
    return f"{match.group(1)}-{match.group(2)}"


def clone_repo(url: str, dest: Path, ref: str | None) -> tuple[bool, str]:
    """Clone `url` into `dest` at the reviewed commit (or the default branch)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)

    if ref:
        ok, _ = git(
            ["clone", "--revision", ref, "--depth", "1", url, str(dest)],
            capture=True,
        )
        if ok:
            _update_submodules(dest)
            return True, ""
        print(f"  ! revision {ref[:12]} not fetchable; retrying default branch",
              flush=True)
        shutil.rmtree(dest, ignore_errors=True)

    ok, output = git(["clone", "-q", "--depth", "1", url, str(dest)], capture=True)
    if not ok:
        return False, classify_git_failure(output)
    _update_submodules(dest)
    return True, ""


def clone_contest(record: ReportRecord, contracts_dir: Path) -> tuple[bool, str | None]:
    """Clone every scoped repo for one report under `contracts/<id>/`."""
    dest_root = contracts_dir / str(record.id)
    sentinel = dest_root / CLONE_SENTINEL
    if sentinel.is_file() and has_solidity(dest_root):
        return True, None
    if dest_root.exists():
        shutil.rmtree(dest_root, ignore_errors=True)

    reasons: list[str] = []
    cloned_any = False
    for scope in record.scope:
        dest = dest_root / repo_dir_name(scope.url)
        ok, reason = clone_repo(scope.url, dest, scope.ref)
        if ok:
            cloned_any = True
        else:
            reasons.append(f"{scope.url}@{scope.ref or 'HEAD'}: {reason}")
            print(f"  ! clone failed ({reason}): {scope.url}", flush=True)

    if cloned_any and has_solidity(dest_root):
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
    """Carry permanently-settled outcomes forward (learned, deterministic skips)."""
    path = dataset_dir / "manifest.json"
    if not path.is_file():
        return 0
    try:
        previous = {int(e["id"]): e for e in json.loads(path.read_text())}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 0

    carried = {"learned", "skipped:no-scope", "skipped:no-repo", "skipped:manual"}
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
    skipped = [r for r in records if r.status not in ("learned", "ready")]
    path = dataset_dir / SKIPPED_FILENAME
    if not skipped:
        if path.exists():
            path.unlink()
        return

    lines = [
        "# iam0x52 reports NOT learned — handle these manually.",
        "# Each entry: id  slug  status  reason  (repo URLs @ refs)",
        "",
    ]
    for record in sorted(skipped, key=lambda r: (r.id, r.slug)):
        lines.append(f"{record.id}\t{record.slug}\t{record.status}\t{record.error or ''}")
        for scope in record.scope:
            lines.append(f"\t\t{scope.url} @ {scope.ref or 'HEAD'}")
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
        prog="build_iam0x52_dataset.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reports-dir", type=Path, default=Path(DEFAULT_REPORTS_DIR),
                        help=f"iam0x52 report .md files (default: {DEFAULT_REPORTS_DIR}).")
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR),
                        help="Target C4-layout dataset dir with audits/ contracts/ "
                             f"reports/ (default: {DEFAULT_DATASET_DIR}).")
    parser.add_argument("--database-url", default=None,
                        help="Historical KG URL for `knowdit learn` (sqlite://, mysql://, "
                             "postgres://). Required unless --prepare-only/--skip-learn.")
    parser.add_argument("--knowdit-bin", default="./target/release/knowdit",
                        help="Path to the knowdit binary (default: ./target/release/knowdit).")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Reports cloned+learned per batch (default: 10).")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Passed through to `learn c4 --concurrency` (default: 1).")
    parser.add_argument("--merge-concurrency", type=int, default=DEFAULT_MERGE_CONCURRENCY,
                        help="Passed through to `learn c4 --merge-concurrency` (default: "
                             f"{DEFAULT_MERGE_CONCURRENCY}).")
    parser.add_argument("--context-window-utilization", type=float,
                        default=DEFAULT_CONTEXT_WINDOW_UTILIZATION,
                        help="Passed through to `learn c4 --context-window-utilization` "
                             f"(default: {DEFAULT_CONTEXT_WINDOW_UTILIZATION}).")
    parser.add_argument("--clone-timeout", type=int, default=DEFAULT_CLONE_TIMEOUT,
                        help=f"Timeout for a single `git clone` (default: "
                             f"{DEFAULT_CLONE_TIMEOUT}s).")
    parser.add_argument("--submodule-timeout", type=int, default=DEFAULT_SUBMODULE_TIMEOUT,
                        help="Timeout for best-effort `git submodule update` "
                             f"(default: {DEFAULT_SUBMODULE_TIMEOUT}s).")
    parser.add_argument("--skip-ids", action="append", default=[], metavar="IDS",
                        help="Report ids and/or slugs to skip entirely (recorded as "
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
                        help="Overwrite existing audits/reports files and ignore resume state.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore manifest.json; re-clone and re-learn every ready report.")
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
        raise SystemExit("no ready reports to process")


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
            print(f"Resuming: {resumed} report(s) already settled in manifest.json "
                  "(use --no-resume or --force to redo them).")

    skip = parse_skip_ids(args.skip_ids, args.dataset_dir)
    if skip:
        n = apply_skip_list(records, skip)
        if n:
            print(f"Skipping {n} report(s) from --skip-ids / {SKIP_LIST_FILENAME}.")

    validate_numeric_args(args)

    print_summary(records)
    print(f"\nDataset dir: {args.dataset_dir}")
    print(f"Learn flags: --concurrency {args.concurrency} "
          f"--merge-concurrency {args.merge_concurrency} "
          f"--context-window-utilization {args.context_window_utilization}")
    if args.dry_run:
        for record in sorted(records, key=lambda r: r.id):
            print(f"  {record.id}  {record.slug:44}  {record.status:22}  "
                  f"{record.primary_url or '-'}")
        print("\n[dry-run] no files written.")
        return 0

    check_preconditions(args, records)

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    counters = materialize(args.reports_dir, args.dataset_dir, records,
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
        print(f"\n=== Batch {index}/{len(batches)}: {len(batch)} report(s) ===", flush=True)

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
            print("\n--skip-learn: cloned this batch and stopping; contracts/ left in place.")
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
            print(f"\n[batch {index}] learned {len(ids_to_learn)} report(s).", flush=True)
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
        print(f"\nReports needing manual attention: {skipped_path}")
    if skipped_notes:
        print("\nNot learned:")
        for note in skipped_notes:
            print(f"  - {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
