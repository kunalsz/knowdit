#!/usr/bin/env python3
"""Build a learnable dataset from Cantina competition reports, then learn it in
space-bounded batches — via the `learn c4` pipeline.

Why the C4 layout
-----------------
`learn c4` is the one ingestion path that wires BOTH source code -> semantics
AND report -> findings, and it keys every project on a NUMERIC u32 id:

    <dataset>/audits/<id>.json    contest metadata (contestId + title)
    <dataset>/contracts/<id>/     Solidity sources (walked for **/*.sol)
    <dataset>/reports/<id>.md     the markdown report

so Cantina reports — which are just `*.pdf.md` files with no id — are laid out
in that shape. The id becomes the KG platform id `c4-<id>` downstream, exactly
as the Sherlock builder does; the two datasets are intentionally
indistinguishable from genuine C4 contests in the database.

Report -> repo mapping
----------------------
Cantina reports do NOT contain the repo or the audited commit. Both live on the
competition page's "Scope" section, which is rendered client-side. The portfolio
page (`/portfolio?section=cantina-competitions`) ships them in its embedded
Next.js RSC payload as a per-competition object:

    {"title": "...", "pdfLink":
     "https://cdn.cantina.xyz/reports/<pdf-basename>.pdf",
     "repoLinks": ["https://github.com/<org>/<repo>", ...],
     "commitHashes": ["<sha>", ...]}

The report's filename IS the pdf basename (`<basename>.pdf` -> `<basename>.pdf.md`),
so that payload is the authoritative bridge from a local report to its exact
repo(s) + commit(s). The payload is scraped once and cached to
`<dataset>/mapping.json`, which `--mapping-json` reuses for offline runs.

Clone strategy
--------------
Every in-scope repo is cloned at its PINNED commit with

    git clone --revision=<sha> --depth=1 <url> <dest>

into `contracts/<id>/<repo-basename>/`. Multi-repo competitions (2-5 repos, e.g.
VII Finance) get every repo; the loader walks the tree recursively. Competitions
with no fetchable repo, or that yield no in-scope `.sol`, are recorded in
`skipped_cantina.txt`.

Usage:
    scripts/build_cantina_dataset.py --dry-run
    scripts/build_cantina_dataset.py --prepare-only
    scripts/build_cantina_dataset.py --skip-learn --batch-size 5
    scripts/build_cantina_dataset.py --batch-size 5 \\
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
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_REPORTS_DIR = "/home/zeit/Desktop/reports/cantina/solidity-competitions"
DEFAULT_DATASET_DIR = "cantina"

PORTFOLIO_URL = "https://cantina.xyz/portfolio?section=cantina-competitions"

# Directory names the C4 loader drops from a project's scope (mirrors
# `DEFAULT_VENDORED_EXCLUDES` in crates/knowdit-project/src/data.rs).
EXCLUDED_DIRS = {
    "node_modules", "lib", "build", "artifacts", "cache",
    "test", "mock", "deploy", "script", "target", ".git",
}

SKIPPED_FILENAME = "skipped_cantina.txt"
SKIP_LIST_FILENAME = "skip-ids.txt"
MAPPING_FILENAME = "mapping.json"

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

# Cantina reports have no numeric id, so ids are synthesized. A C4 id is a u32
# whose real values top out around 20_000_000 in practice, so this reserved
# 90_000_000+ band cannot collide with a genuine contest id. Ids are assigned
# in sorted-report order, deterministically.
ID_BASE = 90_000_000

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

GITHUB_URL_RE = re.compile(r"https?://github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
NEXT_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', re.DOTALL)


# ---------------------------------------------------------------------------
# Portfolio scraping (report basename -> repos + commits)
# ---------------------------------------------------------------------------

@dataclass
class Competition:
    """One portfolio entry: its report PDF and the exact audited scope."""

    title: str
    pdf_basename: str
    repo_urls: list[str] = field(default_factory=list)
    commits: list[str] = field(default_factory=list)
    kind: str | None = None


def fetch_portfolio_html(url: str = PORTFOLIO_URL) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "accept": "text/html",
            "user-agent": "Mozilla/5.0 (compatible; knowdit-build)",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8", errors="replace")


def rsc_payload(html: str) -> str:
    """Concatenate and unescape every `self.__next_f.push([1,"..."])` chunk.

    This is the serialized React Server Components tree; the competition
    objects (with `pdfLink` / `repoLinks` / `commitHashes`) are buried in it.
    """
    raw = "".join(NEXT_PUSH_RE.findall(html))
    # Each chunk is the body of a JS string literal. Wrap it back in quotes and
    # let JSON do the unescaping; fall back to Python's unicode_escape for the
    # handful of chunks JSON rejects.
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw.encode("utf-8", errors="surrogatepass").decode(
            "unicode_escape", errors="replace"
        )


def _enclosing_object(text: str, index: int) -> str:
    """The `{...}` object that encloses `index`, via brace matching."""
    depth = 0
    start = index
    while start > 0:
        char = text[start]
        if char == "}":
            depth += 1
        elif char == "{":
            if depth == 0:
                break
            depth -= 1
        start -= 1
    depth = 0
    end = start
    while end < len(text):
        char = text[end]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        end += 1
    return text[start:end + 1]


def parse_portfolio(html: str) -> dict[str, Competition]:
    """`pdf-basename -> Competition` from the portfolio page payload.

    Walks every `"pdfLink"` occurrence, lifts its enclosing object, and keeps the
    ones that carry a usable `repoLinks` / `commitHashes` pair. Entries whose
    repo list is empty or whose commits are absent are dropped — they cannot be
    cloned anyway.
    """
    payload = rsc_payload(html)
    found: dict[str, Competition] = {}
    for match in re.finditer(r'"pdfLink"', payload):
        fragment = _enclosing_object(payload, match.start())
        try:
            obj = json.loads(fragment)
        except json.JSONDecodeError:
            continue
        pdf = obj.get("pdfLink") or ""
        basename = os.path.basename(pdf)
        if not basename:
            continue
        repos = [r for r in (obj.get("repoLinks") or []) if isinstance(r, str)]
        commits = [c for c in (obj.get("commitHashes") or []) if isinstance(c, str)]
        if not repos:
            continue
        found[basename] = Competition(
            title=obj.get("title") or basename,
            pdf_basename=basename,
            repo_urls=repos,
            commits=commits,
            kind=obj.get("kind"),
        )
    return found


def load_mapping(mapping_json: Path | None, offline_url: str) -> dict[str, Competition]:
    if mapping_json and mapping_json.is_file():
        raw = json.loads(mapping_json.read_text(encoding="utf-8"))
        return {
            entry["pdf_basename"]: Competition(**entry)
            for entry in raw
        }
    print(f"Fetching portfolio mapping from {offline_url} ...", flush=True)
    return parse_portfolio(fetch_portfolio_html(offline_url))


def save_mapping(dataset_dir: Path, mapping: dict[str, Competition]) -> None:
    payload = [asdict(c) for c in sorted(mapping.values(), key=lambda c: c.pdf_basename)]
    (dataset_dir / MAPPING_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Report records
# ---------------------------------------------------------------------------

@dataclass
class ReportRecord:
    slug: str
    id: int
    title: str
    repos: list[str] = field(default_factory=list)
    commits: list[str] = field(default_factory=list)
    status: str = "ready"
    error: str | None = None

    @property
    def primary_url(self) -> str | None:
        return self.repos[0] if self.repos else None

    def commit_for(self, index: int) -> str | None:
        """Commit for repo `index`; falls back to the first when lists differ."""
        if index < len(self.commits):
            return self.commits[index]
        return self.commits[0] if self.commits else None


def github_repo(url: str) -> tuple[str, str] | None:
    match = GITHUB_URL_RE.search(url)
    if match is None:
        return None
    return match.group(1), match.group(2)


def assign_ids(slugs: list[str]) -> dict[str, int]:
    """`slug -> ID_BASE + 1-based index`, in sorted-slug order (deterministic)."""
    return {slug: ID_BASE + index for index, slug in enumerate(sorted(slugs), start=1)}


def parse_report(path: Path, contest_id: int, mapping: dict[str, Competition]) -> ReportRecord:
    slug = path.stem
    # The report file is `<pdf-basename>.pdf.md`; the payload keys on the PDF
    # basename, i.e. the filename with the trailing `.md` removed.
    pdf_basename = path.name[:-3] if path.name.endswith(".md") else path.name
    entry = mapping.get(pdf_basename)
    if entry is None:
        return ReportRecord(slug=slug, id=contest_id, title=slug,
                            status="skipped:no-scope",
                            error=f"no portfolio entry for pdf {pdf_basename}")

    repos: list[str] = []
    seen: set[str] = set()
    for raw in entry.repo_urls:
        parsed = github_repo(raw)
        if parsed is None:
            continue
        org, repo = parsed
        url = f"https://github.com/{org}/{repo}"
        if url in seen:
            continue
        seen.add(url)
        repos.append(url)
    if not repos:
        return ReportRecord(slug=slug, id=contest_id, title=entry.title,
                            status="skipped:no-repo",
                            error="portfolio entry has no usable GitHub repo")

    commits = [c for c in entry.commits if COMMIT_RE.match(c)]
    return ReportRecord(slug=slug, id=contest_id, title=entry.title,
                        repos=repos, commits=commits)


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
    """Clone `url` into `dest` at the pinned `commit`.

    Uses git's native `--revision=<sha> --depth=1`, which fetches exactly that
    object and checks it out detached. If the sha is not fetchable (e.g. it was
    garbage-collected upstream) we fall back to a plain shallow clone of the
    default branch — a slightly newer tree is far better than losing the
    contest; the post-clone `.sol` check still gates acceptance.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)

    if commit:
        ok, output = git(
            ["clone", "--revision", commit, "--depth", "1", url, str(dest)],
            capture=True,
        )
        if ok:
            _update_submodules(dest)
            return True, ""
        print(f"  ! revision {commit[:12]} not fetchable; retrying default branch",
              flush=True)
        shutil.rmtree(dest, ignore_errors=True)

    ok, output = git(["clone", "-q", "--depth", "1", url, str(dest)], capture=True)
    if not ok:
        return False, classify_git_failure(output)
    _update_submodules(dest)
    return True, ""


def clone_contest(record: ReportRecord, contracts_dir: Path) -> tuple[bool, str | None]:
    """Clone every in-scope repo for one contest under `contracts/<id>/`.

    Each repo lands in its own `<org>-<repo>/` subdirectory so multi-repo
    competitions don't collide; the loader walks the tree recursively.
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

    for index, url in enumerate(record.repos):
        parsed = github_repo(url)
        if parsed is None:
            continue
        org, repo = parsed
        commit = record.commit_for(index)
        ok, reason = clone_repo(url, dest_root / f"{org}-{repo}", commit)
        if ok:
            cloned_any = True
        else:
            reasons.append(f"{url}: {reason}")
            print(f"  ! clone failed ({reason}): {url}", flush=True)

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
    """Carry permanently-settled outcomes forward.

    Only deterministic decisions are carried: `learned`, the text-derived skips
    (`no-scope`, `no-repo`, `manual`). Transient failures (`clone-failed`,
    `empty`, `failed`) are NOT carried, so a network blip or a learn error is
    retried on the next run.
    """
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
        "# Cantina competitions NOT learned — handle these manually.",
        "# Each entry: id  slug  status  reason  (repo URLs @ commits)",
        "",
    ]
    for record in sorted(skipped, key=lambda r: (r.id, r.slug)):
        lines.append(f"{record.id}\t{record.slug}\t{record.status}\t{record.error or ''}")
        for index, url in enumerate(record.repos):
            commit = record.commit_for(index) or "?"
            lines.append(f"\t\t{url} @ {commit}")
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
        prog="build_cantina_dataset.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reports-dir", type=Path, default=Path(DEFAULT_REPORTS_DIR),
                        help=f"Cantina report .md files (default: {DEFAULT_REPORTS_DIR}).")
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR),
                        help="Target C4-layout dataset dir with audits/ contracts/ "
                             f"reports/ (default: {DEFAULT_DATASET_DIR}).")
    parser.add_argument("--mapping-json", type=Path, default=None,
                        help="Reuse a previously scraped portfolio mapping instead of "
                             f"re-fetching the page (defaults to <dataset>/{MAPPING_FILENAME} "
                             "when present).")
    parser.add_argument("--portfolio-url", default=PORTFOLIO_URL,
                        help=f"Portfolio page to scrape (default: {PORTFOLIO_URL}).")
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
                        help="Overwrite existing audits/reports files and ignore resume state.")
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

    # The mapping is needed even for a dry-run, so it is fetched/cached whether
    # or not anything is written.
    mapping_json = args.mapping_json
    if mapping_json is None:
        candidate = args.dataset_dir / MAPPING_FILENAME
        mapping_json = candidate if candidate.is_file() else None
    mapping = load_mapping(mapping_json, args.portfolio_url)
    if not mapping:
        raise SystemExit("portfolio mapping is empty (scrape failed or layout changed)")

    ids = assign_ids([p.stem for p in report_paths])
    records = [parse_report(path, ids[path.stem], mapping) for path in report_paths]

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
        for record in sorted(records, key=lambda r: r.id):
            print(f"  {record.id}  {record.slug:44}  {record.status:22}  "
                  f"{(record.primary_url or '-')}")
        print("\n[dry-run] no files written.")
        return 0

    check_preconditions(args, records)

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    save_mapping(args.dataset_dir, mapping)

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
