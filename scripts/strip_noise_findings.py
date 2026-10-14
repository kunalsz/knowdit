#!/usr/bin/env python3
"""Strip gas / informational / non-critical findings from C4 reports.

The learn pipeline extracts High/Medium/Low findings, but it reads the WHOLE
report — so the `[G-xx]` gas optimizations, `[N-xx]` non-critical notes,
`[QA-xx]` nitpicks and `[I-xx]` informational items are fed to the model as
well, burning tokens (and occasionally landing as noise `Low` findings). This
script removes that material before learning.

Two granularities, because C4 uses two layouts:

  1. Whole sections that are ENTIRELY noise are deleted outright:
       # Gas Optimizations (31)
       # Gas Optimizations [13]
       # Gas Report / # Gas Optimizations Report
       # Non-Critical Findings (10)
       # Informational Findings (1)
       # Informational Issues
       # QA Report

  2. Sections that MIX Low with noise keep their Low findings and lose only the
     noise entries — deleting the section would drop real Low-severity data:
       # Low Risk and Non-Critical Issues      -> drop [N-xx], keep [L-xx]
       # Low Risk and Informational Issues     -> drop [I-xx]/[N-xx], keep [L-xx]

     Inside those, both heading-style (`## [N-01] ...`) and bullet-style
     (`* [[N-01] ...](...)`) entries are removed; Low (`[L-xx]`) is preserved.

Reports are only touched from `--from-year` (default 2022) onward; 2021 reports
use a different structure and are left alone.

Defaults to a DRY RUN (prints what it would do). Pass --apply to write.

Usage:
    scripts/strip_noise_findings.py                          # dry run
    scripts/strip_noise_findings.py --apply --backup-dir /tmp/c4bak
    scripts/strip_noise_findings.py --from-year 2023 --apply
    scripts/strip_noise_findings.py --only 20230101 --apply   # single id/file
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DEFAULT_DIR = "contests/reports"

# A top-level (`# `) section heading, normalised: lowercase, counts removed.
SECTION_COUNT_RE = re.compile(r"\s*[\(\[]\d+[\)\]]\s*$")

# Sections that are 100% noise -> delete the whole section.
PURE_NOISE_SECTIONS = {
    "gas optimizations",
    "gas optimization",
    "gas optimizations report",
    "gas report",
    "non-critical findings",
    "non-critical issues",
    "informational findings",
    "informational issues",
    "qa report",
}

# Sections that MIX Low with noise (`# Low Risk and Non-Critical Issues`,
# `# Low Risk and Informational Issues`) are NOT in PURE_NOISE_SECTIONS — they
# keep the section so their `[L-xx]` findings survive, and the per-marker scan
# in strip_report removes only the `[N-xx]`/`[I-xx]` entries inside them.

# Noise-finding marker codes: N/NC = non-critical, QA = QA report,
# I/INFO = informational, G = gas. `L` is deliberately absent — Low findings
# are real data and must survive.
_NOISE_CODES = r"(?:NC|QA|INFO|N|I|G)"

# The `-<digits>` suffix is REQUIRED: without it `[In ...]` / `[Governor]`
# would match the bare `I`/`G` prefix and prose headings would be deleted.
# Every real marker in the corpus is `[N-01]` / `[G-4]` / `[INFO-1]` shaped.
_MARKER = rf"\[\[?\s*{_NOISE_CODES}\s*[-–—]\s*\d+\s*\]"

# Heading-style finding marker: `## [N-01] ...` / `### [G-01] ...` (also `[[N-01]`).
FINDING_HEADING_RE = re.compile(rf"^(#{{1,6}})\s*{_MARKER}", re.IGNORECASE)

# Bullet-style finding marker: `* [[N-01] title](url) ...`.
FINDING_BULLET_RE = re.compile(rf"^\s*[*+-]\s*{_MARKER}", re.IGNORECASE)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass
class ReportResult:
    path: Path
    changed: bool = False
    sections_removed: list[str] = field(default_factory=list)
    findings_removed: list[str] = field(default_factory=list)
    bullets_removed: int = 0
    chars_before: int = 0
    chars_after: int = 0

    @property
    def chars_saved(self) -> int:
        return max(0, self.chars_before - self.chars_after)


def normalise_section(title: str) -> str:
    """Lowercase a section title and drop a trailing `(31)` / `[13]` count."""
    return SECTION_COUNT_RE.sub("", title.strip().lower()).strip()


def section_heading(lines: list[str], index: int) -> bool:
    """True when line `index` is a top-level `# ` heading at column 0."""
    match = HEADING_RE.match(lines[index])
    return bool(match) and len(match.group(1)) == 1


def find_sections(lines: list[str]) -> list[tuple[int, str]]:
    """`(line_index, normalised_title)` for each top-level `# ` heading.

    Fenced code blocks are deliberately NOT tracked: the scraped reports
    frequently contain unbalanced ``` fences (odd counts inside a section),
    which would desynchronise a parity counter and hide every later section.
    The cost of ignoring fences is that a `# <known section title>` line inside
    a code block would be mistaken for a boundary — which does not occur in
    practice, and the titles we match are specific enough that a stray shell
    comment cannot collide.
    """
    sections: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if line.startswith("# ") or line.rstrip() == "#":
            sections.append((index, normalise_section(line[1:].strip())))
    return sections


def section_end(lines: list[str], start: int) -> int:
    """Index just past the section starting at `start` (next top-level `# `)."""
    for index in range(start + 1, len(lines)):
        if section_heading(lines, index):
            return index
    return len(lines)


def finding_end(lines: list[str], start: int, level: int) -> int:
    """Index just past a finding: the next heading at the same or higher level.

    No fence tracking (see [`find_sections`]); a `##`/`#` line inside a code
    block is rare enough in these reports that the simpler rule is safer than a
    parity counter that can desynchronise.
    """
    for index in range(start + 1, len(lines)):
        match = HEADING_RE.match(lines[index])
        if match and len(match.group(1)) <= level:
            return index
    return len(lines)


def trim_blank_edges(lines: list[str], start: int, end: int) -> tuple[int, int]:
    """Shrink a removed span so it does not leave stray blank lines behind."""
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return start, end


def strip_report(text: str, path: Path) -> tuple[str, ReportResult]:
    lines = text.splitlines()
    result = ReportResult(path=path, chars_before=len(text))

    # Decide per section, then mark spans for deletion.
    delete = [False] * len(lines)
    sections = find_sections(lines)

    for position, (start, title) in enumerate(sections):
        end = section_end(lines, start)

        if title in PURE_NOISE_SECTIONS:
            span_start, span_end = trim_blank_edges(lines, start, end)
            for index in range(span_start, span_end):
                delete[index] = True
            result.sections_removed.append(lines[start].strip())
            continue

        # Every other section (including the mixed Low+noise ones AND
        # unrecognised headings like `# Audit Analysis` or a stray `# `, where
        # some reports park their gas findings) is scanned for noise findings.
        # A pure-noise section was already deleted above, so this only needs to
        # remove individual markers here. Matching is on the `[N-xx]` marker
        # alone (no fence tracking) for the same unbalanced-fence reason as
        # find_sections: a marker only ever appears in prose, never mid-codeblock.
        index = start + 1
        while index < end:
            line = lines[index]
            heading = FINDING_HEADING_RE.match(line)
            if heading:
                level = len(heading.group(1))
                stop = finding_end(lines, index, level)
                span_start, span_end = trim_blank_edges(lines, index, stop)
                for inner in range(span_start, span_end):
                    delete[inner] = True
                result.findings_removed.append(line.strip())
                index = stop
                continue

            if FINDING_BULLET_RE.match(line):
                delete[index] = True
                result.bullets_removed += 1
            index += 1

    if not any(delete):
        result.chars_after = len(text)
        return text, result

    kept = [line for index, line in enumerate(lines) if not delete[index]]

    # Collapse any runs of blank lines left where a section used to be.
    out: list[str] = []
    blanks = 0
    for line in kept:
        if line.strip():
            blanks = 0
            out.append(line)
        else:
            blanks += 1
            if blanks <= 2:
                out.append(line)

    new_text = "\n".join(out)
    if text.endswith("\n"):
        new_text += "\n"
    else:
        new_text = new_text.rstrip("\n")
    if new_text and not new_text.endswith("\n"):
        new_text += "\n"

    result.changed = new_text != text
    result.chars_after = len(new_text)
    return new_text, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="strip_noise_findings.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", type=Path, default=Path(DEFAULT_DIR),
                        help=f"Reports directory (default: {DEFAULT_DIR}).")
    parser.add_argument("--from-year", type=int, default=2022,
                        help="Only touch reports from this year onward (default: 2022).")
    parser.add_argument("--apply", action="store_true",
                        help="Write changes. Without this it is a dry run.")
    parser.add_argument("--backup-dir", type=Path, default=None,
                        help="Copy each original here before overwriting.")
    parser.add_argument("--only", action="append", default=[],
                        help="Restrict to specific report ids/filenames (repeatable).")
    parser.add_argument("--quiet", action="store_true", help="Only print the summary.")
    args = parser.parse_args(argv)

    if not args.dir.is_dir():
        raise SystemExit(f"reports dir not found: {args.dir}")

    paths = sorted(args.dir.glob("*.md"))
    if args.only:
        wanted = {o.removesuffix(".md") for o in args.only}
        paths = [p for p in paths if p.stem in wanted]
    else:
        paths = [p for p in paths if p.stem[:4].isdigit() and int(p.stem[:4]) >= args.from_year]

    if not paths:
        raise SystemExit("no reports matched")

    if args.backup_dir and args.apply:
        args.backup_dir.mkdir(parents=True, exist_ok=True)

    results: list[ReportResult] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"  ! {path.name}: {exc}", file=sys.stderr)
            continue
        new_text, result = strip_report(text, path)
        results.append(result)

        if result.changed and args.apply:
            if args.backup_dir:
                shutil.copy2(path, args.backup_dir / path.name)
            path.write_text(new_text, encoding="utf-8")

        if result.changed and not args.quiet:
            removed = len(result.sections_removed)
            print(
                f"  {path.name}: -{removed} section(s), "
                f"-{len(result.findings_removed)} finding(s), "
                f"-{result.bullets_removed} bullet(s), "
                f"{result.chars_saved:,} chars"
            )
            for title in result.sections_removed:
                print(f"        section: {title}")
            for finding in result.findings_removed[:6]:
                print(f"        finding: {finding[:96]}")
            if len(result.findings_removed) > 6:
                print(f"        … +{len(result.findings_removed) - 6} more")

    changed = [r for r in results if r.changed]
    saved = sum(r.chars_saved for r in changed)
    print()
    print(f"Reports scanned : {len(results)}")
    print(f"Reports changed : {len(changed)}")
    print(f"Sections removed: {sum(len(r.sections_removed) for r in changed)}")
    print(f"Findings removed: {sum(len(r.findings_removed) for r in changed)}")
    print(f"Bullets removed : {sum(r.bullets_removed for r in changed)}")
    print(f"Chars saved     : {saved:,} (~{saved // 4:,} tokens at 4 chars/token)")
    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to modify the reports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
