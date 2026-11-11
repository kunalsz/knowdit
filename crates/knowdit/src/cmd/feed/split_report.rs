use std::path::PathBuf;

use clap::Args;
use color_eyre::eyre::Result;

/// Split a multi-finding audit report (.md) into individual per-finding
/// .md files. Parses "## Severity" or "# [X-XX]" section headers as
/// finding boundaries. The output files are written to `--out-dir`
/// as `finding-NN.md` and are ready for `knowdit feed reports`.
#[derive(Args)]
pub struct SplitReportArgs {
    /// Path to the input audit report .md file
    #[arg(long)]
    pub input: PathBuf,

    /// Output directory for per-finding .md files
    #[arg(long)]
    pub out_dir: PathBuf,
}

impl SplitReportArgs {
    pub async fn run(self) -> Result<()> {
        let input = self.input.canonicalize()?;
        if !input.is_file() {
            return Err(color_eyre::eyre::eyre!("not a file: {}", input.display()));
        }

        let content = tokio::fs::read_to_string(&input).await?;
        let findings = Self::split_findings(&content);

        if findings.is_empty() {
            tracing::warn!("No findings detected in {}", input.display());
            return Ok(());
        }

        tokio::fs::create_dir_all(&self.out_dir).await?;

        for (i, finding) in findings.iter().enumerate() {
            let out_path = self.out_dir.join(format!("finding-{:02}.md", i + 1));
            tokio::fs::write(&out_path, finding).await?;
            tracing::info!("Wrote {}", out_path.display());
        }

        tracing::info!(
            "Split {} into {} finding(s) under {}",
            input.display(),
            findings.len(),
            self.out_dir.display()
        );

        Ok(())
    }

    /// Split markdown content into per-finding sections while ignoring headings
    /// inside fenced code blocks (` ``` ` or `~~~`).
    ///
    /// Heuristic finding boundaries:
    /// - Markdown headings (`#`..`####`) starting with a bracketed finding ID
    ///   such as `# [H-01] ...`, `## [M-02] ...`, `### [C-1] ...`, or
    ///   `## [OS-SOL-01] ...` (excluding markdown links `[label](url)`).
    /// - Fallback when no bracketed IDs exist: `## Severity` / `### Severity`
    ///   sections, walking backwards to the preceding finding title heading.
    ///
    /// Preamble text before the first boundary is discarded.
    pub fn split_findings(content: &str) -> Vec<String> {
        let lines: Vec<&str> = content.lines().collect();
        let mut fence_marker: Option<&str> = None;
        let mut non_fenced_lines = vec![false; lines.len()];

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            if let Some(active) = fence_marker {
                if trimmed.starts_with(active) {
                    fence_marker = None;
                }
                continue;
            }
            if trimmed.starts_with("```") {
                fence_marker = Some("```");
                continue;
            }
            if trimmed.starts_with("~~~") {
                fence_marker = Some("~~~");
                continue;
            }
            non_fenced_lines[i] = true;
        }

        let mut boundaries: Vec<usize> = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if non_fenced_lines[i] && Self::is_bracketed_finding_heading(line.trim()) {
                boundaries.push(i);
            }
        }

        // Fallback: "## Severity" / "### Severity" boundaries (Shieldify and
        // narrative audit formats where each finding has a Severity subsection
        // right under its title heading).
        if boundaries.is_empty() {
            for (i, line) in lines.iter().enumerate() {
                if !non_fenced_lines[i] {
                    continue;
                }
                let trimmed = line.trim();
                if trimmed == "## Severity" || trimmed == "### Severity" {
                    let mut start = i;
                    for j in (0..i).rev() {
                        if !non_fenced_lines[j] {
                            continue;
                        }
                        let prev = lines[j].trim();
                        if prev.starts_with('#') && prev.trim_start_matches('#').starts_with(' ') {
                            start = j;
                            break;
                        }
                    }
                    if boundaries.last().copied() != Some(start) {
                        boundaries.push(start);
                    }
                }
            }
        }

        let Some(&last_start) = boundaries.last() else {
            return vec![content.to_string()];
        };

        let mut results = Vec::with_capacity(boundaries.len());
        for window in boundaries.windows(2) {
            results.push(lines[window[0]..window[1]].join("\n"));
        }
        results.push(lines[last_start..].join("\n"));
        results
    }

    /// True when `trimmed` is a `#`..`####` heading beginning with a bracketed
    /// finding identifier like `[H-01]`, `[M-2]`, `[V-03]`, or `[OS-SOL-01]`,
    /// and NOT a markdown link `[text](url)`.
    fn is_bracketed_finding_heading(trimmed: &str) -> bool {
        let after_hashes = trimmed.trim_start_matches('#');
        let hash_count = trimmed.len() - after_hashes.len();
        if !(1..=4).contains(&hash_count) || !after_hashes.starts_with(' ') {
            return false;
        }
        let rest = after_hashes.trim_start();
        if !rest.starts_with('[') {
            return false;
        }
        let Some(close_idx) = rest.find(']') else {
            return false;
        };
        if rest[close_idx + 1..].starts_with('(') {
            return false;
        }
        let tag = &rest[1..close_idx];
        !tag.is_empty()
            && tag.len() <= 24
            && tag.contains('-')
            && tag.chars().any(|c| c.is_ascii_alphabetic())
            && tag.chars().any(|c| c.is_ascii_digit())
            && tag
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    }
}

#[cfg(test)]
mod tests {
    use super::SplitReportArgs;

    #[test]
    fn splits_h1_h2_h3_bracketed_finding_headers_and_skips_preamble_and_links() {
        let report = "\
# Solana Vault Security Review
## [Scope](https://github.com/example/solana-vault-1)
Preamble summary text.

# [H-01] Missing Signer Check on Withdraw
Body of H-01.
```rust
// # [H-99] comment inside fenced code block must not split
pub fn withdraw() {}
```
Trailing line of H-01.

## [M-02] Unvalidated PDA Bump Seed
Body of M-02.

### [OS-SOL-03] Stale Account State After CPI
Body of OS-SOL-03.";

        let sections = SplitReportArgs::split_findings(report);
        assert_eq!(sections.len(), 3);
        assert!(sections[0].starts_with("# [H-01] Missing Signer Check"));
        assert!(sections[0].contains("Trailing line of H-01."));
        assert!(sections[1].starts_with("## [M-02] Unvalidated PDA Bump Seed"));
        assert!(sections[2].starts_with("### [OS-SOL-03] Stale Account State After CPI"));
    }

    #[test]
    fn falls_back_to_severity_subsections_when_unnumbered() {
        let report = "\
# Audit Overview
Intro paragraph.

## Arbitrary CPI in Swap Router
## Severity
High
## Description
Target program is unchecked.

## Duplicate Mutable Token Accounts
## Severity
Medium
## Description
Self-transfer inflates balance.";

        let sections = SplitReportArgs::split_findings(report);
        assert_eq!(sections.len(), 2);
        assert!(sections[0].starts_with("## Arbitrary CPI in Swap Router"));
        assert!(sections[1].starts_with("## Duplicate Mutable Token Accounts"));
    }

    #[test]
    fn single_unstructured_attack_analysis_stays_intact() {
        let postmortem = "# Exploit Analysis\n\nAttacker drained liquidity via stale CPI reload.";
        let sections = SplitReportArgs::split_findings(postmortem);
        assert_eq!(sections, vec![postmortem.to_string()]);
    }
}
