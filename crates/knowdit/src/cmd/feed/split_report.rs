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
        let findings = split_findings(&content);

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
}

/// Split markdown content into per-finding sections.
///
/// Heuristic finding boundaries:
/// - Lines matching `# [X-XX] ...` or `## [X-XX] ...` (audit report finding headers)
/// - Lines matching `## Severity` immediately after a finding body (Shieldify format)
///
/// The first boundary (or document start) captures preamble text which is discarded.
fn split_findings(content: &str) -> Vec<String> {
    let lines: Vec<&str> = content.lines().collect();
    let mut boundaries: Vec<usize> = Vec::new();

    // Lines matching "# [I-01]", "## [M-02]", "# [H-03]", etc.
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        // Match "# [X-XX]" or "## [X-XX]" patterns (audit finding headers)
        if (trimmed.starts_with("# [") || trimmed.starts_with("## ["))
            && trimmed.len() >= 5
            && trimmed.as_bytes().get(3) == Some(&b'-')
        {
            boundaries.push(i);
        }
    }

    // If no numbered headings found, try "## Severity" boundaries (Shieldify format:
    // each finding has a "## Severity" section after the title)
    if boundaries.is_empty() {
        for (i, line) in lines.iter().enumerate() {
            if line.trim() == "## Severity" {
                // Walk backwards to find the finding title (preceding ## header).
                // The finding starts at the title line.
                let mut start = i;
                for j in (0..i).rev() {
                    let prev = lines[j].trim();
                    if prev.starts_with("## ") || prev.starts_with("# ") {
                        start = j;
                        break;
                    }
                }
                boundaries.push(start);
            }
        }
    }

    if boundaries.is_empty() {
        return vec![content.to_string()];
    }

    // Extract each finding section: from boundary[i] up to boundary[i+1] (or end).
    let mut results = Vec::new();
    for w in boundaries.windows(2) {
        let start = w[0];
        let end = w[1];
        let finding_lines = &lines[start..end];
        results.push(finding_lines.join("\n"));
    }
    // Last finding: from last boundary to end.
    let remaining = &lines[*boundaries.last().unwrap()..];
    results.push(remaining.join("\n"));

    results
}
