use clap::Args;
use color_eyre::eyre::{Result, eyre};
use knowdit_kg::db::HistoricalDatabase;

#[derive(Args)]
pub struct ListSemanticsArgs {
    /// Project reference: auto-increment ID or platform_id (e.g. "c4-420")
    pub project: String,
}

impl ListSemanticsArgs {
    pub async fn run(self, db: &HistoricalDatabase) -> Result<()> {
        // Resolve project by ID or platform_id (e.g. "c4-420")
        let proj = if let Ok(id) = self.project.parse::<i32>() {
            db.get_project_by_id(id).await?
        } else {
            db.get_project_by_platform_id(&self.project).await?
        };
        let proj = proj.ok_or_else(|| eyre!("Project '{}' not found", self.project))?;

        let results = db.list_semantics_by_project(proj.id).await?;
        if results.is_empty() {
            println!("No semantics found for project '{}'", self.project);
        } else {
            println!("Semantics for project '{}':", self.project);
            println!("{}", "=".repeat(60));
            for (node, funcs) in &results {
                println!("\n[{}] {}", node.id, node.name);
                println!("  Definition: {}", node.definition);
                println!(
                    "  Description: {}",
                    truncate_utf8_safe(&node.description, 200)
                );
                if !funcs.is_empty() {
                    println!("  Functions:");
                    for f in funcs {
                        println!("    - {} ({})", f.function_name, f.contract_path);
                    }
                }
            }
            println!("\nTotal: {} semantics", results.len());
        }
        Ok(())
    }
}

/// UTF-8-safe prefix cut. Byte-slicing `&s[..min(len, max)]` panics when the
/// cut lands inside a multi-byte char — KG descriptions routinely carry
/// em-dashes / arrows / CJK.
fn truncate_utf8_safe(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
