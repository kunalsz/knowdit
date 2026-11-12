//! Small shared helpers for the workflow orchestrators.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use color_eyre::eyre::{Result, WrapErr};
use serde::Serialize;

/// Process-lifetime nonce for [`write_json_atomic`] tmp-file names.
static TMP_NONCE: AtomicU64 = AtomicU64::new(0);

/// Serialize `value` as pretty JSON and write it to `final_path` atomically:
/// write to a sibling `*.tmp` file, then rename it over the target, so a crash
/// mid-write never leaves a half-written file behind. The parent directory must
/// already exist.
///
/// The tmp name carries a pid + per-call nonce: concurrent workers writing the
/// SAME final path (e.g. parallel `dump_valid_findings` calls for one link
/// summary) would otherwise share one `.tmp` and race — one worker's rename
/// yanking the file out from under another's `fs::write`, crashing it with
/// ENOENT.
pub(crate) fn write_json_atomic<T: Serialize>(final_path: &Path, value: &T) -> Result<()> {
    let body = serde_json::to_string_pretty(value)
        .wrap_err_with(|| format!("failed to serialize JSON for {}", final_path.display()))?;
    let mut tmp = final_path.as_os_str().to_owned();
    tmp.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        TMP_NONCE.fetch_add(1, Ordering::Relaxed)
    ));
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, body).wrap_err_with(|| format!("failed to write {}", tmp.display()))?;
    std::fs::rename(&tmp, final_path)
        .wrap_err_with(|| format!("failed to rename into {}", final_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::write_json_atomic;

    #[test]
    fn concurrent_atomic_writes_to_same_path_do_not_collide() {
        let dir = std::env::temp_dir().join(format!("knowdit-atomic-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("out.json");
        let handles: Vec<_> = (0..16)
            .map(|i| {
                let t = target.clone();
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        write_json_atomic(&t, &serde_json::json!({ "w": i })).unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(
            serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(&target).unwrap())
                .is_ok()
        );
        // No stray tmp files left behind.
        let leftovers = std::fs::read_dir(&dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".tmp")
            })
            .count();
        assert_eq!(leftovers, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
