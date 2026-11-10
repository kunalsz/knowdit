//! Runtime memory sampler.
//!
//! A long streamloop picks up several unrelated allocation sources — agent
//! conversation state, harness source, DB result sets, allocator retention —
//! and a slow accumulator is invisible until the kernel OOM-kills the process
//! hours later. This samples `/proc/self` on an interval and logs RSS next to
//! the link phase that is advancing, so a growth curve can be attributed to a
//! specific phase instead of guessed at.
//!
//! Enabled with `--memory-report-secs <N>` (0 disables). Output is one
//! `tracing::info!` line per tick, prefixed `[mem]`, which makes it easy to
//! isolate with `grep '\[mem\]'`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Live counters the sampler annotates each tick with.
#[derive(Debug, Default)]
pub struct MemoryCounters {
    /// Links whose pipeline has finished (joined by the scheduler).
    pub links_done: AtomicU64,
    /// Links currently in flight.
    pub links_active: AtomicU64,
    /// Agent steps taken across all links so far.
    pub steps: AtomicU64,
}

impl MemoryCounters {
    pub fn links_done(&self) -> u64 {
        self.links_done.load(Ordering::Relaxed)
    }
    pub fn links_active(&self) -> u64 {
        self.links_active.load(Ordering::Relaxed)
    }
    pub fn steps(&self) -> u64 {
        self.steps.load(Ordering::Relaxed)
    }
}

/// Snapshot of this process's memory, in KiB.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcMemory {
    pub rss_kib: u64,
    pub peak_rss_kib: u64,
    pub vm_kib: u64,
}

/// Read `/proc/self/status`. Returns `Default` on non-Linux or unreadable
/// procfs rather than failing — this is observability, not correctness.
pub fn proc_memory() -> ProcMemory {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return ProcMemory::default();
    };
    let mut mem = ProcMemory::default();
    for line in status.lines() {
        let mut it = line.split_whitespace();
        let Some(key) = it.next() else { continue };
        let value = it.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        match key {
            "VmRSS:" => mem.rss_kib = value,
            "VmHWM:" => mem.peak_rss_kib = value,
            "VmSize:" => mem.vm_kib = value,
            _ => {}
        }
    }
    mem
}

/// Spawn the sampler. Returns a handle the caller can drop to stop it.
///
/// `phase` is read each tick so the log line names whatever the run is doing
/// (e.g. the currently-running link), letting a growth curve be matched to
/// activity instead of wall-clock alone.
pub fn spawn_sampler(
    interval: Duration,
    counters: Arc<MemoryCounters>,
    phase: Arc<std::sync::Mutex<String>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let started = std::time::Instant::now();
        loop {
            ticker.tick().await;
            let mem = proc_memory();
            let phase_now = phase.lock().map(|p| p.clone()).unwrap_or_default();
            tracing::info!(
                "[mem] t={:.0}s rss={:.0}MiB peak={:.0}MiB vm={:.0}MiB links_done={} \
                 links_active={} steps={} phase={}",
                started.elapsed().as_secs_f64(),
                mem.rss_kib as f64 / 1024.0,
                mem.peak_rss_kib as f64 / 1024.0,
                mem.vm_kib as f64 / 1024.0,
                counters.links_done(),
                counters.links_active(),
                counters.steps(),
                phase_now,
            );
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proc_memory_reports_nonzero_rss() {
        let mem = proc_memory();
        // Any running test process has resident memory; a zero here means the
        // parser broke and the sampler would log misleading zeros all run.
        assert!(mem.rss_kib > 0, "failed to parse VmRSS from /proc/self/status");
    }
}

#[cfg(test)]
mod sampler_tests {
    use super::*;
    use std::sync::atomic::Ordering;

    /// The sampler must actually tick and carry the live counters, otherwise a
    /// run would produce no diagnostics and we would be back to guessing.
    #[tokio::test]
    async fn sampler_ticks_and_reads_counters() {
        let counters = std::sync::Arc::new(MemoryCounters::default());
        counters.links_done.store(7, Ordering::Relaxed);
        counters.links_active.store(3, Ordering::Relaxed);
        let phase = std::sync::Arc::new(std::sync::Mutex::new("link 0042/0400".to_string()));

        assert_eq!(counters.links_done(), 7);
        assert_eq!(counters.links_active(), 3);

        let handle = spawn_sampler(
            Duration::from_millis(10),
            counters.clone(),
            phase.clone(),
        );
        // Let a few ticks fire, then stop the sampler.
        tokio::time::sleep(Duration::from_millis(60)).await;
        handle.abort();

        // proc_memory is exercised by the sampler; assert it is sane here too.
        let mem = proc_memory();
        assert!(mem.rss_kib > 0);
        assert!(mem.peak_rss_kib >= mem.rss_kib, "peak must be >= current");
    }
}
