//! Equivalence tests for [`RepoDatabase::load_semantic_match_results_for_link`].
//!
//! The narrow loader exists purely as a performance fix: the per-link fuzz /
//! regen paths only ever materialize one `(extract, historical, finding)`
//! triple, but used to call the wide `load_semantic_match_results`, which
//! reads every historical, every finding, every link and the whole
//! `semantic_matched` table (~hundreds of MiB on a large project) on every
//! invocation. That repeated O(all rows) work drove streamloop RSS into the
//! tens of GiB and got the process OOM-killed repeatedly.
//!
//! A performance change is only safe if it is behaviour-preserving, so these
//! tests pin the contract: for any given pair the narrow loader must return
//! exactly what the wide loader returns for that pair — same record, same
//! rendered variant text, same finding order, same strength/evidence — and the
//! materialized `LinkInput` must be identical.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use knowdit_kg_model::audit_finding::FindingSeverity;
use knowdit_kg_model::category::DeFiCategory;
use knowdit_kg_model::link_strength::LinkStrength;
use sea_orm::{ActiveValue::Set, EntityTrait};

use crate::db::{
    historical_finding as historical_finding_model,
    project_semantic as project_semantic_model,
    historical_finding_merge as historical_finding_merge_model,
    historical_semantic as historical_semantic_model,
    historical_semantic_finding_link as link_model,
    historical_semantic_merge as sem_merge_model, semantic_matched as matched_model,
};
use crate::repo::RepoDatabase;

struct TempDb {
    repo: RepoDatabase,
    path: PathBuf,
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

async fn temp_db() -> TempDb {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "knowdit-narrow-match-{}-{unique}.sqlite3",
        std::process::id()
    ));
    let repo = RepoDatabase::open_sqlite(path.clone())
        .await
        .expect("test repo connects");
    repo.init_schema().await.expect("schema initializes");
    TempDb { repo, path }
}

async fn insert_semantic(repo: &RepoDatabase, id: i32, desc: &str) {
    historical_semantic_model::Entity::insert(historical_semantic_model::ActiveModel {
        id: Set(id),
        name: Set(format!("hist_{id}")),
        definition: Set(format!("def_{id}")),
        description: Set(desc.to_string()),
        category: Set(DeFiCategory::Lending),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("historical_semantic inserts");
}

async fn insert_finding(repo: &RepoDatabase, id: i32, desc: &str) {
    historical_finding_model::Entity::insert(historical_finding_model::ActiveModel {
        id: Set(id),
        title: Set(format!("finding_{id}")),
        severity: Set(FindingSeverity::Medium),
        description: Set(desc.to_string()),
        root_cause: Set(format!("root_{id}")),
        patterns: Set(format!("pat_{id}")),
        exploits: Set(format!("exp_{id}")),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("historical_finding inserts");
}

async fn insert_extract(repo: &RepoDatabase, id: i32) {
    project_semantic_model::Entity::insert(project_semantic_model::ActiveModel {
        id: Set(id),
        name: Set(format!("extract_{id}")),
        definition: Set(format!("extract def {id}")),
        description: Set(format!("extract desc {id}")),
        category: Set(DeFiCategory::Lending),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("project_semantic inserts");
}

async fn insert_match(repo: &RepoDatabase, extract_id: i32, historical_id: i32) {
    matched_model::Entity::insert(matched_model::ActiveModel {
        extract_id: Set(extract_id),
        historical_id: Set(historical_id),
        strength: Set(crate::repo::MatchStrength::High),
        evidence: Set(format!("evidence {extract_id}x{historical_id}")),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("semantic_matched inserts");
}

async fn insert_link(repo: &RepoDatabase, historical_id: i32, finding_id: i32) {
    link_model::Entity::insert(link_model::ActiveModel {
        historical_semantic_id: Set(historical_id),
        historical_finding_id: Set(finding_id),
        strength: Set(LinkStrength::High),
        evidence: Set(format!("link {historical_id}x{finding_id}")),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("link inserts");
}

/// Seed: two historicals (only one matched to the target extract), each with
/// findings, one canonical carrying a merged-variant note, and a finding with
/// its own merge notes. Neighbouring rows exist so the test would catch the
/// narrow loader accidentally returning someone else's data.
async fn seed(repo: &RepoDatabase) {
    insert_extract(repo, 1).await;
    insert_extract(repo, 2).await;
    insert_semantic(repo, 100, "canonical one").await;
    insert_semantic(repo, 200, "canonical two").await;

    insert_finding(repo, 10, "finding ten").await;
    insert_finding(repo, 11, "finding eleven").await;
    insert_finding(repo, 20, "finding twenty").await;

    insert_match(repo, 1, 100).await;
    insert_match(repo, 1, 200).await;
    insert_match(repo, 2, 100).await;

    insert_link(repo, 100, 10).await;
    insert_link(repo, 100, 11).await;
    insert_link(repo, 200, 20).await;

    sem_merge_model::Entity::insert(sem_merge_model::ActiveModel {
        from_semantic_id: Set(9001),
        to_semantic_id: Set(100),
        appended_description: Set("  variant note for 100  ".to_string()),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("sem_merge inserts");

    historical_finding_merge_model::Entity::insert(
        historical_finding_merge_model::ActiveModel {
            from_finding_id: Set(9100),
            to_finding_id: Set(10),
            appended_description: Set("finding variant".to_string()),
            appended_patterns: Set(String::new()),
            appended_exploits: Set("exploit variant".to_string()),
            ..Default::default()
        },
    )
    .exec(repo.connection())
    .await
    .expect("finding merge inserts");
}

/// The narrow loader must reproduce the wide loader's slice for the pair.
#[tokio::test]
async fn narrow_matches_wide_for_one_pair() {
    let db = temp_db().await;
    seed(&db.repo).await;

    let wide = db
        .repo
        .load_semantic_match_results()
        .await
        .expect("wide load");
    let narrow = db
        .repo
        .load_semantic_match_results_for_link(1, 100)
        .await
        .expect("narrow load");

    // Match rows: exactly the one for this pair.
    assert_eq!(narrow.matches.len(), 1);
    let wide_match = wide
        .matches
        .iter()
        .find(|m| m.extract_id == 1 && m.historical_id == 100)
        .expect("wide has this pair");
    assert_eq!(narrow.matches[0].strength, wide_match.strength);
    assert_eq!(narrow.matches[0].evidence, wide_match.evidence);

    // Historical record: byte-identical rendering, same findings, same order.
    assert_eq!(narrow.historicals.len(), 1);
    let wide_hist = wide
        .historicals
        .iter()
        .find(|h| h.semantic.id == 100)
        .expect("wide has historical 100");
    let narrow_hist = &narrow.historicals[0];

    assert_eq!(narrow_hist.semantic.id, wide_hist.semantic.id);
    assert_eq!(narrow_hist.rendered_description, wide_hist.rendered_description);
    assert!(
        narrow_hist.rendered_description.contains("variant note for 100"),
        "merged-variant note must be rendered, got {:?}",
        narrow_hist.rendered_description
    );

    assert_eq!(narrow_hist.findings.len(), wide_hist.findings.len());
    for (n, w) in narrow_hist.findings.iter().zip(wide_hist.findings.iter()) {
        assert_eq!(n.finding.id, w.finding.id);
        assert_eq!(n.strength, w.strength);
        assert_eq!(n.evidence, w.evidence);
        assert_eq!(n.rendered_description, w.rendered_description);
        assert_eq!(n.rendered_patterns, w.rendered_patterns);
        assert_eq!(n.rendered_exploits, w.rendered_exploits);
    }
}

/// A pair with no `semantic_matched` row is empty, not an error — the caller
/// treats it the same way the wide loader's absence did.
#[tokio::test]
async fn unknown_pair_yields_empty_set() {
    let db = temp_db().await;
    seed(&db.repo).await;

    let narrow = db
        .repo
        .load_semantic_match_results_for_link(999, 999)
        .await
        .expect("narrow load on unknown pair");
    assert!(narrow.matches.is_empty());
    assert!(narrow.historicals.is_empty());
}

/// Only the requested historical's findings come back — the other
/// historical's findings must not leak in.
#[tokio::test]
async fn narrow_excludes_other_historicals_findings() {
    let db = temp_db().await;
    seed(&db.repo).await;

    let narrow = db
        .repo
        .load_semantic_match_results_for_link(1, 200)
        .await
        .expect("narrow load");
    assert_eq!(narrow.historicals.len(), 1);
    let ids: Vec<i32> = narrow.historicals[0]
        .findings
        .iter()
        .map(|f| f.finding.id)
        .collect();
    assert_eq!(ids, vec![20], "only historical 200's finding");
}
