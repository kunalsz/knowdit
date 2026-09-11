//! Tests for [`RepoDatabase::reset_for_regenerate`] — the FK-safe full
//! downstream reset used by `--gen-specs-regenerate`.
//!
//! The reset must clear specs **and** everything that references them
//! (code_gen → harness_run → line_coverage, reflection → valid_finding,
//! report rows) in child-before-parent order, so it cannot fail against
//! `PRAGMA foreign_keys=ON` no matter how much of the fuzz/reflect pipeline
//! ran. `clear_specifications` alone is deliberately not FK-safe, which this
//! suite pins down.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use knowdit_kg_model::audit_finding::FindingSeverity;
use knowdit_kg_model::category::DeFiCategory;
use sea_orm::{ActiveValue::Set, EntityTrait};

use crate::db::{
    code_gen as code_gen_model, harness_run as harness_run_model,
    historical_semantic as historical_semantic_model, line_coverage as line_coverage_model,
    project_semantic as project_semantic_model, reflection as reflection_model,
    specification as specification_model, valid_finding as valid_finding_model,
};
use crate::repo::{CodeGenStatus, ReflectionResult, RepoDatabase, RunKind};

struct TempDb {
    repo: RepoDatabase,
    path: PathBuf,
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_file(self.path.with_extension("sqlite3-shm"));
        let _ = std::fs::remove_file(self.path.with_extension("sqlite3-wal"));
    }
}

async fn temp_db() -> TempDb {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "knowdit-regenerate-test-{}-{unique}.sqlite3",
        std::process::id()
    ));
    let repo = RepoDatabase::open_sqlite(path.clone())
        .await
        .expect("test repo connects");
    repo.init_schema().await.expect("schema initializes");
    TempDb { repo, path }
}

async fn insert_extract(repo: &RepoDatabase, id: i32) {
    project_semantic_model::Entity::insert(project_semantic_model::ActiveModel {
        id: Set(id),
        name: Set(format!("extract_{id}")),
        category: Set(DeFiCategory::Lending),
        definition: Set(String::new()),
        description: Set(String::new()),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("project_semantic inserts");
}

async fn insert_historical(repo: &RepoDatabase, id: i32) {
    historical_semantic_model::Entity::insert(historical_semantic_model::ActiveModel {
        id: Set(id),
        name: Set(format!("hist_{id}")),
        definition: Set(String::new()),
        description: Set(String::new()),
        category: Set(DeFiCategory::Lending),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("historical_semantic inserts");
}

async fn insert_spec(repo: &RepoDatabase, extract_id: i32, historical_id: i32, finding_id: i32) -> i32 {
    specification_model::Entity::insert(specification_model::ActiveModel {
        semantic_id: Set(extract_id),
        historical_id: Set(historical_id),
        finding_id: Set(finding_id),
        specification: Set("{}".to_string()),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("specification inserts")
    .last_insert_id
}

async fn insert_code_gen(repo: &RepoDatabase, spec_id: i32) -> i32 {
    code_gen_model::Entity::insert(code_gen_model::ActiveModel {
        spec_id: Set(spec_id),
        harness_relative_path: Set(String::new()),
        harness_source: Set(String::new()),
        status: Set(CodeGenStatus::Completed),
        final_reason: Set(String::new()),
        agent_steps: Set(0),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("code_gen inserts")
    .last_insert_id
}

async fn insert_run(repo: &RepoDatabase, code_id: i32) -> i32 {
    harness_run_model::Entity::insert(harness_run_model::ActiveModel {
        code_id: Set(code_id),
        kind: Set(RunKind::Test),
        seed: Set(None),
        runs: Set(1),
        forge_args: Set("[]".to_string()),
        exit_code: Set(0),
        stdout: Set(String::new()),
        stderr: Set(String::new()),
        duration_ms: Set(0),
        violated: Set(false),
        sequence_json: Set(None),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("harness_run inserts")
    .last_insert_id
}

async fn insert_coverage(repo: &RepoDatabase, run_id: i32) {
    line_coverage_model::Entity::insert(line_coverage_model::ActiveModel {
        run_id: Set(run_id),
        relative_contract_path: Set("src/A.sol".to_string()),
        line_number: Set(1),
        hit_count: Set(1),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("line_coverage inserts");
}

async fn insert_reflection(repo: &RepoDatabase, run_id: i32, spec_id: i32) -> i32 {
    reflection_model::Entity::insert(reflection_model::ActiveModel {
        run_id: Set(run_id),
        spec_id: Set(spec_id),
        result: Set(ReflectionResult::ValidFinding),
        reason: Set(String::new()),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("reflection inserts")
    .last_insert_id
}

async fn insert_valid_finding(repo: &RepoDatabase, reflection_id: i32) {
    valid_finding_model::Entity::insert(valid_finding_model::ActiveModel {
        reflection_id: Set(reflection_id),
        severity: Set(FindingSeverity::Medium),
        severity_reason: Set(String::new()),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("valid_finding inserts");
}

async fn counts(repo: &RepoDatabase) -> (u64, u64, u64, u64, u64) {
    let specs = specification_model::Entity::find().all(repo.connection()).await.unwrap().len() as u64;
    let codegens = code_gen_model::Entity::find().all(repo.connection()).await.unwrap().len() as u64;
    let runs = harness_run_model::Entity::find().all(repo.connection()).await.unwrap().len() as u64;
    let cov = line_coverage_model::Entity::find().all(repo.connection()).await.unwrap().len() as u64;
    let reflections = reflection_model::Entity::find().all(repo.connection()).await.unwrap().len() as u64;
    (specs, codegens, runs, cov, reflections)
}

/// Populate the full spec → code_gen → harness_run → line_coverage and
/// reflection → valid_finding chains.
async fn seed_full_downstream(repo: &RepoDatabase) {
    insert_extract(repo, 1).await;
    insert_historical(repo, 100).await;
    let spec = insert_spec(repo, 1, 100, 7).await;
    let cg = insert_code_gen(repo, spec).await;
    let run = insert_run(repo, cg).await;
    insert_coverage(repo, run).await;
    let refl = insert_reflection(repo, run, spec).await;
    insert_valid_finding(repo, refl).await;
}

#[tokio::test]
async fn clear_specifications_alone_is_not_fk_safe() {
    let temp = temp_db().await;
    seed_full_downstream(&temp.repo).await;

    // A bare `DELETE FROM specification` violates the code_gen/reflection FKs
    // under `PRAGMA foreign_keys=ON`. This pins down *why* the reset exists.
    let res = temp.repo.clear_specifications().await;
    assert!(
        res.is_err(),
        "clear_specifications should fail while FK children exist (got {res:?})"
    );
    let (specs, ..) = counts(&temp.repo).await;
    assert_eq!(specs, 1, "failed delete left the spec in place");
}

#[tokio::test]
async fn reset_for_regenerate_clears_entire_downstream_safely() {
    let temp = temp_db().await;
    seed_full_downstream(&temp.repo).await;
    let before = counts(&temp.repo).await;
    assert_eq!(before, (1, 1, 1, 1, 1));

    temp.repo
        .reset_for_regenerate()
        .await
        .expect("FK-safe reset succeeds");

    let after = counts(&temp.repo).await;
    assert_eq!(
        after,
        (0, 0, 0, 0, 0),
        "specs + every FK-dependent table cleared"
    );

    // Upstream inputs are preserved.
    assert_eq!(
        project_semantic_model::Entity::find()
            .all(temp.repo.connection())
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        historical_semantic_model::Entity::find()
            .all(temp.repo.connection())
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn reset_for_regenerate_is_idempotent_on_empty_db() {
    let temp = temp_db().await;
    temp.repo
        .reset_for_regenerate()
        .await
        .expect("reset on empty db is a no-op");
    assert_eq!(counts(&temp.repo).await, (0, 0, 0, 0, 0));
}
