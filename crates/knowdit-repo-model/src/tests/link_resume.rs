//! Tests for [`RepoDatabase::link_resume_state`].
//!
//! The method drives streamloop's DB-only resume policy. Each test
//! stands up a fresh on-disk SQLite, seeds the FK chain by hand, and
//! asserts the returned [`LinkResumeState`] for one
//! `(extract, historical, finding)` triple.
//!
//! State map (mirrors the doc-comment on `link_resume_state`):
//!
//! | rows in `specification` for (E, H, F) | rows in `code_gen` (for those specs) | state         |
//! |---------------------------------------|--------------------------------------|---------------|
//! | 0                                     | (n/a)                                | `NotStarted`  |
//! | ≥1                                    | 0                                    | `Partial`     |
//! | ≥1                                    | ≥1                                   | `Built`       |
//!
//! Each test covers ONE branch of the table above, plus a few scoping
//! invariants (the lookup is keyed on the full triple — sibling
//! historicals don't bleed into each other).

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use knowdit_kg_model::category::DeFiCategory;
use sea_orm::{ActiveValue::Set, EntityTrait};

use crate::db::{
    code_gen as code_gen_model, historical_semantic as historical_semantic_model,
    project_semantic as project_semantic_model, specification as specification_model,
};
use crate::link::LinkKey;
use crate::repo::{CodeGenStatus, LinkResumeState, RepoDatabase};

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
        .expect("system clock should be after unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "knowdit-link-resume-test-{}-{unique}.sqlite3",
        std::process::id()
    ));
    let repo = RepoDatabase::open_sqlite(path.clone())
        .await
        .expect("test repo database should connect");
    repo.init_schema().await.expect("schema should initialize");
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
    .expect("project_semantic row should insert");
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
    .expect("historical_semantic row should insert");
}

async fn insert_spec(
    repo: &RepoDatabase,
    extract_id: i32,
    historical_id: i32,
    finding_id: i32,
) -> i32 {
    let res = specification_model::Entity::insert(specification_model::ActiveModel {
        semantic_id: Set(extract_id),
        historical_id: Set(historical_id),
        finding_id: Set(finding_id),
        specification: Set("{}".to_string()),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("specification row should insert");
    res.last_insert_id
}

async fn insert_code_gen(repo: &RepoDatabase, spec_id: i32) -> i32 {
    let res = code_gen_model::Entity::insert(code_gen_model::ActiveModel {
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
    .expect("code_gen row should insert");
    res.last_insert_id
}

#[tokio::test]
async fn not_started_when_no_specs() {
    let temp = temp_db().await;
    insert_extract(&temp.repo, 1).await;
    insert_historical(&temp.repo, 100).await;
    let state = temp.repo.link_resume_state(1, 100, 99).await.unwrap();
    assert_eq!(state, LinkResumeState::NotStarted);
}

#[tokio::test]
async fn partial_when_specs_but_no_codegen() {
    let temp = temp_db().await;
    insert_extract(&temp.repo, 1).await;
    insert_historical(&temp.repo, 100).await;
    let s1 = insert_spec(&temp.repo, 1, 100, 7).await;
    let s2 = insert_spec(&temp.repo, 1, 100, 7).await;
    let state = temp.repo.link_resume_state(1, 100, 7).await.unwrap();
    // Both specs surface in the Partial payload, ordered by id ASC
    // (matches the loader's `.order_by_asc(Id)`).
    assert_eq!(
        state,
        LinkResumeState::Partial {
            spec_ids: vec![s1, s2]
        }
    );
}

#[tokio::test]
async fn partial_returns_only_unfuzzed_specs_and_built_when_all_specs_have_resumable_codegen() {
    let temp = temp_db().await;
    insert_extract(&temp.repo, 1).await;
    insert_historical(&temp.repo, 100).await;
    let s1 = insert_spec(&temp.repo, 1, 100, 7).await;
    let s2 = insert_spec(&temp.repo, 1, 100, 7).await;
    // When only s1 has a Completed code_gen (e.g. run interrupted mid-link),
    // s2 MUST remain in Partial so it is not silently orphaned on resume.
    insert_code_gen(&temp.repo, s1).await;
    let state = temp.repo.link_resume_state(1, 100, 7).await.unwrap();
    assert_eq!(state, LinkResumeState::Partial { spec_ids: vec![s2] });

    // Once s2 also finishes with a resumable status, the link flips to Built.
    insert_code_gen(&temp.repo, s2).await;
    let state_after = temp.repo.link_resume_state(1, 100, 7).await.unwrap();
    assert_eq!(state_after, LinkResumeState::Built);
}

#[tokio::test]
async fn scoped_by_extract_historical_and_finding() {
    let temp = temp_db().await;
    insert_extract(&temp.repo, 1).await;
    insert_extract(&temp.repo, 2).await;
    insert_historical(&temp.repo, 100).await;
    insert_historical(&temp.repo, 200).await;
    // (extract=1, historical=100, finding=7) has specs.
    insert_spec(&temp.repo, 1, 100, 7).await;

    // Different extract id.
    assert_eq!(
        temp.repo.link_resume_state(2, 100, 7).await.unwrap(),
        LinkResumeState::NotStarted,
        "different extract id must not see specs from other extracts"
    );
    // Different finding id.
    assert_eq!(
        temp.repo.link_resume_state(1, 100, 8).await.unwrap(),
        LinkResumeState::NotStarted,
        "different finding id must not see specs from other findings"
    );
    // Different historical id — this is the new isolation we get from
    // Tier 2's (E, H, F) keying; pre-Tier 2 this would have spuriously
    // returned Partial because the (E, F) pair had specs.
    assert_eq!(
        temp.repo.link_resume_state(1, 200, 7).await.unwrap(),
        LinkResumeState::NotStarted,
        "different historical id must not see specs from other historicals"
    );
}

#[tokio::test]
async fn sibling_historicals_get_independent_states() {
    let temp = temp_db().await;
    insert_extract(&temp.repo, 1).await;
    insert_historical(&temp.repo, 100).await;
    insert_historical(&temp.repo, 200).await;
    // The fresh-run intentional fan-out: extract 1 matched to two
    // historicals, both linking to finding 7, each emits its own
    // specs. After a Ctrl-C between gen-spec and first fuzz, both
    // sets sit in `specification` with NO code_gen yet.
    let h100_specs = vec![
        insert_spec(&temp.repo, 1, 100, 7).await,
        insert_spec(&temp.repo, 1, 100, 7).await,
    ];
    let h200_specs = vec![
        insert_spec(&temp.repo, 1, 200, 7).await,
        insert_spec(&temp.repo, 1, 200, 7).await,
    ];

    // Each sibling resumes its OWN spec ids — no cross-historical
    // pollution.
    assert_eq!(
        temp.repo.link_resume_state(1, 100, 7).await.unwrap(),
        LinkResumeState::Partial {
            spec_ids: h100_specs.clone()
        }
    );
    assert_eq!(
        temp.repo.link_resume_state(1, 200, 7).await.unwrap(),
        LinkResumeState::Partial {
            spec_ids: h200_specs.clone()
        }
    );

    // If ALL specs on the H=100 side get a resumable code_gen, ONLY that side
    // flips to Built; when only one of two has a code_gen, the remaining spec stays Partial.
    insert_code_gen(&temp.repo, h100_specs[0]).await;
    assert_eq!(
        temp.repo.link_resume_state(1, 100, 7).await.unwrap(),
        LinkResumeState::Partial {
            spec_ids: vec![h100_specs[1]]
        }
    );
    insert_code_gen(&temp.repo, h100_specs[1]).await;
    assert_eq!(
        temp.repo.link_resume_state(1, 100, 7).await.unwrap(),
        LinkResumeState::Built
    );
    assert_eq!(
        temp.repo.link_resume_state(1, 200, 7).await.unwrap(),
        LinkResumeState::Partial {
            spec_ids: h200_specs
        }
    );
}

#[tokio::test]
async fn codegen_on_unrelated_spec_does_not_count() {
    let temp = temp_db().await;
    insert_extract(&temp.repo, 1).await;
    insert_historical(&temp.repo, 100).await;
    // Spec for the link under test.
    insert_spec(&temp.repo, 1, 100, 7).await;
    // Spec for a different finding under same (E, H) — has a code_gen.
    let other = insert_spec(&temp.repo, 1, 100, 99).await;
    insert_code_gen(&temp.repo, other).await;

    // Our link (1, 100, 7) is still Partial: its OWN specs lack a code_gen.
    let state = temp.repo.link_resume_state(1, 100, 7).await.unwrap();
    match state {
        LinkResumeState::Partial { .. } => (),
        other => panic!("expected Partial, got {other:?}"),
    }
}

#[tokio::test]
async fn bulk_resume_returns_all_states_and_ascending_ids() {
    let temp = temp_db().await;
    insert_extract(&temp.repo, 1).await;
    insert_extract(&temp.repo, 2).await;
    insert_historical(&temp.repo, 100).await;
    insert_historical(&temp.repo, 200).await;

    // Partial: two specs, no code_gen — ids must come back ascending.
    let p1 = insert_spec(&temp.repo, 1, 100, 7).await;
    let p2 = insert_spec(&temp.repo, 1, 100, 7).await;
    // Built: one spec with a code_gen.
    let b1 = insert_spec(&temp.repo, 1, 200, 8).await;
    insert_code_gen(&temp.repo, b1).await;
    // NotStarted: (2, 100, 9) has no rows.

    let partial_key = LinkKey {
        extract_id: 1,
        historical_id: 100,
        finding_id: 7,
    };
    let built_key = LinkKey {
        extract_id: 1,
        historical_id: 200,
        finding_id: 8,
    };
    let fresh_key = LinkKey {
        extract_id: 2,
        historical_id: 100,
        finding_id: 9,
    };

    let map = temp
        .repo
        .load_link_resume_states(&[partial_key, built_key, fresh_key])
        .await
        .unwrap();
    assert_eq!(map.len(), 3);
    assert_eq!(
        map[&partial_key],
        LinkResumeState::Partial {
            spec_ids: vec![p1, p2]
        }
    );
    assert_eq!(map[&built_key], LinkResumeState::Built);
    assert_eq!(map[&fresh_key], LinkResumeState::NotStarted);
}

/// Sibling historicals must not bleed into each other through the bulk
/// loader, and the chunking path (>300 keys) must still be exact.
#[tokio::test]
async fn bulk_resume_chunks_above_bind_limit() {
    let temp = temp_db().await;
    insert_extract(&temp.repo, 1).await;
    insert_historical(&temp.repo, 100).await;
    insert_historical(&temp.repo, 200).await;
    let h100 = insert_spec(&temp.repo, 1, 100, 7).await;
    let h200 = insert_spec(&temp.repo, 1, 200, 7).await;

    let mut keys: Vec<LinkKey> = (0..700)
        .map(|i| LinkKey {
            extract_id: 1,
            historical_id: 100,
            finding_id: 1_000 + i,
        })
        .collect();
    keys.push(LinkKey {
        extract_id: 1,
        historical_id: 100,
        finding_id: 7,
    });
    keys.push(LinkKey {
        extract_id: 1,
        historical_id: 200,
        finding_id: 7,
    });

    let map = temp.repo.load_link_resume_states(&keys).await.unwrap();
    assert_eq!(map.len(), 702);
    assert_eq!(
        map[&LinkKey {
            extract_id: 1,
            historical_id: 100,
            finding_id: 7,
        }],
        LinkResumeState::Partial {
            spec_ids: vec![h100]
        }
    );
    assert_eq!(
        map[&LinkKey {
            extract_id: 1,
            historical_id: 200,
            finding_id: 7,
        }],
        LinkResumeState::Partial {
            spec_ids: vec![h200]
        }
    );
    assert_eq!(
        map[&LinkKey {
            extract_id: 1,
            historical_id: 100,
            finding_id: 1_000,
        }],
        LinkResumeState::NotStarted
    );
}

/// The bulk resume loader must project only key/id columns — never the JSON
/// spec body or the code_gen harness payload — so a resume read cannot
/// overfetch heavy columns.
#[test]
fn resume_queries_project_only_keys() {
    use sea_orm::{DatabaseBackend, QueryTrait};

    let spec_sql = crate::repo::specification_resume_query(vec![1], vec![2], vec![3])
        .build(DatabaseBackend::Sqlite)
        .sql;
    assert!(
        spec_sql.contains(r#""specification"."id""#),
        "selects id: {spec_sql}"
    );
    assert!(spec_sql.contains(r#""specification"."semantic_id""#));
    assert!(spec_sql.contains(r#""specification"."historical_id""#));
    assert!(spec_sql.contains(r#""specification"."finding_id""#));
    assert!(
        !spec_sql.contains(r#""specification"."specification""#),
        "must not select the JSON payload: {spec_sql}"
    );

    let cg_sql = crate::repo::codegen_resume_query(vec![1])
        .build(DatabaseBackend::Sqlite)
        .sql;
    assert!(
        cg_sql.contains(r#""code_gen"."spec_id""#),
        "selects spec_id: {cg_sql}"
    );
    assert!(
        !cg_sql.to_lowercase().contains("harness_source"),
        "must not select harness_source: {cg_sql}"
    );
    assert!(
        !cg_sql.to_lowercase().contains("harness_relative_path"),
        "must not select harness_relative_path: {cg_sql}"
    );
}

/// The compound `(semantic_id, historical_id, finding_id)` index must actually
/// be usable for the loader's triple filter (no full-table scan).
#[tokio::test]
async fn resume_query_uses_compound_specification_index() {
    use sea_orm::{FromQueryResult, Statement};

    let temp = temp_db().await;
    insert_extract(&temp.repo, 1).await;
    insert_historical(&temp.repo, 100).await;
    insert_spec(&temp.repo, 1, 100, 7).await;

    #[derive(FromQueryResult)]
    struct PlanRow {
        detail: String,
    }

    let plan = PlanRow::find_by_statement(Statement::from_string(
        sea_orm::DatabaseBackend::Sqlite,
        "EXPLAIN QUERY PLAN SELECT id, semantic_id, historical_id, finding_id \
         FROM specification WHERE semantic_id = 1 AND historical_id = 100 AND finding_id = 7"
            .to_string(),
    ))
    .all(temp.repo.connection())
    .await
    .expect("explain query plan runs");

    let joined = plan
        .into_iter()
        .map(|r| r.detail)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        joined.contains("ix_specification_extract_historical_finding"),
        "expected compound index in plan, got:\n{joined}"
    );
}

#[tokio::test]
async fn abandoned_link_idempotent_upsert_and_resume_priority() -> color_eyre::eyre::Result<()> {
    use sea_orm::EntityTrait;

    let temp = temp_db().await;
    // Re-running init_schema on an already-migrated DB must stay idempotent.
    temp.repo.init_schema().await?;

    insert_extract(&temp.repo, 1).await;
    insert_historical(&temp.repo, 100).await;

    let k_abandoned = LinkKey {
        extract_id: 1,
        historical_id: 100,
        finding_id: 10,
    };
    let k_partial = LinkKey {
        extract_id: 1,
        historical_id: 100,
        finding_id: 11,
    };
    let k_built = LinkKey {
        extract_id: 1,
        historical_id: 100,
        finding_id: 12,
    };
    let k_fresh = LinkKey {
        extract_id: 1,
        historical_id: 100,
        finding_id: 13,
    };

    // Record abandoned twice on the same triple: compound unique index + on_conflict
    // updates abort_reason in place rather than duplicating rows.
    temp.repo
        .record_abandoned_links(&[k_abandoned, k_partial, k_built], "first reason")
        .await?;
    temp.repo
        .record_abandoned_links(&[k_abandoned], "updated reason")
        .await?;

    let rows = crate::db::abandoned_link::Entity::find()
        .all(temp.repo.connection())
        .await?;
    assert_eq!(rows.len(), 3);
    let updated = rows
        .iter()
        .find(|r| r.finding_id == 10)
        .ok_or_else(|| color_eyre::eyre::eyre!("missing abandoned row for finding 10"))?;
    assert_eq!(updated.abort_reason, "updated reason");

    // Seed Partial and Built on top of existing abandoned rows to verify
    // Built > Partial > Abandoned > NotStarted priority.
    let p_spec = insert_spec(&temp.repo, 1, 100, 11).await;
    let b_spec = insert_spec(&temp.repo, 1, 100, 12).await;
    insert_code_gen(&temp.repo, b_spec).await;

    assert_eq!(
        temp.repo.link_resume_state(1, 100, 10).await?,
        LinkResumeState::Abandoned
    );
    assert_eq!(
        temp.repo.link_resume_state(1, 100, 11).await?,
        LinkResumeState::Partial {
            spec_ids: vec![p_spec]
        }
    );
    assert_eq!(
        temp.repo.link_resume_state(1, 100, 12).await?,
        LinkResumeState::Built
    );
    assert_eq!(
        temp.repo.link_resume_state(1, 100, 13).await?,
        LinkResumeState::NotStarted
    );

    let bulk = temp
        .repo
        .load_link_resume_states(&[k_abandoned, k_partial, k_built, k_fresh])
        .await?;
    assert_eq!(bulk.get(&k_abandoned), Some(&LinkResumeState::Abandoned));
    assert_eq!(
        bulk.get(&k_partial),
        Some(&LinkResumeState::Partial {
            spec_ids: vec![p_spec]
        })
    );
    assert_eq!(bulk.get(&k_built), Some(&LinkResumeState::Built));
    assert_eq!(bulk.get(&k_fresh), Some(&LinkResumeState::NotStarted));

    // Verify abandoned_link bulk query never projects abort_reason.
    use sea_orm::{DatabaseBackend, QueryTrait};
    let ab_sql = crate::repo::abandoned_link_resume_query(vec![1], vec![100], vec![10])
        .build(DatabaseBackend::Sqlite)
        .sql;
    assert!(
        !ab_sql.to_lowercase().contains("abort_reason"),
        "abandoned resume query must not overfetch abort_reason: {ab_sql}"
    );
    Ok(())
}

#[tokio::test]
async fn skipped_viability_and_terminal_specifications_behavior() -> color_eyre::eyre::Result<()> {
    assert!(CodeGenStatus::SkippedViability.counts_as_resumable_skip());
    assert!(CodeGenStatus::Completed.counts_as_resumable_skip());
    assert!(CodeGenStatus::Duplicate.counts_as_resumable_skip());
    assert!(!CodeGenStatus::Abandoned.counts_as_resumable_skip());

    let temp = temp_db().await;
    insert_extract(&temp.repo, 1).await;
    insert_historical(&temp.repo, 100).await;

    let s_completed = insert_spec(&temp.repo, 1, 100, 1).await;
    let s_valid = insert_spec(&temp.repo, 1, 100, 2).await;
    let s_dup = insert_spec(&temp.repo, 1, 100, 3).await;
    let s_skip = insert_spec(&temp.repo, 1, 100, 4).await;

    let _cg1 = insert_code_gen(&temp.repo, s_completed).await;
    let cg_valid = insert_code_gen(&temp.repo, s_valid).await;

    for (spec_id, status) in [
        (s_dup, CodeGenStatus::Duplicate),
        (s_skip, CodeGenStatus::SkippedViability),
    ] {
        code_gen_model::Entity::insert(code_gen_model::ActiveModel {
            spec_id: Set(spec_id),
            harness_relative_path: Set(String::new()),
            harness_source: Set(String::new()),
            status: Set(status),
            final_reason: Set("screened".to_string()),
            agent_steps: Set(0),
            ..Default::default()
        })
        .exec(temp.repo.connection())
        .await?;
    }

    let run = crate::db::harness_run::Entity::insert(crate::db::harness_run::ActiveModel {
        code_id: Set(cg_valid),
        kind: Set(crate::repo::RunKind::Test),
        seed: Set(None),
        runs: Set(1),
        forge_args: Set("[]".to_string()),
        exit_code: Set(0),
        stdout: Set(String::new()),
        stderr: Set(String::new()),
        duration_ms: Set(0),
        violated: Set(true),
        sequence_json: Set(None),
        ..Default::default()
    })
    .exec(temp.repo.connection())
    .await?
    .last_insert_id;
    crate::db::reflection::Entity::insert(crate::db::reflection::ActiveModel {
        run_id: Set(run),
        spec_id: Set(s_valid),
        result: Set(crate::repo::ReflectionResult::ValidFinding),
        reason: Set("confirmed".to_string()),
        ..Default::default()
    })
    .exec(temp.repo.connection())
    .await?;

    // Both Duplicate and SkippedViability count as resumable skips...
    let resumable = temp.repo.loaded_completed_code_gen_spec_ids().await?;
    assert!(resumable.contains(&s_completed));
    assert!(resumable.contains(&s_valid));
    assert!(resumable.contains(&s_dup));
    assert!(resumable.contains(&s_skip));

    // ...but load_terminal_specifications returns ONLY canonical Completed/StepsExhausted specs,
    // annotated with is_valid_finding.
    let terminals = temp.repo.load_terminal_specifications().await?;
    assert_eq!(terminals.len(), 2);
    assert_eq!(terminals[0].spec.id, s_completed);
    assert!(!terminals[0].is_valid_finding);
    assert_eq!(terminals[1].spec.id, s_valid);
    assert!(terminals[1].is_valid_finding);
    Ok(())
}

/// Specs whose only codegen ended `StepsExhausted` (secure function, nothing
/// found) are terminal for novelty purposes: otherwise every later link
/// re-fuzzes the identical spec.
#[tokio::test]
async fn terminal_specifications_include_steps_exhausted() -> color_eyre::eyre::Result<()> {
    let temp = temp_db().await;
    insert_extract(&temp.repo, 1).await;
    insert_historical(&temp.repo, 100).await;
    let s = insert_spec(&temp.repo, 1, 100, 7).await;
    code_gen_model::Entity::insert(code_gen_model::ActiveModel {
        spec_id: Set(s),
        harness_relative_path: Set(String::new()),
        harness_source: Set(String::new()),
        status: Set(CodeGenStatus::StepsExhausted),
        final_reason: Set(String::new()),
        agent_steps: Set(0),
        ..Default::default()
    })
    .exec(temp.repo.connection())
    .await?;
    let terminals = temp.repo.load_terminal_specifications().await?;
    assert_eq!(terminals.len(), 1);
    assert_eq!(terminals[0].spec.id, s);
    assert!(!terminals[0].is_valid_finding);
    Ok(())
}

#[tokio::test]
async fn append_project_semantics_preserves_existing_ids() -> color_eyre::eyre::Result<()> {
    use knowdit_kg_model::{ExtractedFunction, ExtractedSemantic};

    let temp = temp_db().await;
    let initial = vec![
        ExtractedSemantic {
            name: "Deposit Flow".to_string(),
            category: DeFiCategory::Lending,
            definition: "def1".to_string(),
            description: "desc1".to_string(),
            functions: vec![ExtractedFunction {
                name: "deposit".to_string(),
                contract: "Vault".to_string(),
                signature: Some("deposit(uint256)".to_string()),
            }],
        },
        ExtractedSemantic {
            name: "Withdraw Flow".to_string(),
            category: DeFiCategory::Lending,
            definition: "def2".to_string(),
            description: "desc2".to_string(),
            functions: vec![ExtractedFunction {
                name: "withdraw".to_string(),
                contract: "Vault".to_string(),
                signature: Some("withdraw(uint256)".to_string()),
            }],
        },
    ];
    temp.repo.replace_project_semantics(&initial).await?;

    let appended = vec![ExtractedSemantic {
        name: "Vault Stateful Entrypoints".to_string(),
        category: DeFiCategory::Lending,
        definition: "def3".to_string(),
        description: "desc3".to_string(),
        functions: vec![ExtractedFunction {
            name: "emergencySweep".to_string(),
            contract: "Vault".to_string(),
            signature: Some("emergencySweep(address)".to_string()),
        }],
    }];
    temp.repo.append_project_semantics(&appended).await?;

    let rows = project_semantic_model::Entity::find()
        .all(temp.repo.connection())
        .await?;
    let ids_and_names: Vec<(i32, String)> = rows.into_iter().map(|r| (r.id, r.name)).collect();
    assert_eq!(
        ids_and_names,
        vec![
            (1, "Deposit Flow".to_string()),
            (2, "Withdraw Flow".to_string()),
            (3, "Vault Stateful Entrypoints".to_string()),
        ]
    );
    let loaded = temp.repo.load_project_semantics().await?;
    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded[2].functions[0].name, "emergencySweep");
    Ok(())
}
