use sea_orm::entity::prelude::*;

/// Durable source identity for a Markdown report ingested by `feed reports`.
/// Legacy `project_platform` rows remain unchanged; this table maps a stable
/// namespace/path identity to the already-learned project.
#[sea_orm::model]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, DeriveEntityModel)]
#[sea_orm(table_name = "feed_report_source")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub project_id: i32,
    pub source_namespace: String,
    pub relative_path: String,
    #[sea_orm(unique)]
    pub stable_source_id: String,
    pub legacy_platform_id: Option<String>,
    pub content_hash: String,
    pub active: bool,
    pub last_seen_unix: i64,

    #[sea_orm(belongs_to, from = "project_id", to = "id")]
    pub project: HasOne<super::project::Entity>,
}

impl ActiveModelBehavior for ActiveModel {}
