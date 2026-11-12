use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

/// Explicit agent-finalized abandon record for an `(extract_id, historical_id, finding_id)`
/// link triple. Written **only** when the spec-gen agent explicitly invokes
/// `finalize(status="abandoned")` — never on transient step, timeout, or provider
/// errors — so resume skips evaluated dead-end links while retrying transient errors.
#[sea_orm::model]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, DeriveEntityModel)]
#[sea_orm(table_name = "abandoned_link")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
    #[sea_orm(indexed)]
    pub extract_id: i32,
    #[sea_orm(indexed)]
    pub historical_id: i32,
    #[sea_orm(indexed)]
    pub finding_id: i32,
    #[sea_orm(column_type = "Text")]
    pub abort_reason: String,
}

impl ActiveModelBehavior for ActiveModel {}
