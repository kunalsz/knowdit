//! Error type for the evaluation harness.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("database error: {0}")]
    Db(#[from] sea_orm::DbErr),

    #[error("kg error: {0}")]
    Kg(#[from] knowdit_kg::error::KgError),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("manifest error: {0}")]
    Manifest(String),

    #[error("corpus error: {0}")]
    Corpus(String),

    #[error("sandbox error: {0}")]
    Sandbox(String),

    #[error("score error: {0}")]
    Score(String),

    #[error("compare error: {0}")]
    Compare(String),

    #[error("gate failure: {0}")]
    Gate(String),

    #[error("{0}")]
    Other(#[from] color_eyre::Report),
}

pub type Result<T> = std::result::Result<T, EvalError>;

impl EvalError {
    pub fn manifest(msg: impl Into<String>) -> Self {
        Self::Manifest(msg.into())
    }

    pub fn corpus(msg: impl Into<String>) -> Self {
        Self::Corpus(msg.into())
    }

    pub fn sandbox(msg: impl Into<String>) -> Self {
        Self::Sandbox(msg.into())
    }

    pub fn score(msg: impl Into<String>) -> Self {
        Self::Score(msg.into())
    }

    pub fn compare(msg: impl Into<String>) -> Self {
        Self::Compare(msg.into())
    }

    pub fn gate(msg: impl Into<String>) -> Self {
        Self::Gate(msg.into())
    }
}
