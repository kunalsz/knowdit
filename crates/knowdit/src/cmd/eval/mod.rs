//! `knowdit eval ...` — benchmarking and evaluation harness for the
//! learning pipeline. Suite verification, baseline freezing, replay
//! runs, scoring, comparison, and promotion gates all live here.
//!
//! The harness never mutates the production KG: it imports an
//! immutable baseline snapshot into fresh temporary SQLite sandboxes
//! and drives the production learn stages against those.

pub mod baseline;
pub mod compare;
pub mod corpus;
pub mod inspect;
pub mod run;
pub mod score;
