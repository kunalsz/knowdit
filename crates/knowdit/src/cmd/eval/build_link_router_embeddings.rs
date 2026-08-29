use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::embeddings::{CreateEmbeddingRequestArgs, EmbeddingInput};
use clap::{Args, ValueEnum};
use color_eyre::eyre::{Context, Result, ensure};
use knowdit_kg::db::HistoricalDatabase;
use knowdit_kg::router_eval::{
    ROUTER_EMBEDDING_CACHE_VERSION, RouterEmbeddingCache, RouterEmbeddingDocument,
    RouterEmbeddingRecord,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Args, Clone)]
pub struct BuildLinkRouterEmbeddingsArgs {
    /// Embedding backend. Local uses BAAI/bge-small-en-v1.5 through Python fastembed.
    #[arg(long, value_enum, default_value_t = EmbeddingBackend::Openai)]
    pub backend: EmbeddingBackend,

    /// OpenAI-compatible API key. Read from OPENAI_API_KEY by default.
    #[arg(long, env = "OPENAI_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,

    /// OpenAI-compatible API base URL.
    #[arg(
        long,
        env = "OPENAI_BASE_URL",
        default_value = "https://api.openai.com/v1"
    )]
    pub api_base: String,

    /// Embedding model name accepted by the configured provider.
    #[arg(long, default_value = "text-embedding-3-small")]
    pub model: String,

    /// Requested vector dimensions. Supported by text-embedding-3 models.
    #[arg(long, default_value_t = 512)]
    pub dimensions: u32,

    /// Documents sent per embedding request.
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,

    /// Max merged raw variants represented per canonical.
    #[arg(long, default_value_t = 8)]
    pub variant_render_cap: usize,

    /// Max characters represented from each merged raw description.
    #[arg(long, default_value_t = 400)]
    pub raw_child_char_cap: usize,

    /// Cache path. Existing matching records are reused by fingerprint.
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum EmbeddingBackend {
    Local,
    Openai,
}

impl BuildLinkRouterEmbeddingsArgs {
    pub async fn run(self, db: &HistoricalDatabase) -> Result<()> {
        ensure!(self.dimensions > 0, "dimensions must be greater than zero");
        ensure!(self.batch_size > 0, "batch-size must be greater than zero");

        let documents = db
            .link_router_embedding_documents(self.variant_render_cap, self.raw_child_char_cap)
            .await?;
        let cache_model = if matches!(self.backend, EmbeddingBackend::Local) {
            "BAAI/bge-small-en-v1.5"
        } else {
            &self.model
        };
        let mut cache = load_compatible_cache(&self.output, cache_model, self.dimensions as usize)?;
        let existing = cache
            .records
            .iter()
            .map(|record| ((record.kind, record.id), record.fingerprint.clone()))
            .collect::<HashMap<_, _>>();
        let pending = documents
            .into_iter()
            .filter(|document| {
                existing.get(&(document.kind, document.id)) != Some(&document.fingerprint)
            })
            .collect::<Vec<_>>();

        if pending.is_empty() {
            println!(
                "embedding cache is current: {} records at {} dimensions",
                cache.records.len(),
                cache.dimensions
            );
            return Ok(());
        }

        if matches!(self.backend, EmbeddingBackend::Local) {
            ensure!(
                self.dimensions == 384,
                "local BAAI/bge-small-en-v1.5 embeddings have 384 dimensions; use --dimensions 384"
            );
            let temporary = self.output.with_file_name(format!(
                ".{}.documents.tmp.json",
                self.output
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("router-embeddings")
            ));
            if let Some(parent) = temporary.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&temporary, serde_json::to_vec(&pending)?)?;
            let status = std::process::Command::new("python3")
                .args([
                    "scripts/build_link_router_embeddings.py",
                    "--input",
                    temporary.to_str().unwrap_or_default(),
                    "--output",
                    self.output.to_str().unwrap_or_default(),
                    "--batch-size",
                    &self.batch_size.to_string(),
                ])
                .status()
                .wrap_err("failed to start local embedding worker; install `fastembed` with pip")?;
            let _ = std::fs::remove_file(&temporary);
            ensure!(
                status.success(),
                "local embedding worker failed with {status}"
            );
            return Ok(());
        }

        let api_key = self
            .api_key
            .filter(|key| !key.trim().is_empty())
            .ok_or_else(|| {
                color_eyre::eyre::eyre!("OPENAI_API_KEY is required for --backend openai")
            })?;

        let config = OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(self.api_base);
        let client = Client::with_config(config);
        let mut total_prompt_tokens = 0u64;
        for (batch_index, batch) in pending.chunks(self.batch_size).enumerate() {
            let input = batch
                .iter()
                .map(|document| embedding_input(document))
                .collect::<Vec<_>>();
            let request = CreateEmbeddingRequestArgs::default()
                .model(&self.model)
                .dimensions(self.dimensions)
                .input(EmbeddingInput::StringArray(input))
                .build()?;
            let mut response = client.embeddings().create(request).await?;
            response.data.sort_by_key(|embedding| embedding.index);
            ensure!(
                response.data.len() == batch.len(),
                "embedding provider returned {} vectors for {} inputs",
                response.data.len(),
                batch.len()
            );
            total_prompt_tokens += u64::from(response.usage.prompt_tokens);

            let mut by_key = cache
                .records
                .drain(..)
                .map(|record| ((record.kind, record.id), record))
                .collect::<HashMap<_, _>>();
            for (document, embedding) in batch.iter().zip(response.data) {
                ensure!(
                    embedding.embedding.len() == self.dimensions as usize,
                    "embedding provider returned {} dimensions; expected {}",
                    embedding.embedding.len(),
                    self.dimensions
                );
                by_key.insert(
                    (document.kind, document.id),
                    RouterEmbeddingRecord {
                        kind: document.kind,
                        id: document.id,
                        fingerprint: document.fingerprint.clone(),
                        vector: embedding.embedding,
                    },
                );
            }
            cache.records = by_key.into_values().collect();
            cache.records.sort_by_key(|record| (record.kind, record.id));
            save_cache_atomic(&self.output, &cache)?;
            println!(
                "embedded batch {}/{}; cache now has {} records",
                batch_index + 1,
                pending.len().div_ceil(self.batch_size),
                cache.records.len()
            );
        }
        println!(
            "embedding cache complete: {} records, {} dimensions, {} input tokens",
            cache.records.len(),
            cache.dimensions,
            total_prompt_tokens
        );
        Ok(())
    }
}

fn embedding_input(document: &RouterEmbeddingDocument) -> String {
    let text = document.text.trim();
    if text.is_empty() {
        format!("{:?} {}", document.kind, document.id)
    } else {
        text.to_string()
    }
}

fn load_compatible_cache(
    path: &Path,
    model: &str,
    dimensions: usize,
) -> Result<RouterEmbeddingCache> {
    if !path.exists() {
        return Ok(RouterEmbeddingCache {
            schema_version: ROUTER_EMBEDDING_CACHE_VERSION,
            model: model.to_string(),
            dimensions,
            records: Vec::new(),
        });
    }
    let cache: RouterEmbeddingCache = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(
        cache.schema_version == ROUTER_EMBEDDING_CACHE_VERSION,
        "cache schema version mismatch"
    );
    ensure!(cache.model == model, "cache model mismatch");
    ensure!(cache.dimensions == dimensions, "cache dimensions mismatch");
    Ok(cache)
}

fn save_cache_atomic(path: &Path, cache: &RouterEmbeddingCache) -> Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("router-embeddings.json");
    let temporary = path.with_file_name(format!(".{file_name}.tmp"));
    let bytes = serde_json::to_vec(cache)?;
    std::fs::write(&temporary, bytes)
        .wrap_err_with(|| format!("failed to write {}", temporary.display()))?;
    std::fs::rename(&temporary, path)
        .wrap_err_with(|| format!("failed to replace {}", path.display()))?;
    Ok(())
}
