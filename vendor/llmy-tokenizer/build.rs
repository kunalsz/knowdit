use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Models JSON schema
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelTokens {
    content_multiplier: f64,
    base_overhead: i32,
    per_message: i32,
    tools_exist: i32,
    per_tool: i32,
    per_desc: i32,
    per_first_prop: i32,
    per_additional_prop: i32,
    per_prop_desc: i32,
    per_enum: i32,
    per_nested_object: i32,
    per_array_of_objects: i32,
}

#[derive(Deserialize)]
struct ModelPricing {
    input: f64,
    output: f64,
    #[serde(default)]
    input_cache_read: Option<f64>,
    #[serde(default)]
    input_cache_write: Option<f64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelConfig {
    encoding: String,
    tokens: ModelTokens,
    name: String,
    max_input_tokens: u64,
    max_tokens: u64,
    #[serde(default)]
    pricing: Option<ModelPricing>,
}

// ---------------------------------------------------------------------------
// Claude JSON schema
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ClaudeJsonData {
    pat_str: String,
    special_tokens: HashMap<String, u32>,
    bpe_ranks: String,
}

// ---------------------------------------------------------------------------
// Codegen helpers
// ---------------------------------------------------------------------------

fn fmt_f64(v: f64) -> String {
    // {:?} outputs shortest round-trip representation
    format!("{:?}", v)
}

/// Emit a `Decimal` literal from a JSON-sourced price. `f64`'s `Display`
/// (unlike `Debug`) never uses scientific notation, so it yields the exact,
/// shortest decimal string (e.g. `1e-06` -> `0.000001`) that `dec!` parses
/// losslessly. Prices are exact decimals in `models.json`, so no precision is
/// lost going JSON number -> f64 -> shortest decimal string -> Decimal.
fn fmt_decimal(v: f64) -> String {
    format!("rust_decimal::dec!({})", v)
}

fn fmt_opt_decimal(v: Option<f64>) -> String {
    match v {
        Some(val) => format!("Some({})", fmt_decimal(val)),
        None => "None".to_string(),
    }
}

fn pascal_case(s: &str) -> String {
    let mut out = String::new();
    let mut capitalize = true;
    for c in s.chars() {
        if !c.is_alphanumeric() {
            capitalize = true;
        } else if capitalize {
            out.extend(c.to_uppercase());
            capitalize = false;
        } else {
            out.push(c);
        }
    }
    out
}

fn split_owner_model(key: &str) -> (Option<&str>, &str) {
    match key.split_once('/') {
        Some((owner, model)) => (Some(owner), model),
        None => (None, key),
    }
}

fn fmt_opt_str(v: Option<&str>) -> String {
    match v {
        Some(s) => format!("Some({:?}.to_string())", s),
        None => "None".to_string(),
    }
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let data_dir = Path::new(&manifest_dir).join("data");

    generate_models(&data_dir, &out_dir);
    generate_claude_bpe(&data_dir, &out_dir);

    println!("cargo::rerun-if-changed=data/models.json");
    println!("cargo::rerun-if-changed=data/claude.json");
}

// ---------------------------------------------------------------------------
// Models codegen
// ---------------------------------------------------------------------------

fn generate_models(data_dir: &Path, out_dir: &str) {
    let json = fs::read_to_string(data_dir.join("models.json")).unwrap();
    let models: HashMap<String, ModelConfig> = serde_json::from_str(&json).unwrap();

    let mut sorted: Vec<_> = models.iter().collect();
    sorted.sort_by_key(|(k, _)| k.as_str());

    // Compute variant names. Prefer the bare model_name part; fall back to
    // owner-prefixed PascalCase if the short form collides across owners.
    let mut short_counts: HashMap<String, u32> = HashMap::new();
    for (key, _) in &sorted {
        let (_, name) = split_owner_model(key);
        *short_counts.entry(pascal_case(name)).or_insert(0) += 1;
    }
    let mut variant_for_key: HashMap<String, String> = HashMap::new();
    let mut seen: HashMap<String, String> = HashMap::new();
    for (key, _) in &sorted {
        let (owner, name) = split_owner_model(key);
        let short = pascal_case(name);
        let variant = if short_counts[&short] == 1 {
            short
        } else {
            match owner {
                Some(o) => format!("{}{}", pascal_case(o), pascal_case(name)),
                None => short,
            }
        };
        if let Some(other) = seen.get(&variant) {
            panic!(
                "ModelId variant collision: keys {:?} and {:?} both produce variant {}",
                other, key, variant
            );
        }
        seen.insert(variant.clone(), key.to_string());
        variant_for_key.insert(key.to_string(), variant);
    }

    let mut out = String::with_capacity(128 * 1024);
    out.push_str("// Generated by build.rs — do not edit\n\n");

    // Enum definition
    out.push_str("#[derive(Debug, Clone, PartialEq, Eq, Hash)]\n");
    out.push_str("pub enum ModelId {\n");
    for (key, _) in &sorted {
        let v = &variant_for_key[key.as_str()];
        out.push_str(&format!("    {v},\n"));
    }
    out.push_str("    Custom(String),\n");
    out.push_str("}\n\n");

    out.push_str("impl ModelId {\n");

    // ALL_KNOWN
    out.push_str("    /// All built-in model variants, in canonical (sorted) order.\n");
    out.push_str("    pub const ALL_KNOWN: &'static [ModelId] = &[\n");
    for (key, _) in &sorted {
        let v = &variant_for_key[key.as_str()];
        out.push_str(&format!("        ModelId::{v},\n"));
    }
    out.push_str("    ];\n\n");

    // as_str: canonical "owner/model" key
    out.push_str("    /// Canonical identifier, e.g. \"google/gemini-3.1-pro-preview\".\n");
    out.push_str("    pub fn as_str(&self) -> &str {\n");
    out.push_str("        match self {\n");
    for (key, _) in &sorted {
        let v = &variant_for_key[key.as_str()];
        out.push_str(&format!("            ModelId::{v} => {key:?},\n"));
    }
    out.push_str("            ModelId::Custom(s) => s.as_str(),\n");
    out.push_str("        }\n");
    out.push_str("    }\n\n");

    // parse_known
    out.push_str(
        "    /// Look up a variant by its canonical identifier. Returns `None` for unknown ids.\n",
    );
    out.push_str("    pub fn parse_known(s: &str) -> Option<Self> {\n");
    out.push_str("        match s {\n");
    for (key, _) in &sorted {
        let v = &variant_for_key[key.as_str()];
        out.push_str(&format!("            {key:?} => Some(ModelId::{v}),\n"));
    }
    out.push_str("            _ => None,\n");
    out.push_str("        }\n");
    out.push_str("    }\n\n");

    // from_str_or_custom
    out.push_str(
        "    /// Look up a variant by its canonical identifier, falling back to `Custom`.\n",
    );
    out.push_str("    pub fn from_str_or_custom(s: &str) -> Self {\n");
    out.push_str(
        "        Self::parse_known(s).unwrap_or_else(|| ModelId::Custom(s.to_string()))\n",
    );
    out.push_str("    }\n\n");

    // to_model_config
    out.push_str(
        "    /// Built-in config for known models. `Custom` variants always return `None`.\n",
    );
    out.push_str("    pub fn to_model_config(&self) -> Option<super::ModelConfig> {\n");
    out.push_str("        match self {\n");
    for (key, c) in &sorted {
        let v = &variant_for_key[key.as_str()];
        let (owner, model_name) = split_owner_model(key);
        let pricing = match &c.pricing {
            Some(p) => format!(
                "Some(super::ModelPricing {{ input: {}, output: {}, input_cache_read: {}, input_cache_write: {} }})",
                fmt_decimal(p.input),
                fmt_decimal(p.output),
                fmt_opt_decimal(p.input_cache_read),
                fmt_opt_decimal(p.input_cache_write),
            ),
            None => "None".to_string(),
        };

        out.push_str(&format!(
            "            ModelId::{v} => Some(super::ModelConfig {{\n\
                 owner: {owner},\n\
                 model_name: {mn:?}.to_string(),\n\
                 encoding: {enc:?}.to_string(),\n\
                 tokens: super::ModelTokens {{\n\
                     content_multiplier: {cm},\n\
                     base_overhead: {bo},\n\
                     per_message: {pm},\n\
                     tools_exist: {te},\n\
                     per_tool: {pt},\n\
                     per_desc: {pd},\n\
                     per_first_prop: {pfp},\n\
                     per_additional_prop: {pap},\n\
                     per_prop_desc: {ppd},\n\
                     per_enum: {pe},\n\
                     per_nested_object: {pno},\n\
                     per_array_of_objects: {pao},\n\
                 }},\n\
                 name: {name:?}.to_string(),\n\
                 max_input_tokens: {mit},\n\
                 max_tokens: {mt},\n\
                 pricing: {pricing},\n\
             }}),\n",
            owner = fmt_opt_str(owner),
            mn = model_name,
            enc = c.encoding,
            cm = fmt_f64(c.tokens.content_multiplier),
            bo = c.tokens.base_overhead,
            pm = c.tokens.per_message,
            te = c.tokens.tools_exist,
            pt = c.tokens.per_tool,
            pd = c.tokens.per_desc,
            pfp = c.tokens.per_first_prop,
            pap = c.tokens.per_additional_prop,
            ppd = c.tokens.per_prop_desc,
            pe = c.tokens.per_enum,
            pno = c.tokens.per_nested_object,
            pao = c.tokens.per_array_of_objects,
            name = c.name,
            mit = c.max_input_tokens,
            mt = c.max_tokens,
        ));
    }
    out.push_str("            ModelId::Custom(_) => None,\n");
    out.push_str("        }\n");
    out.push_str("    }\n");

    out.push_str("}\n");

    fs::write(Path::new(out_dir).join("models_generated.rs"), out).unwrap();
}

// ---------------------------------------------------------------------------
// Claude BPE codegen
// ---------------------------------------------------------------------------

fn generate_claude_bpe(data_dir: &Path, out_dir: &str) {
    let json = fs::read_to_string(data_dir.join("claude.json")).unwrap();
    let data: ClaudeJsonData = serde_json::from_str(&json).unwrap();

    // Decode all base64 BPE tokens → compact binary:
    //   repeated [rank: u32 LE][len: u16 LE][token_bytes...]
    let mut bin = Vec::with_capacity(1024 * 1024);
    let mut token_count: u32 = 0;

    for line in data.bpe_ranks.split('\n').filter(|l| !l.is_empty()) {
        let parts: Vec<&str> = line.split(' ').collect();
        if parts.len() < 3 {
            continue;
        }
        let offset: u32 = parts[1].parse().unwrap();
        for (i, b64) in parts[2..].iter().enumerate() {
            let bytes = STANDARD.decode(b64).unwrap();
            let rank = offset + i as u32;
            bin.extend_from_slice(&rank.to_le_bytes());
            bin.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            bin.extend_from_slice(&bytes);
            token_count += 1;
        }
    }

    fs::write(Path::new(out_dir).join("claude_bpe.bin"), &bin).unwrap();

    // Generate Rust source with Claude metadata
    let mut out = String::with_capacity(1024);
    out.push_str("// Generated by build.rs — do not edit\n\n");
    out.push_str(&format!(
        "pub(crate) const CLAUDE_PAT_STR: &str = {:?};\n\n",
        data.pat_str
    ));
    out.push_str(&format!(
        "pub(crate) const CLAUDE_TOKEN_COUNT: u32 = {token_count};\n\n"
    ));
    out.push_str("pub(crate) const CLAUDE_SPECIAL_TOKENS: &[(&str, u32)] = &[\n");

    let mut specials: Vec<_> = data.special_tokens.iter().collect();
    specials.sort_by_key(|(_, v)| *v);
    for (k, v) in &specials {
        out.push_str(&format!("    ({k:?}, {v}),\n"));
    }
    out.push_str("];\n\n");

    out.push_str(
        "pub(crate) const CLAUDE_BPE_DATA: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/claude_bpe.bin\"));\n",
    );

    fs::write(Path::new(out_dir).join("claude_generated.rs"), out).unwrap();
}
