//! Signature-head extraction for a single Rust function / method / trait fn.
//!
//! Mirrors `knowdit_sol::signature::extract_function_signature`: given the
//! source text of one `fn` item, returns its signature header from visibility /
//! `fn` keyword up to (but excluding) the `{ ... }` body block, with outer
//! attributes and comments removed and interior whitespace collapsed.

use color_eyre::eyre::Result;
use proc_macro2::LineColumn;
use syn::spanned::Spanned;

/// Byte-offset lookup over a UTF-8 source string indexed by 1-based line and
/// 0-based character column (`proc_macro2::LineColumn`).
pub(crate) struct SourceLineIndex<'a> {
    source: &'a str,
    line_starts: Vec<usize>,
}

impl<'a> SourceLineIndex<'a> {
    pub(crate) fn new(source: &'a str) -> Self {
        let mut line_starts = vec![0usize];
        for (idx, byte) in source.bytes().enumerate() {
            if byte == b'\n' {
                line_starts.push(idx + 1);
            }
        }
        Self {
            source,
            line_starts,
        }
    }

    /// Convert a `proc_macro2::LineColumn` (1-based `line`, 0-based `char`
    /// `column`) into a char-aligned byte offset inside `self.source`.
    pub(crate) fn byte_offset(&self, lc: LineColumn) -> usize {
        if self.source.is_empty() || lc.line == 0 {
            return 0;
        }
        let line_idx = lc.line.saturating_sub(1);
        let Some(&line_start) = self.line_starts.get(line_idx) else {
            return self.source.len();
        };
        let line_end = self
            .line_starts
            .get(line_idx + 1)
            .copied()
            .unwrap_or(self.source.len());
        let slice = &self.source[line_start..line_end];
        if lc.column == 0 {
            return line_start;
        }
        match slice.char_indices().nth(lc.column) {
            Some((rel_byte, _)) => line_start + rel_byte,
            None => line_end,
        }
    }

    pub(crate) fn slice_span(&self, span: proc_macro2::Span) -> &'a str {
        let lo = self.byte_offset(span.start());
        let hi = self.byte_offset(span.end()).max(lo).min(self.source.len());
        &self.source[lo..hi]
    }
}

/// Extract the collapsed Rust signature line of the first `fn` item in `source`.
///
/// Returns `Ok(None)` when `source` does not parse as a Rust function, method,
/// or trait function declaration.
pub fn extract_function_signature(source: &str) -> Result<Option<String>> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let index = SourceLineIndex::new(trimmed);

    if let Ok(item_fn) = syn::parse_str::<syn::ItemFn>(trimmed) {
        let sig_start = index.byte_offset(vis_or_sig_start(&item_fn.vis, &item_fn.sig));
        let body_start = index.byte_offset(item_fn.block.brace_token.span.open().start());
        if sig_start < body_start && body_start <= trimmed.len() {
            return Ok(collapse_rust_header(&trimmed[sig_start..body_start]));
        }
    }

    if let Ok(impl_fn) = syn::parse_str::<syn::ImplItemFn>(trimmed) {
        let sig_start = index.byte_offset(vis_or_sig_start(&impl_fn.vis, &impl_fn.sig));
        let body_start = index.byte_offset(impl_fn.block.brace_token.span.open().start());
        if sig_start < body_start && body_start <= trimmed.len() {
            return Ok(collapse_rust_header(&trimmed[sig_start..body_start]));
        }
    }

    if let Ok(trait_fn) = syn::parse_str::<syn::TraitItemFn>(trimmed) {
        let sig_start = index.byte_offset(trait_fn.sig.span().start());
        let sig_end = match &trait_fn.default {
            Some(block) => index.byte_offset(block.brace_token.span.open().start()),
            None => index.byte_offset(trait_fn.span().end()),
        };
        if sig_start < sig_end && sig_end <= trimmed.len() {
            return Ok(collapse_rust_header(&trimmed[sig_start..sig_end]));
        }
    }

    Ok(None)
}

fn vis_or_sig_start(vis: &syn::Visibility, sig: &syn::Signature) -> LineColumn {
    match vis {
        syn::Visibility::Inherited => sig.span().start(),
        other => other.span().start(),
    }
}

/// Strip `//` line comments and `/* ... */` block comments from a header slice
/// while preserving string literals, then squash interior whitespace.
fn collapse_rust_header(raw_header: &str) -> Option<String> {
    let mut cleaned = String::with_capacity(raw_header.len());
    let mut chars = raw_header.chars().peekable();
    let mut in_string = false;
    let mut escape = false;

    while let Some(ch) = chars.next() {
        if in_string {
            cleaned.push(ch);
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            cleaned.push(ch);
            continue;
        }
        if ch == '/' {
            if chars.peek() == Some(&'/') {
                for c in chars.by_ref() {
                    if c == '\n' {
                        cleaned.push(' ');
                        break;
                    }
                }
                continue;
            }
            if chars.peek() == Some(&'*') {
                let _ = chars.next();
                let mut depth = 1usize;
                while let Some(c) = chars.next() {
                    if c == '/' && chars.peek() == Some(&'*') {
                        let _ = chars.next();
                        depth += 1;
                    } else if c == '*' && chars.peek() == Some(&'/') {
                        let _ = chars.next();
                        depth -= 1;
                        if depth == 0 {
                            cleaned.push(' ');
                            break;
                        }
                    }
                }
                continue;
            }
        }
        cleaned.push(ch);
    }

    let collapsed = cleaned
        .trim_end_matches(';')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    (!collapsed.is_empty()).then_some(collapsed)
}

#[cfg(test)]
mod tests {
    use super::extract_function_signature;

    fn sig(src: &str) -> Option<String> {
        extract_function_signature(src).expect("signature parse should not error")
    }

    #[test]
    fn anchor_instruction_handler_strips_attrs_and_body() {
        let src = r#"
            /// Deposits collateral into the vault PDA.
            #[access_control(ctx.accounts.validate())]
            pub fn deposit<'info>(
                ctx: Context<'_, '_, '_, 'info, Deposit<'info>>,
                amount: u64, // requested deposit amount
            ) -> Result<()> {
                let vault = &mut ctx.accounts.vault;
                vault.total_assets += amount;
                Ok(())
            }
        "#;
        assert_eq!(
            sig(src).as_deref(),
            Some("pub fn deposit<'info>( ctx: Context<'_, '_, '_, 'info, Deposit<'info>>, amount: u64, ) -> Result<()>"),
        );
    }

    #[test]
    fn trait_declaration_strips_semicolon() {
        assert_eq!(
            sig("fn check_invariant(&self, slot: u64) -> ProgramResult;").as_deref(),
            Some("fn check_invariant(&self, slot: u64) -> ProgramResult"),
        );
    }

    #[test]
    fn non_function_returns_none() {
        assert_eq!(sig("pub struct Vault { pub assets: u64 }"), None);
        assert_eq!(sig("// comment only"), None);
    }
}
