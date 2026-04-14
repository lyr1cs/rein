//! Custom Tantivy tokenizer using jieba-rs for CJK segmentation.
//!
//! Replaces the `enrich_cjk` pre-processing workaround: instead of appending
//! jieba tokens to the raw text before indexing, this tokenizer handles all
//! segmentation natively inside Tantivy's indexing and query pipeline.
//!
//! For Latin/ASCII-only text the tokenizer behaves like SimpleTokenizer
//! (lowercase + split on non-alphanumeric). For CJK and mixed text, jieba
//! provides proper word-boundary segmentation (HMM mode).

use tantivy::tokenizer::{Token, TokenStream, Tokenizer};

/// Tokenizer name registered with Tantivy's TokenizerManager.
pub const TOKENIZER_NAME: &str = "jieba";

/// Tantivy-compatible tokenizer backed by jieba-rs.
/// Clone is required by the Tokenizer trait; the underlying Jieba instance is
/// shared via a `OnceLock` static in `extract::dedup`.
#[derive(Clone)]
pub struct JiebaTokenizer;

/// Pre-collected token stream returned by JiebaTokenizer.
pub struct JiebaTokenStream {
    tokens: Vec<Token>,
    /// Starts at `usize::MAX`; wraps to 0 on first `advance()`.
    position: usize,
}

impl TokenStream for JiebaTokenStream {
    fn advance(&mut self) -> bool {
        self.position = self.position.wrapping_add(1);
        self.position < self.tokens.len()
    }

    fn token(&self) -> &Token {
        &self.tokens[self.position]
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.tokens[self.position]
    }
}

impl Tokenizer for JiebaTokenizer {
    type TokenStream<'a> = JiebaTokenStream;

    fn token_stream(&mut self, text: &str) -> JiebaTokenStream {
        let jieba = crate::extract::dedup::jieba();

        // Use jieba HMM mode for better unknown-word detection.
        let words = jieba.cut(text, true);

        let base_ptr = text.as_ptr() as usize;
        let mut tokens: Vec<Token> = Vec::with_capacity(words.len());
        let mut token_position = 0usize;

        for word in &words {
            // Compute byte offset from pointer delta (words are subslices of `text`)
            let word_offset = word.as_ptr() as usize - base_ptr;

            if crate::extract::dedup::contains_cjk(word) {
                // CJK: jieba already segmented at word boundaries; emit as one token
                let lower: String = word.chars().flat_map(char::to_lowercase).collect();
                let trimmed = lower.trim();
                if !trimmed.is_empty() {
                    tokens.push(Token {
                        offset_from: word_offset,
                        offset_to: word_offset + word.len(),
                        position: token_position,
                        text: trimmed.to_string(),
                        position_length: 1,
                    });
                    token_position += 1;
                }
            } else {
                // ASCII/Latin: split further on non-alphanumeric boundaries so that
                // tokens like "foo-bar", "node.js", and "C++" are indexed correctly.
                let mut run_start: Option<usize> = None;
                for (char_idx, ch) in word.char_indices() {
                    if ch.is_alphanumeric() {
                        run_start.get_or_insert(char_idx);
                    } else if let Some(start) = run_start.take() {
                        let sub = &word[start..char_idx];
                        let lower: String = sub.chars().flat_map(char::to_lowercase).collect();
                        tokens.push(Token {
                            offset_from: word_offset + start,
                            offset_to: word_offset + char_idx,
                            position: token_position,
                            text: lower,
                            position_length: 1,
                        });
                        token_position += 1;
                    }
                }
                // Flush any trailing alphanumeric run
                if let Some(start) = run_start {
                    let sub = &word[start..];
                    let lower: String = sub.chars().flat_map(char::to_lowercase).collect();
                    if !lower.is_empty() {
                        tokens.push(Token {
                            offset_from: word_offset + start,
                            offset_to: word_offset + word.len(),
                            position: token_position,
                            text: lower,
                            position_length: 1,
                        });
                        token_position += 1;
                    }
                }
            }
        }

        JiebaTokenStream {
            tokens,
            position: usize::MAX,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tantivy::tokenizer::TokenStream;

    fn collect_tokens(text: &str) -> Vec<String> {
        let mut tokenizer = JiebaTokenizer;
        let mut stream = tokenizer.token_stream(text);
        let mut result = Vec::new();
        while stream.advance() {
            result.push(stream.token().text.clone());
        }
        result
    }

    #[test]
    fn test_ascii_tokenization() {
        let tokens = collect_tokens("Rust ownership and borrowing");
        assert!(tokens.iter().any(|t| t == "rust"), "expected 'rust' in {tokens:?}");
        assert!(tokens.iter().any(|t| t == "ownership"), "expected 'ownership' in {tokens:?}");
    }

    #[test]
    fn test_cjk_tokenization() {
        let tokens = collect_tokens("机器学习是人工智能的子领域");
        // jieba should split "机器学习" as a word, not individual characters
        let joined = tokens.join("|");
        assert!(
            tokens.iter().any(|t| t == "机器学习" || t == "机器" || t == "学习"),
            "expected CJK segmentation in {joined}"
        );
        assert!(tokens.len() > 1, "should produce multiple tokens from CJK text");
    }

    #[test]
    fn test_mixed_tokenization() {
        let tokens = collect_tokens("Rust编程语言所有权");
        let joined = tokens.join("|");
        assert!(
            tokens.iter().any(|t| t == "rust" || t.contains("rust")),
            "expected 'rust' in {joined}"
        );
        assert!(tokens.len() > 1, "should split mixed text: {joined}");
    }

    #[test]
    fn test_empty_and_whitespace() {
        assert!(collect_tokens("").is_empty());
        assert!(collect_tokens("   ").is_empty());
        assert!(collect_tokens("，。！").is_empty());
    }

    #[test]
    fn test_ascii_punctuation_splitting() {
        // "foo-bar" should yield ["foo", "bar"], not ["foo-bar"]
        let tokens = collect_tokens("foo-bar node.js C++");
        assert!(tokens.iter().any(|t| t == "foo"), "expected 'foo' in {tokens:?}");
        assert!(tokens.iter().any(|t| t == "bar"), "expected 'bar' in {tokens:?}");
        assert!(tokens.iter().any(|t| t == "node"), "expected 'node' in {tokens:?}");
        assert!(tokens.iter().any(|t| t == "js"), "expected 'js' in {tokens:?}");
        assert!(tokens.iter().any(|t| t == "c"), "expected 'c' in {tokens:?}");
    }

    #[test]
    fn test_positions_are_sequential() {
        let mut tokenizer = JiebaTokenizer;
        let mut stream = tokenizer.token_stream("hello world foo");
        let mut positions = Vec::new();
        while stream.advance() {
            positions.push(stream.token().position);
        }
        for (i, &pos) in positions.iter().enumerate() {
            assert_eq!(pos, i, "positions should be sequential");
        }
    }
}
