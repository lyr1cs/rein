/// Split text into chunks by Markdown headings, then by paragraph, then by sentence.
/// Each chunk <= max_tokens (estimated at 4 chars per token).
/// Overlap: overlap_percent of max_tokens carried from end of previous chunk.
pub fn semantic_chunk(text: &str, max_tokens: usize, overlap_percent: usize) -> Vec<String> {
    let max_chars = max_tokens * 4;
    let overlap_chars = max_chars * overlap_percent / 100;

    if text.is_empty() {
        return vec![];
    }

    // 1. Try to split by Markdown headings (## or ###)
    let sections = split_by_headings(text);

    // If no heading splits and the whole text fits, return as-is
    if sections.len() <= 1 && text.len() <= max_chars {
        return vec![text.to_string()];
    }
    let raw_chunks = if sections.len() > 1 {
        // Split sections that are still too large
        let mut chunks = Vec::new();
        for section in sections {
            if section.len() <= max_chars {
                chunks.push(section);
            } else {
                chunks.extend(split_by_paragraphs(&section, max_chars));
            }
        }
        chunks
    } else {
        // 2. No headings, split by paragraphs
        let paras = split_by_paragraphs(text, max_chars);
        if paras.len() > 1 {
            paras
        } else {
            // 3. Split by sentences
            split_by_sentences(text, max_chars)
        }
    };

    // 4. Apply overlap between consecutive chunks
    apply_overlap(raw_chunks, overlap_chars)
}

/// Split text at Markdown heading boundaries (## or ###).
fn split_by_headings(text: &str) -> Vec<String> {
    let mut sections = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        let trimmed = line.trim_start();
        if (trimmed.starts_with("## ") || trimmed.starts_with("### ")) && !current.is_empty() {
            sections.push(current.trim().to_string());
            current = String::new();
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    if !current.trim().is_empty() {
        sections.push(current.trim().to_string());
    }

    sections
}

/// Split text by double-newline (paragraphs), merging small paragraphs.
fn split_by_paragraphs(text: &str, max_chars: usize) -> Vec<String> {
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut chunks = Vec::new();
    let mut current = String::new();

    for para in paragraphs {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if current.is_empty() {
            current = para.to_string();
        } else if current.len() + para.len() + 2 <= max_chars {
            current.push_str("\n\n");
            current.push_str(para);
        } else {
            chunks.push(current);
            current = para.to_string();
        }
    }
    if !current.is_empty() {
        // If a single chunk is still too large, split by sentences
        if current.len() > max_chars {
            chunks.extend(split_by_sentences(&current, max_chars));
        } else {
            chunks.push(current);
        }
    }

    // Post-process: split any oversized chunks by sentences
    let mut result = Vec::new();
    for chunk in chunks {
        if chunk.len() > max_chars {
            result.extend(split_by_sentences(&chunk, max_chars));
        } else {
            result.push(chunk);
        }
    }
    result
}

/// Split text by sentence boundaries (. ! ?), never breaking mid-sentence.
fn split_by_sentences(text: &str, max_chars: usize) -> Vec<String> {
    let sentences = extract_sentences(text);
    let mut chunks = Vec::new();
    let mut current = String::new();

    for sentence in &sentences {
        if current.is_empty() {
            current = sentence.clone();
        } else if current.len() + sentence.len() + 1 <= max_chars {
            current.push(' ');
            current.push_str(sentence);
        } else {
            chunks.push(current);
            current = sentence.clone();
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }

    // If no chunks were produced (e.g., single very long sentence), force-split
    if chunks.is_empty() && !text.is_empty() {
        chunks.push(text.to_string());
    }

    chunks
}

/// Extract sentences from text, splitting on . ! ? followed by whitespace.
fn extract_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    let mut i = 0;
    while i < len {
        current.push(chars[i]);
        if (chars[i] == '.' || chars[i] == '!' || chars[i] == '?')
            && (i + 1 >= len || chars[i + 1].is_whitespace())
        {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current = String::new();
            // Skip the whitespace after the sentence-ending punctuation
            if i + 1 < len && chars[i + 1].is_whitespace() {
                i += 1;
            }
        }
        i += 1;
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

/// Apply overlap: last N chars of chunk[i] are prepended to chunk[i+1].
fn apply_overlap(chunks: Vec<String>, overlap_chars: usize) -> Vec<String> {
    if chunks.len() <= 1 || overlap_chars == 0 {
        return chunks;
    }

    let mut result = Vec::with_capacity(chunks.len());
    result.push(chunks[0].clone());

    for i in 1..chunks.len() {
        let prev = &chunks[i - 1];
        if overlap_chars > 0 && prev.len() > overlap_chars {
            // Take the last overlap_chars from the previous chunk
            // Find a valid UTF-8 char boundary near the overlap start
            let target_start = prev.len().saturating_sub(overlap_chars);
            let mut overlap_start = target_start;
            while overlap_start < prev.len() && !prev.is_char_boundary(overlap_start) {
                overlap_start += 1;
            }
            // Find a word boundary near the overlap start
            let boundary = prev[overlap_start..]
                .find(' ')
                .map(|p| overlap_start + p + 1)
                .unwrap_or(overlap_start);
            let overlap_text = &prev[boundary..];
            result.push(format!("{} {}", overlap_text.trim(), chunks[i]));
        } else {
            result.push(chunks[i].clone());
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_by_heading() {
        let text = "## Introduction\nThis is the intro.\n\n## Methods\nThese are the methods.\n\n## Results\nHere are results.";
        let chunks = semantic_chunk(text, 512, 0);
        assert!(chunks.len() >= 3, "Should split by headings, got {} chunks", chunks.len());
        assert!(chunks[0].contains("Introduction"));
        assert!(chunks[1].contains("Methods"));
        assert!(chunks[2].contains("Results"));
    }

    #[test]
    fn test_chunk_long_text() {
        // Generate text > 512 tokens (2048 chars) without headings
        let sentence = "This is a test sentence that should be used to fill up content. ";
        let text = sentence.repeat(50); // ~3200 chars = ~800 tokens
        let chunks = semantic_chunk(&text, 512, 0);
        assert!(chunks.len() > 1, "Long text should produce multiple chunks, got {}", chunks.len());
        for chunk in &chunks {
            assert!(
                chunk.len() <= 512 * 4 + 100, // allow small margin for sentence boundaries
                "Chunk too large: {} chars",
                chunk.len()
            );
        }
    }

    #[test]
    fn test_chunk_overlap() {
        let text = "## Part One\nFirst section content here.\n\n## Part Two\nSecond section content here.\n\n## Part Three\nThird section content here.";
        let chunks_no_overlap = semantic_chunk(text, 512, 0);
        let chunks_with_overlap = semantic_chunk(text, 512, 20);

        // With overlap, later chunks should contain text from previous chunks
        assert_eq!(chunks_no_overlap.len(), chunks_with_overlap.len());
        // The first chunk should be the same
        assert_eq!(chunks_no_overlap[0], chunks_with_overlap[0]);
        // Subsequent chunks with overlap should be longer
        if chunks_with_overlap.len() > 1 {
            assert!(
                chunks_with_overlap[1].len() >= chunks_no_overlap[1].len(),
                "Overlap chunk should be at least as long"
            );
        }
    }

    #[test]
    fn test_chunk_no_mid_sentence() {
        let text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here.";
        let chunks = semantic_chunk(text, 20, 0); // 20 tokens = 80 chars
        for chunk in &chunks {
            // Each chunk should end with a sentence-ending punctuation or be the last chunk
            let trimmed = chunk.trim();
            assert!(
                trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?') || trimmed == chunks.last().unwrap().trim(),
                "Chunk should end at sentence boundary: '{}'",
                trimmed
            );
        }
    }

    #[test]
    fn test_chunk_short_text() {
        let text = "Short text that fits in one chunk.";
        let chunks = semantic_chunk(text, 512, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_metadata_prefix() {
        let result = crate::embed::prepend_metadata("debug", "OOM fix", "connection pool leak");
        assert_eq!(result, "topic:debug | OOM fix | connection pool leak");
    }
}
