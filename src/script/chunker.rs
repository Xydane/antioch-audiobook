use crate::script::{Chunk, ScriptEntry};

/// Utility for splitting / grouping script entries.
pub struct Chunker;

impl Chunker {
    /// Single-speaker mode: split text at paragraph boundaries and attribute
    /// every segment to the narrator voice.
    ///
    /// Segments are sized for the TTS model's practical input capacity (~500 chars).
    /// Voice consistency across segments is guaranteed by the shared speaker_embed
    /// computed from the VoiceDesign reference, not by segment size.
    /// Single-speaker mode: split text at paragraph boundaries and attribute
    /// every segment to the narrator voice.
    ///
    /// Paragraphs prefixed with `@@TITLE:`/`@@CHAPTER:`/`@@SECTION:` (injected by
    /// `MarkdownParser::extract_plain_text`) are emitted as structural entries so
    /// `build_chapters` can place accurate M4B chapter markers.  Their text is
    /// stripped of the prefix before TTS so the narrator reads them cleanly.
    pub fn single_speaker(text: &str, narrator_voice: &str, narrator_style: &str) -> Vec<ScriptEntry> {
        // Split on paragraph boundaries first so headings stay as their own entry.
        let mut entries: Vec<ScriptEntry> = Vec::new();
        let mut body = String::new();

        for para in text.split("\n\n") {
            let para = para.split('\n')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            let para = para.trim();
            if para.is_empty() { continue; }

            // Detect structural heading tags
            let structural = if let Some(t) = para.strip_prefix("@@TITLE: ") {
                Some(("TITLE", t.trim(), "Slow, atmospheric."))
            } else if let Some(t) = para.strip_prefix("@@CHAPTER: ") {
                Some(("CHAPTER", t.trim(), "Measured, clear chapter announcement."))
            } else if let Some(t) = para.strip_prefix("@@SECTION: ") {
                Some(("SECTION", t.trim(), "Neutral, even narration."))
            } else {
                None
            };

            if let Some((speaker, clean_text, instruct)) = structural {
                // Flush any accumulated body text first
                if !body.trim().is_empty() {
                    for seg in split_into_segments(body.trim(), 500) {
                        entries.push(ScriptEntry {
                            speaker: narrator_voice.to_string(),
                            text: seg,
                            instruct: narrator_style.to_string(),
                        });
                    }
                    body.clear();
                }
                // Emit the heading as a structural entry
                entries.push(ScriptEntry {
                    speaker: speaker.to_string(),
                    text: clean_text.to_string(),
                    instruct: instruct.to_string(),
                });
            } else {
                // Accumulate body paragraphs
                if !body.is_empty() { body.push_str("\n\n"); }
                body.push_str(para);
            }
        }

        // Flush remaining body text
        if !body.trim().is_empty() {
            for seg in split_into_segments(body.trim(), 500) {
                entries.push(ScriptEntry {
                    speaker: narrator_voice.to_string(),
                    text: seg,
                    instruct: narrator_style.to_string(),
                });
            }
        }

        entries
    }

    /// Merge consecutive same-speaker entries into one segment per speaker run.
    ///
    /// Segments are capped at `max_chars` to stay within the TTS model's practical
    /// input capacity. Speaker changes are always a split point.
    pub fn group_by_speaker(entries: Vec<ScriptEntry>, max_chars: usize) -> Vec<Chunk> {
        if entries.is_empty() {
            return Vec::new();
        }

        let mut chunks: Vec<Chunk> = Vec::new();
        let mut cur = Chunk {
            speaker: entries[0].speaker.clone(),
            text:    entries[0].text.clone(),
            instruct: entries[0].instruct.clone(),
        };

        for entry in entries.into_iter().skip(1) {
            let same_voice = entry.speaker == cur.speaker && entry.instruct == cur.instruct;
            let fits = cur.text.len() + 1 + entry.text.len() <= max_chars;
            // Never merge a heading-like segment with the following body text.
            // A heading has no terminal sentence-ending punctuation and is short.
            // Merging causes the TTS to run the title into the paragraph without
            // a natural pause boundary, producing loops and mis-phrasing.
            let cur_is_heading = is_heading_like(&cur.text);
            let entry_is_heading = is_heading_like(&entry.text);

            if same_voice && fits && !cur_is_heading && !entry_is_heading {
                cur.text.push(' ');
                cur.text.push_str(&entry.text);
            } else {
                chunks.push(cur);
                cur = Chunk {
                    speaker: entry.speaker,
                    text:    entry.text,
                    instruct: entry.instruct,
                };
            }
        }
        chunks.push(cur);
        chunks
    }
}

/// Split `text` at paragraph/sentence boundaries targeting `max_size` chars.
/// True if `text` looks like a section heading rather than body prose.
/// Headings have no terminal sentence-ending punctuation and are short.
fn is_heading_like(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() { return false; }
    let last = t.chars().last().unwrap_or('.');
    let has_terminal = matches!(last, '.' | '!' | '?' | '…' | '"' | '\'' | '\u{201D}' | '\u{2019}');
    !has_terminal && t.len() < 120
}

fn split_into_segments(text: &str, max_size: usize) -> Vec<String> {
    let mut segments: Vec<String> = Vec::new();
    let mut current = String::new();

    for para in text.split("\n\n") {
        // Join soft-wrapped lines within the paragraph into a single line.
        let para = para.split('\n').map(str::trim).filter(|s| !s.is_empty())
            .collect::<Vec<_>>().join(" ");
        let para = para.trim().to_string();
        if para.is_empty() { continue; }

        if current.len() + para.len() + 2 <= max_size {
            if !current.is_empty() { current.push(' '); }
            current.push_str(&para);
        } else {
            if !current.is_empty() { segments.push(std::mem::take(&mut current)); }
            if para.len() > max_size {
                for sentence in split_sentences(&para) {
                    if current.len() + sentence.len() + 1 <= max_size {
                        if !current.is_empty() { current.push(' '); }
                        current.push_str(&sentence);
                    } else {
                        if !current.is_empty() { segments.push(std::mem::take(&mut current)); }
                        current = sentence;
                    }
                }
            } else {
                current = para;
            }
        }
    }
    if !current.is_empty() { segments.push(current); }
    segments
}

/// Split text into sentences at `.` `!` `?` boundaries.
///
/// Guards against false splits on abbreviations by checking the word
/// immediately before the period: if it looks like an abbreviation
/// (single/double letter, or a known title like Dr/Fr/Mr/St/etc.) we
/// do not split regardless of what follows.
/// Also requires the resulting left-hand sentence to be at least 5 words.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        if matches!(chars[i], '.' | '!' | '?') {
            let next = i + 1;
            if next >= len || chars[next].is_whitespace() {
                // Find the start of the word immediately before the punctuation.
                let word_end = i;
                let mut word_start = i;
                while word_start > start && !chars[word_start - 1].is_whitespace() {
                    word_start -= 1;
                }
                let word_before: String = chars[word_start..word_end].iter().collect();

                // Check for abbreviation patterns:
                // - single letter: "A", "B", etc.
                // - two letters separated by nothing or a period: "R.C", "U.S"
                // - known title abbreviations
                let is_abbrev = {
                    let w = &word_before;
                    let known = ["Dr", "Fr", "Mr", "Mrs", "Ms", "St", "Prof",
                                 "Rev", "Lt", "Col", "Gen", "Sgt", "Cpl",
                                 "vs", "etc", "viz", "cf", "al", "op"];
                    w.len() == 1 && w.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false)
                    || w.len() == 2 && w.chars().all(|c| c.is_alphabetic())
                    || w.contains('.')  // already dotted: "R.C", "U.S"
                    || known.iter().any(|&a| w.eq_ignore_ascii_case(a))
                };

                if is_abbrev { i += 1; continue; }

                // Find the first non-whitespace character after the space.
                let mut j = next;
                while j < len && chars[j].is_whitespace() { j += 1; }

                let next_char = chars.get(j).copied();
                let next_is_upper = match next_char {
                    None => true,
                    Some(c) => c.is_uppercase() || c.is_ascii_digit()
                        || matches!(c, '"' | '\'' | '\u{201C}' | '\u{2018}'),
                };

                if next_is_upper {
                    // Require at least 5 words in the candidate sentence to
                    // avoid splitting on short abbreviation-like fragments.
                    let candidate: String = chars[start..=i].iter().collect();
                    let word_count = candidate.split_whitespace().count();
                    if word_count >= 5 {
                        let s = candidate.trim().to_string();
                        if !s.is_empty() { sentences.push(s); }
                        start = j;
                        i = j;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    let tail: String = chars[start..].iter().collect();
    let tail = tail.trim().to_string();
    if !tail.is_empty() { sentences.push(tail); }
    sentences
}
