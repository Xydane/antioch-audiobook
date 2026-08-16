use anyhow::{bail, Result};
use tracing::debug;

use crate::{
    llm::LlmBackend,
    script::ScriptEntry,
};



// ─── Annotator ────────────────────────────────────────────────────────────────

pub struct ScriptAnnotator {
    backend: Box<dyn LlmBackend + Send + Sync>,
}

impl ScriptAnnotator {
    pub fn new(backend: Box<dyn LlmBackend + Send + Sync>) -> Self {
        Self { backend }
    }

    /// Annotate a full text by splitting into chunks.
    ///
    /// Strategy:
    ///   1. Deterministically parse each paragraph into NARRATOR / CHARACTER entries
    ///      using `split_dialogue_paragraph` (no LLM needed for structure).
    ///   2. Send the structural entries to the LLM in a single compact call to
    ///      generate `instruct` voice-direction fields.
    ///      Falls back to sensible defaults if the LLM call fails.
    pub async fn annotate(
        &self,
        text: &str,
        chunk_size: usize,
    ) -> Result<Vec<ScriptEntry>> {
        let chunks = split_text_into_chunks(text, chunk_size);
        let total = chunks.len();
        debug!("Annotating {} chunks", total);

        let mut all_entries: Vec<ScriptEntry> = Vec::new();
        // Running character roster for pronoun resolution across chunks
        let mut known_chars: Vec<String> = Vec::new();
        // Gender hints: map from character name -> "M" or "F"
        let mut gender_map: std::collections::HashMap<String, &'static str> = std::collections::HashMap::new();

        for (i, chunk) in chunks.iter().enumerate() {
            let part_num = i + 1;
            eprintln!("  chunk {part_num}/{total} ({} chars)…", chunk.len());

            // Step 1: detect names and update roster
            let chunk_clean = preprocess_chunk(chunk);
            for n in extract_character_names(&chunk_clean) {
                if !known_chars.contains(&n) {
                    known_chars.push(n);
                }
            }
    infer_gender_hints(&chunk_clean, &known_chars, &mut gender_map);
    // Binary inference: in a 2-character scene if one is M, the other is F.
    // For 3+ characters we can only do this if exactly one gender is represented.
    {
        let m_count = known_chars.iter().filter(|n| gender_map.get(*n) == Some(&"M")).count();
        let f_count = known_chars.iter().filter(|n| gender_map.get(*n) == Some(&"F")).count();
        let unresolved: Vec<_> = known_chars.iter()
            .filter(|n| !gender_map.contains_key(*n))
            .collect();
        // If exactly one gender is known and only one char is unresolved, assign opposite
        if unresolved.len() == 1 {
            if m_count >= 1 && f_count == 0 {
                gender_map.entry(unresolved[0].clone()).or_insert("F");
            } else if f_count >= 1 && m_count == 0 {
                gender_map.entry(unresolved[0].clone()).or_insert("M");
            }
        }
    }

            // Step 2: deterministic structural parse
            let mut chunk_entries: Vec<ScriptEntry> = Vec::new();
            let mut last_speaker: Option<String> = None;
            for para in chunk_clean.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
                chunk_entries.extend(split_dialogue_paragraph(para, &known_chars, &gender_map, &mut last_speaker));
            }
            eprintln!("    parsed {} entries", chunk_entries.len());

            // Step 3: fill instruct fields via LLM
            let instruct_map = self.generate_instructs(&chunk_entries).await
                .unwrap_or_else(|e| {
                    eprintln!("    instruct LLM failed: {e}");
                    Default::default()
                });
            for e in &mut chunk_entries {
                let key = format!("{}|{}", e.speaker, &e.text[..e.text.len().min(60)]);
                if let Some(inst) = instruct_map.get(&key) {
                    e.instruct = inst.clone();
                }
            }

            all_entries.extend(chunk_entries);
        }

        if all_entries.is_empty() {
            bail!("No script entries were generated.");
        }

        // Remove duplicate (speaker, text) pairs
        let mut seen = std::collections::HashSet::new();
        all_entries.retain(|e| seen.insert((e.speaker.clone(), e.text.clone())));

        Ok(all_entries)
    }

    /// Ask the LLM to generate instruct voice-direction fields for a list of entries.
    /// Returns a map of "SPEAKER|text_prefix" -> instruct string.
    async fn generate_instructs(
        &self,
        entries: &[ScriptEntry],
    ) -> Result<std::collections::HashMap<String, String>> {
        if entries.is_empty() {
            return Ok(Default::default());
        }

        // One call per entry: avoids batch-output repetition loops that small
        // models fall into when asked to echo a parallel JSON array.
        let system = concat!(
            "Give a 5-8 word TTS delivery direction for this script line.\n",
            "Narrator lines: pacing/atmosphere (e.g. 'Slow, atmospheric.').\n",
            "Character lines: emotional subtext (e.g. 'Flat surprise.').\n",
            "Reply with ONLY the direction, nothing else."
        );

        // Regexes compiled once outside the loop.
        // 1. Strip speaker labels: "[REEVES]", "NARRATOR:", "Pacing:", "Character:", etc.
        let re_label   = regex::Regex::new(r"(?i)^(?:\[[^\]]*\]|[A-Z][A-Za-z ]{0,20}):\s*").unwrap();
        // 2. Strip leading junk chars: quotes, brackets, spaces
        let re_leading = regex::Regex::new(r#"^[\[\]()"'`\s]+(\S)"#).unwrap();
        // 3. Detect echo: response contains a large chunk of the input text
        let mut map = std::collections::HashMap::new();

        for e in entries {
            let user = format!("[{}] {}", e.speaker, &e.text[..e.text.len().min(80)]);
            let raw = match self.backend.complete(system, &user).await {
                Ok(r) => r,
                Err(_) => continue,
            };
            let trimmed = raw.trim();

            // Discard echo responses: model repeated ≥30 chars of the input.
            let input_snip = &e.text[..e.text.len().min(30)];
            if trimmed.contains(input_snip) {
                continue;
            }

            // Strip label prefixes ("[REEVES] ...", "Narrator:", "Pacing:", ...)
            let inst = re_label.replace(trimmed, "").into_owned();
            // Strip leading punctuation / quotes / brackets
            let inst = re_leading.replace(inst.trim(), "$1").into_owned();
            let inst = inst.trim_start_matches(|c| matches!(c, '[' | '(' | '"' | '\'' | '`'));

            // Take only the first sentence.
            let cut = inst.find(|c| c == '.' || c == '!' || c == '?')
                .map(|i| i + 1).unwrap_or(inst.len());
            let inst = inst[..cut.min(80)].trim().to_string();

            if inst.len() >= 4 {  // discard near-empty results
                let key = format!("{}|{}", e.speaker, &e.text[..e.text.len().min(60)]);
                map.insert(key, inst);
            }
        }

        Ok(map)
    }
}

/// Normalise source text before sending to the LLM:
/// - Strip markdown bold/italic markers (`**text**`, `*text*`, `__text__`) so
///   `**Guard:**` becomes `Guard:`, which the model reliably recognises as a
///   labelled dialogue line.
fn preprocess_chunk(text: &str) -> String {
    // Remove bold/italic markers (greedy-safe: strip ** before *)
    let re = regex::Regex::new(r"\*{1,3}([^*]+)\*{1,3}").unwrap();
    let text = re.replace_all(text, "$1");
    let re2 = regex::Regex::new(r"_{1,2}([^_]+)_{1,2}").unwrap();
    re2.replace_all(&text, "$1").into_owned()
}

/// Join multiple speech spans from a split attribution into one clean string.
/// e.g. `["They're not going to announce anything,", "They never do when it's this bad."]`
///   -> `"They're not going to announce anything. They never do when it's this bad."`
///
/// Rules:
///   - If a span ends with `,` it's a sentence continuation; replace `,` with `. `
///     (or just a space if the next span starts with a lowercase letter — unlikely in dialogue)
///   - Final span punctuation is preserved as-is.
fn join_speech_parts(parts: &[String]) -> String {
    if parts.len() == 1 {
        return parts[0].trim_end_matches(',').trim().to_string();
    }
    let mut out = String::new();
    for (i, part) in parts.iter().enumerate() {
        let part = part.trim();
        if i == parts.len() - 1 {
            // Last part: trim trailing comma if any
            out.push_str(part.trim_end_matches(',').trim());
        } else if part.ends_with(',') {
            // Mid-speech attribution split: comma at end means sentence continues
            out.push_str(part.trim_end_matches(',').trim());
            out.push_str(". "); // convert to sentence break
        } else {
            out.push_str(part);
            out.push(' ');
        }
    }
    out.trim().to_string()
}

/// Strip a leading attribution clause from a narration span.
/// e.g. `"Reeves said. He said it without..."` -> `"He said it without..."`
/// e.g. `"said Priya."` -> `""`
fn strip_leading_attribution(part: &str, attr_re: &regex::Regex, inv_attr_re: &regex::Regex) -> String {
    let trimmed = part.trim();
    // Try forward: `Name said.` at start
    if let Some(m) = attr_re.find(trimmed) {
        if m.start() == 0 {
            // Find the end of the attribution sentence (up to first `.`)
            let tail = &trimmed[m.end()..];
            let skip = tail.find('.').map(|i| i + 1).unwrap_or(tail.len());
            return tail[skip..].trim().to_string();
        }
    }
    // Try inverted: `said Name.` at start
    if let Some(m) = inv_attr_re.find(trimmed) {
        if m.start() == 0 {
            let tail = &trimmed[m.end()..];
            let skip = tail.find('.').map(|i| i + 1).unwrap_or(tail.len());
            return tail[skip..].trim().to_string();
        }
    }
    trimmed.to_string()
}

/// Deterministically split a single paragraph into NARRATOR and CHARACTER
/// `ScriptEntry` values by parsing its quotation structure.
///
/// Returns a Vec that may contain:
///   - A NARRATOR entry for non-quoted prose
///   - One or more CHARACTER entries for quoted speech
///   - Merged consecutive speech spans from the same speaker
fn split_dialogue_paragraph(
    para: &str,
    known_chars: &[String],
    gender_map: &std::collections::HashMap<String, &'static str>,
    last_speaker: &mut Option<String>,
) -> Vec<ScriptEntry> {
    // Split on ASCII double-quote boundaries.
    // Odd-indexed segments are inside quotes; even-indexed are outside.
    let parts: Vec<&str> = para.split('"').collect();
    if parts.len() == 1 {
        // No quotes — check for structural heading tags injected by markdown.rs
        let (speaker, text, instruct) = if let Some(t) = para.strip_prefix("@@TITLE: ") {
            ("TITLE", t.trim(), "Slow, atmospheric.")
        } else if let Some(t) = para.strip_prefix("@@CHAPTER: ") {
            ("CHAPTER", t.trim(), "Measured, clear chapter announcement.")
        } else if let Some(t) = para.strip_prefix("@@SECTION: ") {
            ("SECTION", t.trim(), "Neutral, even narration.")
        } else {
            ("NARRATOR", para.trim(), "Neutral, even narration.")
        };
        *last_speaker = None;
        return vec![ScriptEntry {
            speaker:  speaker.to_string(),
            text:     text.to_string(),
            instruct: instruct.to_string(),
        }];
    }

    // Forward attribution: `Name said/asked/...`
    let attr_re = regex::Regex::new(
        r#"(?i)\b([A-Za-z][a-z]{1,19}|he|she|they|it)\s+(?:said|asked|replied|answered|added|continued|whispered|shouted|muttered|called|cried|laughed)"#,
    ).unwrap();
    // Inverted attribution: `said/asked/... Name`
    // Match is case-insensitive for the verb but we filter the captured name in code.
    let inv_attr_re = regex::Regex::new(
        r#"(?i)\b(?:said|asked|replied|answered|added|continued|whispered|shouted|muttered|called|cried|laughed)\s+([A-Za-z][a-z]{1,19})\b"#,
    ).unwrap();

    // Determine the speaker for this paragraph by scanning non-quoted spans for:
    //   1. Forward attribution: `Name said`
    //   2. Inverted attribution: `said Name`
    //   3. Action beat: paragraph-initial known-character name (`Yemi leaned...`)
    let mut para_speaker: Option<String> = None;
    'scan: for (idx, part) in parts.iter().enumerate() {
        if idx % 2 == 0 {  // non-quoted span
            let part_t = part.trim();
            // 1. Forward attribution
            if let Some(cap) = attr_re.captures(part_t) {
                let raw = cap[1].to_string();
                let upper = raw.to_uppercase();
                let resolved = match upper.as_str() {
                    "HE"   => resolve_gendered_pronoun("M", known_chars, gender_map, last_speaker),
                    "SHE"  => resolve_gendered_pronoun("F", known_chars, gender_map, last_speaker),
                    "THEY" | "IT" => alternate_speaker(known_chars, last_speaker),
                    _ if known_chars.contains(&upper) => upper,
                    _ => upper,
                };
                para_speaker = Some(resolved);
                break 'scan;
            }
            // 2. Inverted attribution: `said Name` — only match proper names (initial capital)
            if let Some(cap) = inv_attr_re.captures(part_t) {
                let raw = &cap[1];
                if raw.chars().next().map_or(false, |c| c.is_uppercase()) {
                    let name = raw.to_uppercase();
                    let resolved = if known_chars.contains(&name) { name }
                                   else { alternate_speaker(known_chars, last_speaker) };
                    para_speaker = Some(resolved);
                    break 'scan;
                }
            }
            // 3. Action beat: a known character name appears in this non-quoted span
            //    (e.g. "Yemi leaned forward" or "Detective Callum Reeves didn't look up")
            for word in part_t.split_whitespace() {
                // Strip trailing punctuation
                let w = word.trim_end_matches(|c: char| !c.is_alphabetic()).to_uppercase();
                if known_chars.contains(&w) {
                    para_speaker = Some(w);
                    break 'scan;
                }
            }
        }
    }

    // If no attribution found, alternate from last known speaker
    let speaker = para_speaker.unwrap_or_else(|| {
        alternate_speaker(known_chars, last_speaker)
    });

    let mut entries: Vec<ScriptEntry> = Vec::new();
    let mut speech_parts: Vec<String> = Vec::new();

    for (idx, part) in parts.iter().enumerate() {
        let part = part.trim();
        if part.is_empty() { continue; }

        if idx % 2 == 1 {
            // Inside quotes = speech.
            // Strip trailing comma only when it's NOT the final speech span
            // (trailing comma indicates an interrupted/continued line).
            let speech = part.trim().to_string();
            if !speech.is_empty() {
                speech_parts.push(speech);
            }
        } else {
            // Outside quotes = narration / attribution beat.
            // Strip a leading attribution clause (`Reeves said.` / `said Priya.`) from
            // the narration span so it doesn't bleed into NARRATOR text.
            let cleaned_part = strip_leading_attribution(part, &attr_re, &inv_attr_re);

            // If nothing remains after stripping, it was a pure attribution tag — skip it.
            if cleaned_part.trim().is_empty() {
                continue;
            }

            // Flush accumulated speech first
            if !speech_parts.is_empty() {
                entries.push(ScriptEntry {
                    speaker:  speaker.clone(),
                    text:     join_speech_parts(&speech_parts),
                    instruct: "Neutral, even narration.".to_string(),
                });
                speech_parts.clear();
            }

            // Emit meaningful narration beat
            entries.push(ScriptEntry {
                speaker:  "NARRATOR".to_string(),
                text:     cleaned_part.trim().to_string(),
                instruct: "Neutral, even narration.".to_string(),
            });
        }
    }
    // Flush any remaining speech
    if !speech_parts.is_empty() {
        entries.push(ScriptEntry {
            speaker:  speaker.clone(),
            text:     join_speech_parts(&speech_parts),
            instruct: "Neutral, even narration.".to_string(),
        });
    }

    // Update last_speaker
    if speaker != "NARRATOR" {
        *last_speaker = Some(speaker);
    }

    entries
}

/// Resolve a gendered pronoun to a known character name using the gender map.
fn resolve_gendered_pronoun(
    gender: &str,
    known_chars: &[String],
    gender_map: &std::collections::HashMap<String, &'static str>,
    last_speaker: &Option<String>,
) -> String {
    if known_chars.is_empty() {
        return if gender == "M" { "HE".to_string() } else { "SHE".to_string() };
    }
    // First try: find a character whose known gender matches
    for name in known_chars {
        if gender_map.get(name).copied() == Some(gender) {
            return name.clone();
        }
    }
    // Fallback: alternate from last speaker
    alternate_speaker(known_chars, last_speaker)
}

/// Scan text for gender hints and update the gender map.
/// Rules:
///   `Name said/asked/...` -> we can't determine gender directly from this alone.
///   `"Name," he said`     -> NAME is male (M).
///   `"Name," she said`    -> NAME is female (F).
///   `he said` standalone  -> last named speaker in the same para is M.
///   `she said` standalone -> last named speaker in the same para is F.
fn infer_gender_hints(
    text: &str,
    known_chars: &[String],
    gender_map: &mut std::collections::HashMap<String, &'static str>,
) {
    // Pattern: `"Name," he/she said`
    let name_pronoun_re = regex::Regex::new(
        r#"(?i)"([A-Z][a-z]{1,19}),?"\s*(he|she)\s+(?:said|asked|replied)"#
    ).unwrap();
    // Pattern: `said/asked Name` (inverted) — name follows verb directly
    let inv_name_re = regex::Regex::new(
        r#"(?i)\b(?:said|asked|replied|answered|added|continued|whispered|shouted|muttered)\s+([A-Za-z][a-z]{1,19})\b"#
    ).unwrap();
    let he_re  = regex::Regex::new(r#"(?i)\bhe\s+(?:said|asked|replied|answered|added|continued|whispered|shouted|muttered)"#).unwrap();
    let she_re = regex::Regex::new(r#"(?i)\bshe\s+(?:said|asked|replied|answered|added|continued|whispered|shouted|muttered)"#).unwrap();
    let named_re = regex::Regex::new(
        r#"(?i)\b([A-Z][a-z]{1,19})\s+(?:said|asked|replied|answered|added|continued|whispered|shouted|muttered)"#
    ).unwrap();

    for para in text.split("\n\n") {
        // Direct: `"Marcus," he said` -> MARCUS = M
        for cap in name_pronoun_re.captures_iter(para) {
            let name   = cap[1].to_uppercase();
            let gender = if cap[2].to_lowercase() == "he" { "M" } else { "F" };
            if known_chars.contains(&name) {
                gender_map.entry(name).or_insert(gender);
            }
        }
        let has_he  = he_re.is_match(para);
        let has_she = she_re.is_match(para);
        // Forward named attribution co-occurrence with pronoun
        for cap in named_re.captures_iter(para) {
            let name = cap[1].to_uppercase();
            if !known_chars.contains(&name) { continue; }
            if has_he && !has_she {
                gender_map.entry(name).or_insert("M");
            } else if has_she && !has_he {
                gender_map.entry(name).or_insert("F");
            }
        }
        // Inverted attribution: `said Name` — same co-occurrence logic
        for cap in inv_name_re.captures_iter(para) {
            if !cap[1].chars().next().map_or(false, |c| c.is_uppercase()) { continue; }
            let name = cap[1].to_uppercase();
            if !known_chars.contains(&name) { continue; }
            if has_he && !has_she {
                gender_map.entry(name).or_insert("M");
            } else if has_she && !has_he {
                gender_map.entry(name).or_insert("F");
            }
        }
    }
}
/// When no attribution is present, alternate from the last known speaker.
fn alternate_speaker(known_chars: &[String], last_speaker: &Option<String>) -> String {
    if known_chars.is_empty() {
        return "UNKNOWN".to_string();
    }
    for name in known_chars {
        if Some(name) != last_speaker.as_ref() {
            return name.clone();
        }
    }
    known_chars[0].clone()
}

/// Scan a chunk of text for character names appearing in attribution patterns
/// like `"Marcus," he said` or `Marcus said` and return them uppercased.
fn extract_character_names(text: &str) -> Vec<String> {
    // Match: `Name said/asked/...` (forward attribution)
    let re = regex::Regex::new(
        r#"(?:"[^"]*,"\s+)?([A-Z][a-z]{1,19})\s+(?:said|asked|replied|answered|added|continued|whispered|shouted|muttered|called|cried|laughed|nodded|smiled|sighed)"#
    ).unwrap();
    // Match: `said/asked Name` (inverted attribution)
    let re_inv = regex::Regex::new(
        r#"(?i)\b(?:said|asked|replied|answered|added|continued|whispered|shouted|muttered)\s+([A-Za-z][a-z]{1,19})\b"#
    ).unwrap();
    // Match: action beats — `Name verb` where verb is common action
    let re_action = regex::Regex::new(
        r#"\b([A-Z][a-z]{1,19})\s+(?:leaned|turned|looked|pulled|dropped|reached|stood|sat|crossed|folded|put|set|held|stepped|walked|moved|nodded|smiled|frowned|paused|hesitated|glanced|watched|studied|opened|closed)"#
    ).unwrap();
    // Match: `"Name."` as a standalone quoted single-word line (name introduction)
    let re2 = regex::Regex::new(r#"^"([A-Z][a-z]{1,19})\.?"$"#).unwrap();

    let mut names: std::collections::HashSet<String> = std::collections::HashSet::new();
    for cap in re.captures_iter(text) { names.insert(cap[1].to_uppercase()); }
    for cap in re_inv.captures_iter(text) {
        // Only treat as a name if the captured word starts with a capital letter
        if cap[1].chars().next().map_or(false, |c| c.is_uppercase()) {
            names.insert(cap[1].to_uppercase());
        }
    }
    for cap in re_action.captures_iter(text) { names.insert(cap[1].to_uppercase()); }
    for line in text.lines() {
        if let Some(cap) = re2.captures(line.trim()) {
            names.insert(cap[1].to_uppercase());
        }
    }
    // Exclude common pronouns/non-names
    let stop: std::collections::HashSet<&str> = ["HE", "SHE", "THEY", "IT", "WE", "YOU", "SAID", "ASKED"].iter().copied().collect();
    names.retain(|n| !stop.contains(n.as_str()));
    let mut v: Vec<String> = names.into_iter().collect();
    v.sort_unstable();
    v
}

/// Split text at paragraph/sentence boundaries, targeting `max_size` chars.
fn split_text_into_chunks(text: &str, max_size: usize) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for para in text.split("\n\n") {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }

        if !current.is_empty() && current.len() + para.len() + 2 > max_size {
            chunks.push(std::mem::take(&mut current));
        }

        if para.len() > max_size {
            // Split long paragraph at sentence boundaries
            for sentence in split_sentences(para) {
                if !current.is_empty() && current.len() + sentence.len() + 1 > max_size {
                    chunks.push(std::mem::take(&mut current));
                }
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(&sentence);
            }
        } else {
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(para);
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        if matches!(chars[i], '.' | '!' | '?') {
            let next = i + 1;
            if next < len && chars[next].is_whitespace() {
                let s: String = chars[start..=i].iter().collect();
                let s = s.trim().to_string();
                if !s.is_empty() { sentences.push(s); }
                let mut j = next;
                while j < len && chars[j].is_whitespace() { j += 1; }
                start = j;
                i = j;
                continue;
            }
        }
        i += 1;
    }
    let tail: String = chars[start..].iter().collect();
    let tail = tail.trim().to_string();
    if !tail.is_empty() { sentences.push(tail); }
    sentences
}

