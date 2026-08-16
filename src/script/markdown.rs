use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};

/// Parse a Markdown document into clean plain text suitable for annotation.
///
/// - H1 headings are prefixed with `@@TITLE: ` so the annotator can tag them
/// - H2/H3 headings are prefixed with `@@CHAPTER: `
/// - H4+ headings are prefixed with `@@SECTION: `
/// - Blockquotes, lists, paragraphs → plain text paragraphs separated by blank lines
/// - Code blocks, HTML, images, footnotes are dropped
/// - Emphasis/bold markers are stripped (only the inner text is kept)
pub struct MarkdownParser;

impl MarkdownParser {
    pub fn extract_plain_text(markdown: &str) -> String {
        let mut opts = Options::empty();
        opts.insert(Options::ENABLE_STRIKETHROUGH);

        let parser = Parser::new_ext(markdown, opts);

        let mut out = String::new();
        let mut in_code_block = false;
        let mut in_skip = false; // HTML, images
        let mut heading_level: Option<u8> = None;
        let mut para_buf = String::new();

        for event in parser {
            match event {
                // ── containers that produce paragraphs ────────────────────
                Event::Start(Tag::Paragraph)
                | Event::Start(Tag::Item)
                | Event::Start(Tag::BlockQuote(_)) => {}

                Event::End(TagEnd::Paragraph)
                | Event::End(TagEnd::Item)
                | Event::End(TagEnd::BlockQuote(_)) => {
                    flush_para(&mut para_buf, &mut out);
                }

                // ── headings ──────────────────────────────────────────────
                Event::Start(Tag::Heading { level, .. }) => {
                    heading_level = Some(level as u8);
                    // Prefix the heading text with a structural tag
                    let prefix = match level {
                        HeadingLevel::H1 => "@@TITLE: ",
                        HeadingLevel::H2 | HeadingLevel::H3 => "@@CHAPTER: ",
                        _ => "@@SECTION: ",
                    };
                    para_buf.push_str(prefix);
                }
                Event::End(TagEnd::Heading(_)) => {
                    heading_level = None;
                    flush_para(&mut para_buf, &mut out);
                }

                // ── code blocks → skip ────────────────────────────────────
                Event::Start(Tag::CodeBlock(_)) => {
                    in_code_block = true;
                }
                Event::End(TagEnd::CodeBlock) => {
                    in_code_block = false;
                    para_buf.clear();
                }

                // ── HTML / images → skip ──────────────────────────────────
                Event::Start(Tag::HtmlBlock) => {
                    in_skip = true;
                }
                Event::End(TagEnd::HtmlBlock) => {
                    in_skip = false;
                    para_buf.clear();
                }
                Event::Html(_) | Event::InlineHtml(_) => {}

                // ── inline text ───────────────────────────────────────────
                Event::Text(t) => {
                    if in_code_block || in_skip {
                        continue;
                    }
                    // Prefix heading text with its level marker so the LLM
                    // recognises structural text (Chapter headings etc.)
                    if let Some(_lvl) = heading_level {
                        if !para_buf.is_empty() {
                            para_buf.push(' ');
                        }
                        para_buf.push_str(&t);
                    } else {
                        if !para_buf.is_empty() {
                            para_buf.push(' ');
                        }
                        para_buf.push_str(&t);
                    }
                }

                Event::SoftBreak => {
                    if !para_buf.is_empty() {
                        para_buf.push(' ');
                    }
                }
                Event::HardBreak => {
                    flush_para(&mut para_buf, &mut out);
                }

                Event::Rule => {
                    // Section break — add a blank line
                    if !out.ends_with("\n\n") {
                        out.push('\n');
                    }
                }

                // Inline formatting — just emit the contained text (handled above)
                Event::Start(_) | Event::End(_) => {}

                Event::Code(c) => {
                    // Inline code — include the text so LLM can read it naturally
                    if !para_buf.is_empty() {
                        para_buf.push(' ');
                    }
                    para_buf.push_str(&c);
                }

                _ => {}
            }
        }

        // Flush any remaining buffer
        flush_para(&mut para_buf, &mut out);

        // Normalise runs of blank lines to at most one
        normalise_blank_lines(out)
    }
}

fn flush_para(buf: &mut String, out: &mut String) {
    let trimmed = buf.trim().to_string();
    buf.clear();
    if !trimmed.is_empty() {
        out.push_str(&trimmed);
        out.push_str("\n\n");
    }
}

fn normalise_blank_lines(s: String) -> String {
    let mut result = String::with_capacity(s.len());
    let mut blank_run = 0usize;
    for line in s.lines() {
        if line.trim().is_empty() {
            blank_run += 1;
            if blank_run <= 1 {
                result.push('\n');
            }
        } else {
            blank_run = 0;
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}
