use std::collections::HashMap;

/// A set of symbols used for text encoding.
///
/// Can be either borrowed (static symbol sets) or owned (extracted from model).
#[derive(Debug, Clone)]
pub enum SymbolSet {
    Borrowed(&'static [&'static str]),
    Owned(Vec<String>),
}

impl SymbolSet {
    pub const fn new(symbols: &'static [&'static str]) -> Self {
        Self::Borrowed(symbols)
    }

    pub fn from_vec(symbols: Vec<String>) -> Self {
        Self::Owned(symbols)
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        match self {
            Self::Borrowed(s) => SymbolIter::Borrowed(s.iter()),
            Self::Owned(s) => SymbolIter::Owned(s.iter()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Borrowed(s) => s.len(),
            Self::Owned(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

enum SymbolIter<'a> {
    Borrowed(std::slice::Iter<'a, &'static str>),
    Owned(std::slice::Iter<'a, String>),
}

impl<'a> Iterator for SymbolIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Borrowed(iter) => iter.next().copied(),
            Self::Owned(iter) => iter.next().map(|s| s.as_str()),
        }
    }
}

pub struct TextProcessor {
    symbols: SymbolSet,
    symbol_to_id: HashMap<String, i64>,
    #[allow(dead_code)]
    id_to_symbol: HashMap<i64, String>,
}

impl TextProcessor {
    pub fn new(symbol_set: SymbolSet) -> Self {
        let symbol_to_id = symbol_set
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), i as i64))
            .collect::<HashMap<_, _>>();

        let id_to_symbol = symbol_set
            .iter()
            .enumerate()
            .map(|(i, s)| (i as i64, s.to_string()))
            .collect::<HashMap<_, _>>();

        TextProcessor {
            symbols: symbol_set,
            symbol_to_id,
            id_to_symbol,
        }
    }

    fn symbols_to_sequence(&self, symbols: &str) -> Vec<i64> {
        symbols
            .chars()
            .filter_map(|c| self.symbol_to_id.get(&c.to_string()).copied())
            .collect()
    }

    fn text_to_sequence(&self, text: &str) -> Vec<i64> {
        self.symbols_to_sequence(text)
    }

    #[allow(dead_code)]
    pub fn sequence_to_text(&self, sequence: &[i64]) -> String {
        sequence
            .iter()
            .filter_map(|&symbol_id| self.id_to_symbol.get(&symbol_id).map(|x| &**x))
            .collect::<Vec<&str>>()
            .join("")
    }

    pub fn encode_text(&self, text: &str) -> Vec<i64> {
        let input_chars: Vec<char> = text.chars().collect();
        let text_encoded = self.text_to_sequence(text);

        let dropped: Vec<char> = input_chars
            .iter()
            .copied()
            .filter(|c| !self.symbol_to_id.contains_key(&c.to_string()))
            .collect();

        tracing::info!(
            "TTS encode: input_chars={}, encoded_tokens={}, dropped_chars={} ({:?}), first_16_tokens={:?}",
            input_chars.len(),
            text_encoded.len(),
            dropped.len(),
            dropped,
            &text_encoded[..text_encoded.len().min(16)]
        );

        text_encoded
    }

    /// Encode text into tokens AND a list of word-boundary spans.
    ///
    /// Returns `(tokens, spans)` where `spans[i]` is the source word and the
    /// half-open `[tok_start, tok_end)` range into `tokens` that the word
    /// produced. Non-word runs (whitespace, punctuation) are attributed to the
    /// preceding word, so the spans tile the token sequence with no gaps.
    /// Characters not in the alphabet are dropped from `tokens` and do not
    /// extend any span.
    ///
    /// A "word" is a maximal run of `char::is_alphanumeric() || c == '\''`.
    pub fn encode_text_with_spans(&self, text: &str) -> (Vec<i64>, Vec<WordSpan>) {
        let mut tokens: Vec<i64> = Vec::new();
        let mut spans: Vec<WordSpan> = Vec::new();
        let mut current_word: Option<(String, usize)> = None; // (word_text, tok_start)
        let mut in_word = false;

        let is_word_char = |c: char| c.is_alphanumeric() || c == '\'';

        for c in text.chars() {
            let c_is_word = is_word_char(c);

            if c_is_word && !in_word {
                // Start of a new word — close any in-progress non-word run by
                // letting the previous span extend up to here (it already does,
                // since we only extend it on non-word chars).
                current_word = Some((String::new(), tokens.len()));
                in_word = true;
            }

            // Try to map char to a token; chars not in the alphabet are dropped.
            if let Some(&tok) = self.symbol_to_id.get(&c.to_string()) {
                tokens.push(tok);
            }

            if c_is_word {
                if let Some((ref mut word, _)) = current_word {
                    word.push(c);
                }
            } else if in_word {
                // Just transitioned out of a word — commit the word span ending
                // at the current token position (exclusive of this non-word char's
                // tokens, which we'll fold into this same span below).
                if let Some((word, tok_start)) = current_word.take() {
                    spans.push(WordSpan {
                        word,
                        tok_start,
                        tok_end: tokens.len(),
                    });
                }
                in_word = false;
            }

            // Non-word chars (after the transition above) extend the most recent
            // word's span. If there is no prior word (leading punctuation), they
            // start a synthetic span tied to the first real word later.
            if !c_is_word && !in_word {
                if let Some(last) = spans.last_mut() {
                    last.tok_end = tokens.len();
                }
                // No-op if spans is empty — leading non-word tokens get
                // absorbed into the first word's span below.
            }
        }

        // Trailing word (no closing non-word char triggered the commit above).
        if let Some((word, tok_start)) = current_word {
            spans.push(WordSpan {
                word,
                tok_start,
                tok_end: tokens.len(),
            });
        }

        // Absorb any leading non-word tokens into the first word's span.
        if let Some(first) = spans.first_mut() {
            first.tok_start = 0;
        } else if !tokens.is_empty() {
            // Input contained only non-word chars that produced tokens — emit a
            // single empty-word span covering them so spans still tile tokens.
            spans.push(WordSpan {
                word: String::new(),
                tok_start: 0,
                tok_end: tokens.len(),
            });
        }

        (tokens, spans)
    }

    pub fn symbols(&self) -> &SymbolSet {
        &self.symbols
    }
}

/// Half-open token-index span covering one source word (plus any
/// punctuation/whitespace that followed it before the next word).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordSpan {
    pub word: String,
    pub tok_start: usize,
    pub tok_end: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbols::ALL_SAMI;

    #[test]
    fn test_encode_text() {
        let processor = TextProcessor::new(ALL_SAMI);
        let tokens = processor.encode_text("Buorre beaivi");
        assert_eq!(tokens.len(), "Buorre beaivi".chars().count());
    }

    fn assert_spans_tile(tokens: &[i64], spans: &[WordSpan]) {
        if tokens.is_empty() {
            assert!(spans.is_empty(), "tokens empty but spans non-empty: {spans:?}");
            return;
        }
        assert!(!spans.is_empty(), "tokens non-empty but no spans");
        assert_eq!(spans[0].tok_start, 0, "first span doesn't start at 0: {:?}", spans[0]);
        assert_eq!(
            spans.last().unwrap().tok_end,
            tokens.len(),
            "last span doesn't reach end: {:?} vs {}",
            spans.last().unwrap(),
            tokens.len()
        );
        for w in spans.windows(2) {
            assert_eq!(
                w[0].tok_end, w[1].tok_start,
                "gap between spans: {:?} -> {:?}", w[0], w[1]
            );
        }
        for span in spans {
            assert!(
                span.tok_start <= span.tok_end,
                "span has tok_start > tok_end: {span:?}"
            );
        }
    }

    #[test]
    fn test_encode_with_spans_round_trip() {
        let processor = TextProcessor::new(ALL_SAMI);
        let cases = [
            "buorre beaivi",
            "soaittášii ahte dát livččii buorre",
            " leading and trailing spaces ",
            "hello, world! punctuation.",
            "single",
        ];
        for text in cases {
            let plain = processor.encode_text(text);
            let (tokens, spans) = processor.encode_text_with_spans(text);
            assert_eq!(tokens, plain, "round-trip mismatch for {text:?}");
            assert_spans_tile(&tokens, &spans);
        }
    }

    #[test]
    fn test_encode_with_spans_word_count() {
        let processor = TextProcessor::new(ALL_SAMI);
        let (_tokens, spans) =
            processor.encode_text_with_spans("buorre beaivi maid lea");
        let words: Vec<&str> = spans.iter().map(|s| s.word.as_str()).collect();
        assert_eq!(words, vec!["buorre", "beaivi", "maid", "lea"]);
    }

    #[test]
    fn test_encode_with_spans_punctuation_and_spaces() {
        let processor = TextProcessor::new(ALL_SAMI);
        // Trailing punctuation should fold into the preceding word.
        let (_tokens, spans) =
            processor.encode_text_with_spans("buorre, beaivi!");
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].word, "buorre");
        assert_eq!(spans[1].word, "beaivi");
    }

    #[test]
    fn test_encode_with_spans_leading_punctuation() {
        let processor = TextProcessor::new(ALL_SAMI);
        // Leading punctuation gets absorbed into the first word's span.
        let (tokens, spans) = processor.encode_text_with_spans("  hei");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].word, "hei");
        assert_eq!(spans[0].tok_start, 0);
        assert_eq!(spans[0].tok_end, tokens.len());
    }

    #[test]
    fn test_encode_with_spans_empty() {
        let processor = TextProcessor::new(ALL_SAMI);
        let (tokens, spans) = processor.encode_text_with_spans("");
        assert!(tokens.is_empty());
        assert!(spans.is_empty());
    }
}
