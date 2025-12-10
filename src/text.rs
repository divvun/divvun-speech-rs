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
        let mut text_encoded = self.text_to_sequence(text);

        // Add space tokens at beginning and end
        // Space is typically at index 9 in the symbol sets
        let space_id = self.symbol_to_id.get(" ").copied().unwrap_or(9);
        text_encoded.insert(0, space_id);
        text_encoded.push(space_id);

        text_encoded
    }

    pub fn symbols(&self) -> &SymbolSet {
        &self.symbols
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbols::ALL_SAMI;

    #[test]
    fn test_encode_text() {
        let processor = TextProcessor::new(ALL_SAMI);
        let tokens = processor.encode_text("Buorre beaivi");
        assert!(!tokens.is_empty());
        // First and last should be space tokens
        assert_eq!(tokens[0], tokens[tokens.len() - 1]);
    }
}
