"""POSnoise preprocessing for LambdaG authorship verification.

The POSnoise algorithm replaces content words (nouns, verbs, adjectives, adverbs)
with their POS tags while keeping function words (pronouns, determiners, prepositions, etc.)
intact. This removes topic-specific content and focuses on stylistic patterns.

Example:
  Input:  "The quick brown fox jumps over the lazy dog."
  Output: "The ADJ ADJ NOUN VERB over the ADJ NOUN ."
"""
import spacy
from typing import List, Optional


class POSnoiseProcessor:
    """Preprocessor that applies POSnoise transformation using spaCy."""

    # Content word POS tags to replace
    CONTENT_TAGS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}

    def __init__(self, model: str = "en_core_web_sm", batch_size: int = 50):
        """Initialize with a spaCy model.
        
        Args:
            model: spaCy model name (default: en_core_web_sm)
            batch_size: batch size for spaCy pipe processing
        """
        self.model_name = model
        self.batch_size = batch_size
        self._nlp = None

    @property
    def nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                raise RuntimeError(
                    f"spaCy model '{self.model_name}' not found. "
                    f"Install it with: python -m spacy download {self.model_name}"
                )
        return self._nlp

    def transform_token(self, token) -> str:
        """Transform a single token: replace content words with POS tag."""
        if token.pos_ in self.CONTENT_TAGS:
            return token.pos_
        else:
            return token.text

    def transform_text(self, text: str) -> str:
        """Apply POSnoise to a single text."""
        doc = self.nlp(text)
        return " ".join(self.transform_token(tok) for tok in doc)

    def transform_batch(self, texts: List[str]) -> List[str]:
        """Apply POSnoise to a batch of texts (more efficient)."""
        docs = self.nlp.pipe(texts, batch_size=self.batch_size)
        return [" ".join(self.transform_token(tok) for tok in doc) for doc in docs]

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def transform_and_tokenize(self, text: str) -> List[str]:
        """Apply POSnoise and return list of transformed sentences."""
        doc = self.nlp(text)
        result = []
        for sent in doc.sents:
            transformed = " ".join(self.transform_token(tok) for tok in sent)
            result.append(transformed)
        return result
