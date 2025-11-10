"""A minimal, scalable implementation of a LambdaG-like method.

Design decisions (minimal, scalable):
- Use HashingVectorizer (stateless) to avoid storing a large vocabulary.
- Aggregate sparse vectors per author as centroid sums and counts so we can stream large datasets.
- Store normalized centroids and compute cosine similarity for scoring.
- Support optional POSnoise preprocessing for style-focused features.

This is intentionally small and dependency-light.
"""
from collections import defaultdict
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from typing import Optional, Iterable, Tuple, List


class LambdaGModel:
    """Minimal LambdaG-style model.

    Methods
    - fit(stream): stream of (text, author) pairs; builds centroids.
    - save(path) / load(path)
    - score(text, author) -> float (cosine similarity)
    - score_all(text) -> list of (author, score) sorted desc
    """

    def __init__(
        self,
        n_features=2 ** 18,
        ngram_range=(3, 5),
        analyzer="char_wb",
        use_posnoise: bool = False,
        posnoise_model: str = "en_core_web_sm",
        sentence_level: bool = False,
    ):
        """Initialize LambdaG model.
        
        Args:
            n_features: hash space size for vectorizer
            ngram_range: character n-gram range
            analyzer: 'char_wb' (character within word boundaries) or 'char'
            use_posnoise: if True, apply POSnoise preprocessing
            posnoise_model: spaCy model for POSnoise (only used if use_posnoise=True)
            sentence_level: if True, split texts into sentences for feature extraction
        """
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.use_posnoise = use_posnoise
        self.posnoise_model = posnoise_model
        self.sentence_level = sentence_level
        
        # HashingVectorizer is stateless; we recreate when needed
        self._make_vectorizer()

        # POSnoise processor (lazy-loaded)
        self._posnoise_processor = None

        # internal: mapping author -> summed sparse vector
        self._sums = {}
        self._counts = defaultdict(int)
        self.authors = []
        # centroids will be a scipy sparse matrix (n_authors x n_features)
        self.centroids = None

    def _make_vectorizer(self):
        self.vectorizer = HashingVectorizer(
            n_features=self.n_features,
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            alternate_sign=False,
            norm=None,
        )

    @property
    def posnoise_processor(self):
        """Lazy-load POSnoise processor."""
        if self.use_posnoise and self._posnoise_processor is None:
            from .posnoise import POSnoiseProcessor
            self._posnoise_processor = POSnoiseProcessor(model=self.posnoise_model)
        return self._posnoise_processor

    def preprocess(self, text: str) -> List[str]:
        """Preprocess text: apply POSnoise and/or sentence tokenization.
        
        Returns a list of text chunks to vectorize.
        """
        if self.use_posnoise and self.sentence_level:
            # POSnoise + sentence-level
            return self.posnoise_processor.transform_and_tokenize(text)
        elif self.use_posnoise:
            # POSnoise only (document-level)
            return [self.posnoise_processor.transform_text(text)]
        elif self.sentence_level:
            # Sentence-level without POSnoise
            return self.posnoise_processor.tokenize_sentences(text)
        else:
            # No preprocessing
            return [text]

    def _ensure_author_entry(self, author):
        if author not in self._sums:
            # zero CSR row
            self._sums[author] = sparse.csr_matrix((1, self.n_features), dtype=np.float64)

    def fit(self, pairs: Iterable[Tuple[str, str]], progress=None):
        """Fit model by streaming (text, author) pairs.

        pairs: iterable of (text, author)
        progress: optional callable(i) for progress reporting
        """
        for i, (text, author) in enumerate(pairs):
            if progress and (i % 1000 == 0):
                progress(i)
            
            # Preprocess text into chunks
            chunks = self.preprocess(text)
            
            # Vectorize all chunks and aggregate
            for chunk in chunks:
                vec = self.vectorizer.transform([chunk])  # 1 x n_features sparse
                self._ensure_author_entry(author)
                # accumulate
                self._sums[author] = self._sums[author] + vec
                self._counts[author] += 1

        # finalize centroids: compute average and L2-normalize
        authors = sorted(self._sums.keys())
        rows = []
        for a in authors:
            avg = self._sums[a] / max(1, self._counts[a])
            rows.append(avg)

        if rows:
            self.centroids = sparse.vstack(rows).tocsr()
            # L2 normalize rows
            self.centroids = normalize(self.centroids, norm="l2", axis=1)
        else:
            self.centroids = sparse.csr_matrix((0, self.n_features), dtype=np.float64)

        self.authors = authors

    def score(self, text: str, author: str) -> float:
        """Return cosine similarity between text and author's centroid.

        If author unknown, returns -1.0
        """
        if author not in self.authors:
            return -1.0
        
        # Preprocess and aggregate chunks
        chunks = self.preprocess(text)
        vecs = [self.vectorizer.transform([chunk]) for chunk in chunks]
        
        if not vecs:
            return -1.0
        
        # Average vectors across chunks
        vec = sum(vecs) / len(vecs)
        
        # ensure vec L2 normalized to match centroids
        vec = normalize(vec, norm="l2", axis=1)
        idx = self.authors.index(author)
        centroid = self.centroids[idx]
        # cosine similarity between 1xN and 1xN -> scalar
        sim = cosine_similarity(vec, centroid)
        return float(sim[0, 0])

    def score_all(self, text: str) -> List[Tuple[str, float]]:
        """Score text against all authors, return list of (author, score) sorted desc."""
        if not self.authors:
            return []
        
        # Preprocess and aggregate chunks
        chunks = self.preprocess(text)
        vecs = [self.vectorizer.transform([chunk]) for chunk in chunks]
        
        if not vecs:
            return [(a, -1.0) for a in self.authors]
        
        # Average vectors across chunks
        vec = sum(vecs) / len(vecs)
        vec = normalize(vec, norm="l2", axis=1)
        
        sims = cosine_similarity(vec, self.centroids).ravel()
        pairs = list(zip(self.authors, sims.tolist()))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def save(self, path):
        """Save the model (pickles the instance)."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
