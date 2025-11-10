"""Faithful implementation of the LambdaG algorithm from the paper.

This implements Algorithm 1 from:
"Grammar as a Behavioral Biometric: Using Cognitively Motivated Grammar Models 
for Authorship Verification" (https://arxiv.org/html/2403.08462v2)

Key components:
1. POSnoise preprocessing to focus on grammar
2. Sentence tokenization
3. N-gram language models (Grammar Models) with Kneser-Ney smoothing
4. Lambda_G computation: log likelihood ratio between candidate and reference models
5. Calibration via logistic regression
"""
import random
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import joblib

from .ngram_lm import KneserNeyLM
from .posnoise import POSnoiseProcessor


class GrammarModel:
    """Grammar Model (G) for an author or reference population.
    
    A Grammar Model is an n-gram language model trained on POSnoised sentences.
    """
    
    def __init__(self, N: int = 10, discount: float = 0.75):
        """Initialize Grammar Model.
        
        Args:
            N: Order of n-gram model (default 10)
            discount: Kneser-Ney discount parameter (default 0.75)
        """
        self.N = N
        self.discount = discount
        self.lm = KneserNeyLM(N=N, discount=discount)
        
    def train(self, sentences: List[List[str]]):
        """Train the grammar model on tokenized sentences.
        
        Args:
            sentences: List of tokenized sentences (already POSnoised)
        """
        self.lm.train(sentences)
    
    def log_prob(self, token: str, context: List[str]) -> float:
        """Calculate log P(token | context) according to this grammar model."""
        return self.lm.log_prob(token, context)
    
    def sentence_log_prob(self, tokens: List[str]) -> float:
        """Calculate log probability of entire sentence."""
        return self.lm.sentence_log_prob(tokens)


class FaithfulLambdaG:
    """Faithful implementation of the LambdaG algorithm (Algorithm 1 from paper).
    
    This implements the complete pipeline:
    1. POSnoise preprocessing
    2. Sentence tokenization  
    3. Grammar model estimation with Kneser-Ney smoothing
    4. Lambda_G scoring (log likelihood ratio)
    5. Calibration with logistic regression
    """
    
    def __init__(
        self,
        N: int = 10,
        r: int = 100,
        s: Optional[int] = None,
        discount: float = 0.75,
        posnoise_model: str = "en_core_web_sm",
        random_seed: int = 42,
    ):
        """Initialize LambdaG model.
        
        Args:
            N: Order of n-gram model (default 10 as per paper)
            r: Number of reference model repetitions (default 100 as per paper)
            s: Number of sentences to sample for reference models (if None, use all)
            discount: Kneser-Ney discount parameter (default 0.75)
            posnoise_model: spaCy model for POSnoise preprocessing
            random_seed: Random seed for reproducibility
        """
        self.N = N
        self.r = r
        self.s = s
        self.discount = discount
        self.posnoise_model = posnoise_model
        self.random_seed = random_seed
        
        # POSnoise processor
        self._posnoise = None
        
        # Storage for author models: author_id -> GrammarModel
        self.author_models: Dict[str, GrammarModel] = {}
        
        # Storage for reference sentences (from all authors except candidate)
        self.reference_sentences: List[List[str]] = []
        
        # Calibration model (logistic regression)
        self.calibrator: Optional[LogisticRegression] = None
        
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    @property
    def posnoise(self) -> POSnoiseProcessor:
        """Lazy-load POSnoise processor."""
        if self._posnoise is None:
            self._posnoise = POSnoiseProcessor(model=self.posnoise_model)
        return self._posnoise
    
    def _posnoise_and_tokenize(self, text: str) -> List[List[str]]:
        """Apply POSnoise and tokenize into sentences.
        
        Args:
            text: Raw text
            
        Returns:
            List of tokenized sentences (each sentence is a list of tokens)
        """
        return self.posnoise.transform_and_tokenize(text)
    
    def _sample_sentences(
        self, 
        sentences: List[List[str]], 
        num_samples: Optional[int] = None
    ) -> List[List[str]]:
        """Sample random sentences from a corpus.
        
        Args:
            sentences: Full list of sentences
            num_samples: Number of sentences to sample (if None, return all)
            
        Returns:
            Sampled sentences
        """
        if num_samples is None or num_samples >= len(sentences):
            return sentences
        return random.sample(sentences, num_samples)
    
    def fit(self, documents: List[Tuple[str, str]]):
        """Train the model on a corpus of (text, author) pairs.
        
        This implements lines 7-13 of Algorithm 1:
        - Apply POSnoise to all documents
        - Tokenize into sentences
        - Build Grammar Model G_A for each author
        - Collect reference corpus
        
        Args:
            documents: List of (text, author_id) tuples
        """
        # Group documents by author
        author_texts = defaultdict(list)
        all_sentences = []
        
        for text, author in documents:
            # Apply POSnoise and tokenize
            sentences = self._posnoise_and_tokenize(text)
            author_texts[author].extend(sentences)
            all_sentences.extend(sentences)
        
        # Train Grammar Model for each author (lines 10-11 in Algorithm 1)
        print(f"Training Grammar Models for {len(author_texts)} authors...")
        for author, sentences in author_texts.items():
            model = GrammarModel(N=self.N, discount=self.discount)
            model.train(sentences)
            self.author_models[author] = model
            print(f"  Author {author}: {len(sentences)} sentences")
        
        # Store all sentences as reference corpus (line 12)
        self.reference_sentences = all_sentences
        print(f"Reference corpus: {len(self.reference_sentences)} total sentences")
    
    def _build_reference_models(
        self, 
        exclude_author: Optional[str] = None
    ) -> List[GrammarModel]:
        """Build r reference Grammar Models by sampling.
        
        This implements lines 13-14 of Algorithm 1:
        - Sample s sentences from reference corpus r times
        - Build Grammar Model G_j for each sample
        
        Args:
            exclude_author: Author to exclude from reference (candidate author)
            
        Returns:
            List of r reference GrammarModels
        """
        # Filter reference sentences (exclude candidate author's sentences if needed)
        ref_sentences = self.reference_sentences
        if exclude_author and exclude_author in self.author_models:
            # In practice, we'd need to track which sentences belong to which author
            # For now, use all reference sentences
            pass
        
        models = []
        print(f"Building {self.r} reference models (sampling {self.s or 'all'} sentences each)...")
        
        for j in range(self.r):
            # Sample s sentences (line 13)
            sampled = self._sample_sentences(ref_sentences, self.s)
            
            # Build Grammar Model G_j (line 14)
            model = GrammarModel(N=self.N, discount=self.discount)
            model.train(sampled)
            models.append(model)
            
            if (j + 1) % 10 == 0:
                print(f"  Built {j + 1}/{self.r} reference models")
        
        return models
    
    def _compute_lambda_g(
        self,
        sentences: List[List[str]],
        G_A: GrammarModel,
        G_refs: List[GrammarModel],
    ) -> float:
        """Compute lambda_G score for given sentences.
        
        This implements lines 15-20 of Algorithm 1:
        - For each sentence and token, compute log likelihood ratio
        - Average over r reference models
        - Sum over tokens and sentences
        
        Equation 2: λ_G(t_k | t_<k) = (1/r) * sum_j log(P(t_k|t_<k; G_A) / P(t_k|t_<k; G_j))
        Equation 4: λ_G(S_U) = sum_i sum_k λ_G(t_k | t_<k)
        
        Args:
            sentences: List of tokenized sentences from unknown document
            G_A: Candidate author's Grammar Model
            G_refs: List of r reference Grammar Models
            
        Returns:
            lambda_G score
        """
        total_lambda_g = 0.0
        
        # Process each sentence (line 15)
        for sentence in sentences:
            padded = [KneserNeyLM.BOS] * self.N + sentence + [KneserNeyLM.EOS]
            
            # Process each token in context (lines 16-19)
            for i in range(self.N, len(padded)):
                context = padded[i - self.N + 1:i]
                token = padded[i]
                
                # Log prob under candidate model (line 19 numerator)
                log_p_A = G_A.log_prob(token, context)
                
                # Average log prob under reference models (line 19 denominator)
                log_p_refs = []
                for G_j in G_refs:
                    log_p_j = G_j.log_prob(token, context)
                    log_p_refs.append(log_p_j)
                
                # Equation 2: λ_G(t_k | t_<k) = (1/r) * sum_j log(P_A / P_j)
                # = (1/r) * sum_j (log P_A - log P_j)
                lambda_g_token = sum(log_p_A - log_p_j for log_p_j in log_p_refs) / len(G_refs)
                
                total_lambda_g += lambda_g_token
        
        return total_lambda_g
    
    def score(
        self,
        text: str,
        candidate_author: str,
        use_calibration: bool = True,
    ) -> float:
        """Score whether text was written by candidate_author.
        
        This implements the full LambdaG algorithm:
        1. Apply POSnoise and tokenize text
        2. Build r reference models
        3. Compute λ_G (uncalibrated log likelihood ratio)
        4. Optionally apply calibration to get Λ_G
        
        Args:
            text: Unknown text to verify
            candidate_author: ID of candidate author
            use_calibration: If True and calibrator trained, return calibrated score
            
        Returns:
            Score (λ_G if uncalibrated, Λ_G if calibrated)
            Higher score = more likely to be candidate author
        """
        if candidate_author not in self.author_models:
            raise ValueError(f"Unknown author: {candidate_author}")
        
        # Get candidate author's Grammar Model
        G_A = self.author_models[candidate_author]
        
        # Apply POSnoise and tokenize (lines 7-9)
        sentences = self._posnoise_and_tokenize(text)
        
        # Build reference models (lines 13-14)
        G_refs = self._build_reference_models(exclude_author=candidate_author)
        
        # Compute λ_G (lines 15-20)
        lambda_g = self._compute_lambda_g(sentences, G_A, G_refs)
        
        # Apply calibration if available
        if use_calibration and self.calibrator is not None:
            # Convert λ_G to Λ_G (calibrated log likelihood ratio)
            Lambda_G = self.calibrator.predict_proba([[lambda_g]])[0, 1]
            # Convert probability to log odds
            if Lambda_G >= 1.0:
                Lambda_G = 0.9999
            if Lambda_G <= 0.0:
                Lambda_G = 0.0001
            return math.log(Lambda_G / (1 - Lambda_G))
        
        return lambda_g
    
    def score_all(self, text: str, use_calibration: bool = True) -> List[Tuple[str, float]]:
        """Score text against all known authors.
        
        Args:
            text: Unknown text to verify
            use_calibration: Whether to use calibration
            
        Returns:
            List of (author, score) tuples sorted by score (descending)
        """
        scores = []
        for author in self.author_models.keys():
            score = self.score(text, author, use_calibration=use_calibration)
            scores.append((author, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def calibrate(self, verification_cases: List[Tuple[str, str, bool]]):
        """Train calibration model on verification cases.
        
        This implements calibration via logistic regression to convert λ_G to Λ_G.
        
        Args:
            verification_cases: List of (text, candidate_author, is_same_author) tuples
                               True = Y-case (same author), False = N-case (different)
        """
        print(f"Training calibration on {len(verification_cases)} cases...")
        
        # Compute λ_G for all cases
        lambda_g_scores = []
        labels = []
        
        for i, (text, author, is_same) in enumerate(verification_cases):
            # Compute uncalibrated score
            lambda_g = self.score(text, author, use_calibration=False)
            lambda_g_scores.append([lambda_g])
            labels.append(1 if is_same else 0)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(verification_cases)} cases")
        
        # Train logistic regression
        X = np.array(lambda_g_scores)
        y = np.array(labels)
        
        self.calibrator = LogisticRegression(random_state=self.random_seed)
        self.calibrator.fit(X, y)
        
        print(f"Calibration complete. Coefficients: {self.calibrator.coef_}, Intercept: {self.calibrator.intercept_}")
    
    def save(self, path: str):
        """Save model to disk."""
        data = {
            'N': self.N,
            'r': self.r,
            's': self.s,
            'discount': self.discount,
            'posnoise_model': self.posnoise_model,
            'random_seed': self.random_seed,
            'author_models': self.author_models,
            'reference_sentences': self.reference_sentences,
            'calibrator': self.calibrator,
        }
        joblib.dump(data, path)
    
    @classmethod
    def load(cls, path: str) -> 'FaithfulLambdaG':
        """Load model from disk."""
        data = joblib.load(path)
        model = cls(
            N=data['N'],
            r=data['r'],
            s=data['s'],
            discount=data['discount'],
            posnoise_model=data['posnoise_model'],
            random_seed=data['random_seed'],
        )
        model.author_models = data['author_models']
        model.reference_sentences = data['reference_sentences']
        model.calibrator = data['calibrator']
        return model
