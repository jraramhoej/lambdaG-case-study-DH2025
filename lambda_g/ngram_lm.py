"""N-gram language model with modified Kneser-Ney smoothing.

This implements the algorithm described in Section 6.3 of the paper:
"Grammar as a Behavioral Biometric: Using Cognitively Motivated Grammar Models 
for Authorship Verification" (https://arxiv.org/html/2403.08462v2)

Key features:
- Modified Kneser-Ney smoothing with configurable discount parameters
- Support for order N (default 10)
- Handles <BOS> and <EOS> tokens properly
- Efficient storage using dictionaries
"""
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import math


class KneserNeyLM:
    """N-gram language model with modified Kneser-Ney smoothing.
    
    Based on Chen & Goodman (1996) and the paper's Section 6.3.
    """
    
    BOS = "<BOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"
    
    def __init__(self, N: int = 10, discount: float = 0.75, r: int = 3):
        """Initialize the language model.
        
        Args:
            N: Order of the n-gram model (default 10 as per paper)
            discount: Fixed discount parameter D (default 0.75 as per paper)
            r: Number of discount levels (default 3 as suggested by Chen & Goodman)
        """
        self.N = N
        self.discount = discount
        self.r = r
        
        # Basic counts: c(g) for all n-grams up to order N
        self.counts = defaultdict(Counter)
        
        # Prefix counts: N_r(•g) = |{t : c(tg) = r}|
        self.prefix_counts = defaultdict(lambda: defaultdict(int))
        self.prefix_counts_gte = defaultdict(lambda: defaultdict(int))
        
        # Suffix counts: N_r(g•) = |{t : c(gt) = r}|
        self.suffix_counts = defaultdict(lambda: defaultdict(int))
        self.suffix_counts_gte = defaultdict(lambda: defaultdict(int))
        
        # Vocabulary
        self.vocab = set()
        self.num_sentences = 0
        
    def _ngram_key(self, tokens: Tuple[str, ...]) -> Tuple[str, ...]:
        """Convert token sequence to n-gram key."""
        return tuple(tokens)
    
    def _pad_sentence(self, tokens: List[str]) -> List[str]:
        """Pad sentence with BOS and EOS tokens."""
        return [self.BOS] * self.N + tokens + [self.EOS]
    
    def train(self, sentences: List[List[str]]):
        """Train the language model on a corpus of sentences.
        
        Args:
            sentences: List of tokenized sentences (each sentence is a list of tokens)
        """
        self.num_sentences = len(sentences)
        
        # First pass: collect all n-gram counts
        for sentence in sentences:
            padded = self._pad_sentence(sentence)
            
            # Update vocabulary
            for token in sentence:
                self.vocab.add(token)
            
            # Count all n-grams up to order N+1
            for i in range(len(padded)):
                for n in range(1, min(self.N + 2, len(padded) - i + 1)):
                    ngram = self._ngram_key(padded[i:i+n])
                    self.counts[n][ngram] += 1
        
        # Second pass: compute prefix and suffix counts
        for n in range(1, self.N + 2):
            for ngram, count in self.counts[n].items():
                if n > 1:
                    # Prefix counts N_r(•g) - count how many different tokens t have c(tg) = r
                    context = ngram[1:]  # remove first token
                    self.prefix_counts[context][count] += 1
                    for r_val in range(1, count + 1):
                        self.prefix_counts_gte[context][r_val] += 1
                    
                    # Suffix counts N_r(g•) - count how many different tokens t have c(gt) = r
                    context = ngram[:-1]  # remove last token
                    self.suffix_counts[context][count] += 1
                    for r_val in range(1, count + 1):
                        self.suffix_counts_gte[context][r_val] += 1
    
    def _continuation_count(self, tokens: Tuple[str, ...]) -> int:
        """Modified count for lower order n-grams (continuation count).
        
        For n < N: c_KN(g) = N_{1+}(•g) = number of distinct contexts before g
        For n = N: c_KN(g) = c(g) (standard count)
        """
        n = len(tokens)
        if n == self.N:
            return self.counts[n].get(tokens, 0)
        else:
            # Continuation count: how many different tokens appear before this sequence
            return self.prefix_counts_gte[tokens].get(1, 0)
    
    def _get_discount(self, count: int) -> float:
        """Get discount value D(count).
        
        For simplicity, using fixed discount D = 0.75 as in the paper.
        Could be extended to count-dependent discounts D(1), D(2), D(3+).
        """
        return self.discount
    
    def _alpha(self, token: str, context: Tuple[str, ...]) -> float:
        """Compute alpha term in Kneser-Ney recursion (Eq. 17 in paper)."""
        ngram = context + (token,)
        c_kn = self._continuation_count(ngram)
        
        if len(context) == 0 or self.counts[len(context)].get(context, 0) == 0:
            return 0.0
        
        # Sum of modified counts for all continuations
        context_total = sum(
            self._continuation_count(context + (t,))
            for t in self._get_vocab_star()
        )
        
        if context_total == 0:
            return 0.0
        
        discount = self._get_discount(c_kn)
        return max(c_kn - discount, 0.0) / context_total
    
    def _gamma(self, context: Tuple[str, ...]) -> float:
        """Compute gamma (interpolation weight) in Kneser-Ney recursion (Eq. 18)."""
        if len(context) == 0 or self.counts[len(context)].get(context, 0) == 0:
            return 1.0
        
        # Sum of discounts over all continuations
        numerator = 0.0
        for r_val in range(1, self.r + 1):
            discount = self._get_discount(r_val)
            n_r = self.suffix_counts[context].get(r_val, 0)
            numerator += discount * n_r
        
        # Sum of continuation counts
        denominator = sum(
            self._continuation_count(context + (t,))
            for t in self._get_vocab_star()
        )
        
        if denominator == 0:
            return 1.0
        
        return numerator / denominator
    
    def _get_vocab_star(self) -> set:
        """Get extended vocabulary including EOS."""
        return self.vocab | {self.EOS}
    
    def _p_kn_recurse(self, token: str, context: Tuple[str, ...]) -> float:
        """Recursive Kneser-Ney probability calculation (Eq. 15-16 in paper)."""
        # Base case: unigram with uniform smoothing
        if len(context) == 0:
            alpha_val = self._alpha(token, ())
            vocab_size = len(self._get_vocab_star())
            return alpha_val + self._gamma(()) / vocab_size
        
        # Recursive case
        alpha_val = self._alpha(token, context)
        gamma_val = self._gamma(context)
        
        # L(g) operation: remove first token from context
        shorter_context = context[1:] if len(context) > 1 else ()
        
        p_lower = self._p_kn_recurse(token, shorter_context)
        
        return alpha_val + gamma_val * p_lower
    
    def prob(self, token: str, context: List[str]) -> float:
        """Calculate probability P(token | context).
        
        Args:
            token: The token to predict
            context: List of previous tokens (up to N-1 tokens)
            
        Returns:
            Probability P(token | context)
        """
        # Handle UNK tokens
        if token not in self._get_vocab_star() and token != self.BOS:
            token = self.UNK
        
        # Truncate context to N-1 tokens (most recent)
        if len(context) > self.N - 1:
            context = context[-(self.N - 1):]
        
        context_tuple = tuple(context)
        prob_val = self._p_kn_recurse(token, context_tuple)
        
        # Ensure non-zero probability
        return max(prob_val, 1e-10)
    
    def log_prob(self, token: str, context: List[str]) -> float:
        """Calculate log probability log P(token | context)."""
        return math.log(self.prob(token, context))
    
    def sentence_log_prob(self, tokens: List[str]) -> float:
        """Calculate log probability of entire sentence.
        
        Args:
            tokens: List of tokens in the sentence
            
        Returns:
            Sum of log probabilities: sum_i log P(t_i | t_{<i})
        """
        padded = self._pad_sentence(tokens)
        log_prob_sum = 0.0
        
        for i in range(self.N, len(padded)):
            context = padded[i - self.N + 1:i]
            token = padded[i]
            log_prob_sum += self.log_prob(token, context)
        
        return log_prob_sum
    
    def perplexity(self, sentences: List[List[str]]) -> float:
        """Calculate perplexity on a test corpus.
        
        Args:
            sentences: List of tokenized sentences
            
        Returns:
            Perplexity = exp(-1/N * sum log P(sentence))
        """
        total_log_prob = 0.0
        total_tokens = 0
        
        for sentence in sentences:
            total_log_prob += self.sentence_log_prob(sentence)
            total_tokens += len(sentence) + 1  # +1 for EOS
        
        if total_tokens == 0:
            return float('inf')
        
        return math.exp(-total_log_prob / total_tokens)
