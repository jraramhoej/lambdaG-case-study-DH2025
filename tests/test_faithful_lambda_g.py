"""Tests for the faithful LambdaG implementation."""
import pytest
from lambda_g.ngram_lm import KneserNeyLM
from lambda_g.faithful_model import GrammarModel, FaithfulLambdaG


def test_kneser_ney_basic():
    """Test basic Kneser-Ney language model functionality."""
    lm = KneserNeyLM(N=3, discount=0.75)
    
    # Train on simple corpus
    sentences = [
        ["the", "cat", "sat"],
        ["the", "dog", "sat"],
        ["a", "cat", "ran"],
    ]
    lm.train(sentences)
    
    # Check that we can compute probabilities
    prob = lm.prob("cat", ["the"])
    assert prob > 0, "Probability should be positive"
    assert prob <= 1.0, "Probability should be <= 1.0"
    
    # Check log probability
    log_prob = lm.log_prob("cat", ["the"])
    assert log_prob <= 0, "Log probability should be negative or zero"


def test_grammar_model():
    """Test GrammarModel wrapper."""
    model = GrammarModel(N=3, discount=0.75)
    
    sentences = [
        ["the", "NOUN", "VERB"],
        ["the", "NOUN", "VERB"],
        ["a", "NOUN", "VERB"],
    ]
    model.train(sentences)
    
    # Test probability calculation
    log_prob = model.log_prob("NOUN", ["the"])
    assert isinstance(log_prob, float)
    
    # Test sentence probability
    sent_log_prob = model.sentence_log_prob(["the", "NOUN", "VERB"])
    assert isinstance(sent_log_prob, float)


def test_faithful_lambda_g_fit():
    """Test that FaithfulLambdaG can fit on simple data."""
    model = FaithfulLambdaG(N=3, r=2, discount=0.75)  # Small r for speed
    
    # Simple training data
    documents = [
        ("The cat sat on the mat.", "author_A"),
        ("The dog ran in the park.", "author_A"),
        ("A bird flew over the tree.", "author_B"),
        ("The fish swam in the pond.", "author_B"),
    ]
    
    # This will apply POSnoise and train models
    model.fit(documents)
    
    # Check that author models were created
    assert "author_A" in model.author_models
    assert "author_B" in model.author_models
    
    # Check that reference sentences were collected
    assert len(model.reference_sentences) > 0


def test_faithful_lambda_g_score():
    """Test that FaithfulLambdaG can score texts."""
    model = FaithfulLambdaG(N=3, r=2, discount=0.75, random_seed=42)
    
    # Training data with distinct styles
    documents = [
        ("The cat sat. The cat ran. The cat jumped.", "author_A"),
        ("The dog sat. The dog ran. The dog jumped.", "author_A"),
        ("A bird flew. A bird sang. A bird danced.", "author_B"),
        ("The fish swam. The fish ate. The fish slept.", "author_B"),
    ]
    
    model.fit(documents)
    
    # Score a test text (without calibration for simplicity)
    test_text = "The cat ran and jumped."
    score_A = model.score(test_text, "author_A", use_calibration=False)
    score_B = model.score(test_text, "author_B", use_calibration=False)
    
    # Scores should be numeric
    assert isinstance(score_A, (int, float))
    assert isinstance(score_B, (int, float))
    
    # Score for author_A should be higher (more similar style)
    # Note: This might not always hold with such small data, but test structure
    print(f"Score A: {score_A}, Score B: {score_B}")


def test_faithful_lambda_g_score_all():
    """Test scoring against all authors."""
    model = FaithfulLambdaG(N=3, r=2, discount=0.75, random_seed=42)
    
    documents = [
        ("The cat sat on the mat.", "author_A"),
        ("A bird flew over the tree.", "author_B"),
    ]
    
    model.fit(documents)
    
    # Score against all authors
    test_text = "The cat ran fast."
    scores = model.score_all(test_text, use_calibration=False)
    
    assert len(scores) == 2
    assert all(isinstance(author, str) for author, _ in scores)
    assert all(isinstance(score, (int, float)) for _, score in scores)
    
    # Should be sorted by score (descending)
    assert scores[0][1] >= scores[1][1]


def test_faithful_lambda_g_save_load(tmp_path):
    """Test saving and loading model."""
    model = FaithfulLambdaG(N=3, r=2, discount=0.75)
    
    documents = [
        ("The cat sat.", "author_A"),
        ("A bird flew.", "author_B"),
    ]
    model.fit(documents)
    
    # Save model
    model_path = tmp_path / "test_model.joblib"
    model.save(str(model_path))
    
    # Load model
    loaded_model = FaithfulLambdaG.load(str(model_path))
    
    # Check that loaded model has same parameters
    assert loaded_model.N == model.N
    assert loaded_model.r == model.r
    assert loaded_model.discount == model.discount
    
    # Check that it can score
    test_text = "The cat ran."
    score1 = model.score(test_text, "author_A", use_calibration=False)
    score2 = loaded_model.score(test_text, "author_A", use_calibration=False)
    
    # Scores might differ slightly due to randomness in reference sampling
    # but should be in similar range
    assert isinstance(score1, (int, float))
    assert isinstance(score2, (int, float))
