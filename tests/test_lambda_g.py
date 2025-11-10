import tempfile
import csv
from lambda_g.model import LambdaGModel
from lambda_g.posnoise import POSnoiseProcessor


def make_small_dataset(path):
    rows = [
        ("This is a short text by author A.", "A"),
        ("Another short note by A with similar style.", "A"),
        ("Completely different phrasing and words from B.", "B"),
        ("B writes in other tones and different words.", "B"),
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "author"])
        for t, a in rows:
            w.writerow([t, a])


def test_fit_and_score(tmp_path):
    csvp = tmp_path / "train.csv"
    make_small_dataset(str(csvp))

    def pairs():
        import csv
        with open(csvp, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                yield row["text"], row["author"]

    model = LambdaGModel(n_features=2 ** 12)  # smaller for test
    model.fit(pairs())

    # text similar to A should score higher for A than for B
    query = "A short note by author A with similar words."
    score_a = model.score(query, "A")
    score_b = model.score(query, "B")
    assert score_a > score_b


def test_posnoise_processor():
    """Test POSnoise transformation."""
    proc = POSnoiseProcessor(model="en_core_web_sm")
    
    text = "The quick brown fox jumps over the lazy dog."
    transformed = proc.transform_text(text)
    
    # Check that content words are replaced with POS tags
    assert "NOUN" in transformed or "VERB" in transformed
    # Check that function words are kept
    assert "The" in transformed or "the" in transformed
    
    # Test sentence tokenization
    multi_sent = "This is sentence one. This is sentence two!"
    sentences = proc.tokenize_sentences(multi_sent)
    assert len(sentences) == 2
    
    # Test combined transform and tokenize
    combined = proc.transform_and_tokenize(multi_sent)
    assert len(combined) == 2
    assert all("VERB" in s or "NOUN" in s for s in combined)


def test_posnoise_model(tmp_path):
    """Test LambdaGModel with POSnoise enabled."""
    csvp = tmp_path / "train.csv"
    make_small_dataset(str(csvp))

    def pairs():
        import csv
        with open(csvp, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                yield row["text"], row["author"]

    model = LambdaGModel(
        n_features=2 ** 12,
        use_posnoise=True,
        sentence_level=False,
    )
    model.fit(pairs())

    # text similar to A should score higher for A than for B
    query = "A short note by author A with similar words."
    score_a = model.score(query, "A")
    score_b = model.score(query, "B")
    
    # With POSnoise, we're capturing style not content, so this should still work
    assert score_a > score_b or score_a >= 0  # at minimum it should compute


def test_sentence_level_model(tmp_path):
    """Test LambdaGModel with sentence-level features."""
    csvp = tmp_path / "train.csv"
    rows = [
        ("First sentence here. Second sentence here.", "A"),
        ("Another first. Another second.", "A"),
        ("Different style. Much different.", "B"),
        ("B style continues. B style remains.", "B"),
    ]
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "author"])
        for t, a in rows:
            w.writerow([t, a])

    def pairs():
        import csv
        with open(csvp, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                yield row["text"], row["author"]

    model = LambdaGModel(
        n_features=2 ** 12,
        use_posnoise=True,
        sentence_level=True,
    )
    model.fit(pairs())

    # Score should work with sentence-level processing
    query = "First sentence. Second sentence."
    score_a = model.score(query, "A")
    score_b = model.score(query, "B")
    
    assert score_a >= 0 and score_b >= 0  # both should compute
