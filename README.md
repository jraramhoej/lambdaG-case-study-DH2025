# LambdaG for Authorship Verification

Python implementation of the LambdaG method for authorship verification. This repository provides **two implementations**:

1. **Simplified** (default): Fast vector-based similarity for practical use
2. **Faithful** (--faithful): Paper-accurate implementation with n-gram language models

## What is LambdaG?

LambdaG is a forensic authorship verification method from the paper ["Grammar as a Behavioral Biometric: Using Cognitively Motivated Grammar Models for Authorship Verification"](https://arxiv.org/html/2403.08462v2) (2024).

The method:
1. Applies **POSnoise** preprocessing to remove content/topic and preserve style
2. Uses **sentence-level tokenization** for fine-grained analysis
3. Builds **Grammar Models** (n-gram language models with Kneser-Ney smoothing)
4. Computes **λG**: log likelihood ratio between candidate and reference models
5. Optionally applies **calibration** via logistic regression

## Two Implementations

### Simplified Implementation (Default)

**Purpose**: Fast, memory-efficient for large datasets

**Method**:
- Character n-grams (3-5) with HashingVectorizer
- Cosine similarity between sparse vectors
- Optional POSnoise preprocessing
- Streaming-friendly

**Use when**: You need speed and scalability

### Faithful Implementation (--faithful)

**Purpose**: Paper-accurate research implementation

**Method**:
- N-gram language models (order N=10)
- Modified Kneser-Ney smoothing (discount D=0.75)
- Reference model sampling (r=100 repetitions)
- Log likelihood ratio computation (λG)
- Logistic regression calibration (ΛG)

**Use when**: You need reproducible research results matching the paper

**Note**: Much slower (trains 100+ language models per query), requires more memory

## Setup

Install [uv](https://github.com/astral-sh/uv) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Generate lockfile and sync environment:

```bash
make lock
make sync
```

Download spaCy model for POSnoise:

```bash
uv run python -m spacy download en_core_web_sm
```

## Usage

### Simplified Implementation (Fast)

Train a model:

```bash
PYTHONPATH=. uv run python scripts/run_lambda_g.py fit \
  --input data/demo/train.csv \
  --model model.joblib \
  --use-posnoise \
  --sentence-level
```

Score verification claims:

```bash
PYTHONPATH=. uv run python scripts/run_lambda_g.py score \
  --model model.joblib \
  --input data/demo/claims.csv \
  --output scored.csv
```

### Faithful Implementation (Paper-Accurate)

Train a faithful model:

```bash
PYTHONPATH=. uv run python scripts/run_lambda_g.py fit \
  --input data/demo/train.csv \
  --model faithful_model.joblib \
  --faithful \
  --N 10 \
  --r 100 \
  --discount 0.75
```

Score with faithful model:

```bash
PYTHONPATH=. uv run python scripts/run_lambda_g.py score \
  --model faithful_model.joblib \
  --input data/demo/claims.csv \
  --output scored_faithful.csv
```

**Faithful Options**:
- `--N`: N-gram order (default 10, as per paper)
- `--r`: Number of reference model repetitions (default 100)
- `--s`: Sentences to sample per reference (default: all)
- `--discount`: Kneser-Ney discount parameter (default 0.75)
- `--use-calibration`: Apply calibration when scoring (requires training calibration first)

### Run tests

```bash
make test
```

## Input Format

**Training CSV** (`train.csv`):
- Columns: `text`, `author`
- One row per training document

**Claims CSV** (`claims.csv`):
- Columns: `text`, `claimed_author`
- One row per verification claim

**Output CSV** (`scored.csv`):
- Original columns plus: `pred_author`, `pred_score`, `claimed_score`
- Higher `claimed_score` → stronger evidence for claimed authorship

## Implementation Details

### Common Components

**POSnoise Preprocessing** (both implementations)
- Content words (NOUN, VERB, ADJ, ADV, PROPN, NUM) → POS tags
- Function words preserved (preserves syntactic style)
- Example: "The quick brown fox" → "The ADJ ADJ NOUN"

**Sentence Tokenization** (both implementations)
- Uses spaCy for robust sentence boundary detection
- Each sentence processed independently

### Simplified Implementation Details

**Feature Extraction**
- Character n-grams (3-5) with word boundaries
- HashingVectorizer for memory efficiency (262,144 features)
- No vocabulary storage required

**Model Training**
- Streaming aggregation per author (memory-efficient)
- L2-normalized sparse centroids
- Saved as joblib files

**Scoring**
- Cosine similarity between query and author centroids
- Sentence-level averaging when enabled
- Returns scores in range [-1, 1]

### Faithful Implementation Details

**N-gram Language Models**
- Order N=10 (captures long-range dependencies)
- Modified Kneser-Ney smoothing (Chen & Goodman 1996)
- Fixed discount parameter D=0.75
- Handles <BOS>, <EOS>, and <UNK> tokens

**Reference Model Sampling** (Algorithm 1, lines 13-14)
- Samples r=100 random subsets from reference corpus
- Each reference model trained on s sentences (or all if s not specified)
- Ensures robust population statistics

**Lambda_G Computation** (Algorithm 1, lines 15-20)
- For each token in context: λG(t|context) = (1/r) × Σ log(P(t|context; G_A) / P(t|context; G_j))
- Aggregated over all tokens and sentences: λG(D_U) = ΣΣ λG(t|context)
- Returns uncalibrated log likelihood ratio

**Calibration** (optional)
- Logistic regression on training Y/N cases
- Converts λG (uncalibrated) to ΛG (calibrated LR)
- Enables forensic-grade likelihood ratios

### Performance Comparison

| Metric | Simplified | Faithful |
|--------|-----------|----------|
| Training time (100 docs) | ~10 seconds | ~5-10 minutes |
| Scoring time (1 doc) | ~0.1 seconds | ~10-30 seconds |
| Memory usage | Low (streaming) | High (full corpus) |
| Accuracy | Good | Best (paper-level) |
| Use case | Production | Research |

## Files

```
lambda_g/
  __init__.py       - Package exports
  model.py          - LambdaGModel class
  posnoise.py       - POSnoise preprocessing
scripts/
  run_lambda_g.py   - CLI for fit/score
tests/
  test_lambda_g.py  - Unit tests
data/demo/
  train.csv         - Example training data
  claims.csv        - Example claims
```

## References

- Nini, A. (2023). *A Theory of Linguistic Individuality for Authorship Analysis*. Cambridge University Press.
- Ishihara, S. (2021). "Score-Based Likelihood Ratios for Linguistic Text Evidence". *Forensic Science International*, 327.


> Nini, A. ‘Examining an author’s individual grammar’. *Comparative Literature Goes Digital Workshop*, *Digital Humanities 2025*. Universidade Nova de Lisboa, Lisbon, Portugal. 14/07/2025.

The tutorial can be found here: <https://andreanini.github.io/lambdaG-case-study-DH2025/case_study.html>.

## Abstract

This demo talk will introduce the audience to a novel algorithm for stylometry called *LambdaG* (Nini et al. 2025). Although this method is designed as an authorship verification algorithm, this talk will demonstrate how it can also be used to study the unique grammar of an author.

Traditional stylometry techniques tend to be disconnected from linguistic theory. For example, although it is well demonstrated that the frequency of function words is a good discriminator of authors, it is still unclear why this is the case. Theoretical advancements to explain this phenomenon were made by Nini (2023), who proposed that these results can be accounted for by principles of Cognitive Linguistics (e.g. Langacker 1987). More specifically, what is crucial to the emergence of *linguistic individuality* is the concept of *entrenchment* (Bybee 2010; Schmid 2015; Divjak 2019).

The *LambdaG* method is fundamentally based on this idea by modelling entrenchment mathematically using grammar models. A *grammar model* is defined as a language model fitted on a text representation only containing functional items (e.g. *the* NOUN VERB *on the* NOUN). The method therefore exploits the notion of a *language model*, a probability distribution over stretches of language, which is a realistic mathematical model for procedural memory, the memory used for grammar processing (Ullman 2004).

The prediction made by Nini’s (2023) theory is that if a text was produced by a certain author, then the grammatical constructions in this text are more entrenched for this author than for random individuals in the reference population. *LambdaG* works by calculating the ratio between these two entrenchments for the questioned text. Experiments carried out on twelve different corpora validate this prediction and demonstrate that *LambdaG* is superior to many other verification methods, including methods based on Large Language Models or using neural networks.

Another advantage of *LambdaG* compared to other methods is that the *LambdaG* score is fully interpretable by an analyst, thus enabling them to identify which constructions characterise the unique language of an author. This can be done by producing text heatmaps that flag those sequences that are the most useful in identifying the author. For this reason, *LambdaG* can also be used very effectively to study the language of an author and to identify their unique linguistic identity.

The case study adopted for this talk is the analysis of Charles Dickens’ language. Using a corpus of 75 novels (3 novels for 25 authors) from Project Gutenberg (Schöch 2017), I will show how even a random sample of 1,000 words from Dicken’s *Bleak House* contains several constructions that seem to be idiosyncratic to Dickens, despite being seemingly unremarkable (e.g. *I could not* VP *without seeing it*; *it would have been* (*so much*/*far better*) *for me never to have* VERB-*past*).

The procedure to carry out this analysis will be shown step by step in R using the `idiolect` package (Nini 2024).

### References

Bybee, Joan. 2010. *Language, Usage and Cognition*. Cambridge, UK: Cambridge University Press.

Divjak, Dagmar. 2019. *Frequency in Language: Memory, Attention and Learning*. Cambridge, UK: Cambridge University Press. (20 October, 2019).

Langacker, Ronald W. 1987. *Foundations of Cognitive Grammar*. Vol. 1. Stanford, CA: Stanford University Press.

Nini, Andrea. 2023. *A Theory of Linguistic Individuality for Authorship Analysis* (Elements in Forensic Linguistics). Cambridge, UK: Cambridge University Press.

Nini, Andrea. 2024. Idiolect: An R package for forensic authorship analysis. <https://andreanini.github.io/idiolect/>.

Nini, Andrea, Oren Halvani, Lukas Graner, Valerio Gherardi & Shunichi Ishihara. 2025. Grammar as a behavioral biometric: Using cognitively motivated grammar models for authorship verification. arXiv. <https://doi.org/10.48550/arXiv.2403.08462>.

Schmid, Hans-Jörg. 2015. A blueprint of the Entrenchment-and-Conventionalization Model. *Yearbook of the German Cognitive Linguistics Association* 3(1). 3–25. <https://doi.org/10.1515/gcla-2015-0002>.

Schöch, Christof. 2017. refcor. <https://github.com/cophi-wue/refcor>.

Ullman, Michael T. 2004. Contributions of memory circuits to language: the declarative/procedural model. *Cognition* 92(1–2). 231–270. <https://doi.org/10.1016/j.cognition.2003.10.008>.
