#!/usr/bin/env python3
"""Command-line entrypoints for LambdaG models.

Usage (examples):
  # Fit simplified model from CSV (fast, approximate)
  python scripts/run_lambda_g.py fit --input data/train.csv --model model.joblib

  # Fit faithful implementation (slow, paper-accurate with Kneser-Ney LM)
  python scripts/run_lambda_g.py fit --input data/train.csv --model model.joblib --faithful

  # Score a CSV with `text` and `claimed_author`, write scores to output.csv
  python scripts/run_lambda_g.py score --model model.joblib --input data/claims.csv --output scores.csv

Two implementations available:
  1. Simplified (default): Fast vector-based similarity, suitable for large datasets
  2. Faithful (--faithful): Paper-accurate with n-gram LM and Kneser-Ney smoothing
"""
import argparse
import csv
from pathlib import Path
from lambda_g.model import LambdaGModel
from lambda_g.faithful_model import FaithfulLambdaG


def cmd_fit(args):
    if args.faithful:
        # Faithful implementation with n-gram LM and Kneser-Ney smoothing
        print("Using faithful LambdaG implementation (paper-accurate, slow)...")
        model = FaithfulLambdaG(
            N=args.N,
            r=args.r,
            s=args.s,
            discount=args.discount,
            posnoise_model=args.posnoise_model,
            random_seed=args.random_seed,
        )
        
        # Load all documents into memory (faithful implementation needs full corpus)
        documents = []
        with open(args.input, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                documents.append((row[args.text_col], row[args.author_col]))
        
        print(f"Loaded {len(documents)} documents")
        model.fit(documents)
        model.save(args.model)
        print(f"Saved faithful model to {args.model}")
        
    else:
        # Simplified implementation (original)
        print("Using simplified LambdaG implementation (fast, approximate)...")
        model = LambdaGModel(
            n_features=args.n_features,
            use_posnoise=args.use_posnoise,
            posnoise_model=args.posnoise_model,
            sentence_level=args.sentence_level,
        )

        def pairs():
            with open(args.input, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield row[args.text_col], row[args.author_col]

        print("Fitting model (streaming)...")
        if args.use_posnoise:
            print(f"  Using POSnoise preprocessing with spaCy model: {args.posnoise_model}")
        if args.sentence_level:
            print("  Using sentence-level features")
        model.fit(pairs())
        model.save(args.model)
        print(f"Saved model to {args.model}")


def cmd_score(args):
    # Try to load as faithful model first, fall back to simplified
    try:
        model = FaithfulLambdaG.load(args.model)
        is_faithful = True
        print("Loaded faithful LambdaG model")
    except:
        model = LambdaGModel.load(args.model)
        is_faithful = False
        print("Loaded simplified LambdaG model")
    
    with open(args.input, newline="", encoding="utf-8") as f_in, open(args.output, "w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        base_fields = list(reader.fieldnames or [])
        fieldnames = base_fields + ["pred_author", "pred_score", "claimed_score"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            text = row[args.text_col]
            claimed = row.get(args.claim_col) or row.get(args.author_col)
            
            if is_faithful:
                # Faithful model scoring
                scored = model.score_all(text, use_calibration=args.use_calibration)
            else:
                # Simplified model scoring
                scored = model.score_all(text)
                
            if scored:
                pred_author, pred_score = scored[0]
            else:
                pred_author, pred_score = "", -1.0
            
            if is_faithful:
                claimed_score = model.score(text, claimed, use_calibration=args.use_calibration) if claimed else -1.0
            else:
                claimed_score = model.score(text, claimed) if claimed else -1.0
            
            row.update({"pred_author": pred_author, "pred_score": pred_score, "claimed_score": claimed_score})
            writer.writerow(row)
    print(f"Wrote scores to {args.output}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    fit = sub.add_parser("fit", help="Train a LambdaG model")
    fit.add_argument("--input", required=True, help="Input CSV with text and author columns")
    fit.add_argument("--model", required=True, help="Output model file path")
    fit.add_argument("--text-col", default="text", help="Name of text column")
    fit.add_argument("--author-col", default="author", help="Name of author column")
    
    # Faithful implementation options
    fit.add_argument("--faithful", action="store_true", 
                    help="Use faithful paper implementation (n-gram LM with Kneser-Ney)")
    fit.add_argument("--N", type=int, default=10, 
                    help="Order of n-gram model (faithful only, default 10)")
    fit.add_argument("--r", type=int, default=100, 
                    help="Number of reference model repetitions (faithful only, default 100)")
    fit.add_argument("--s", type=int, default=None, 
                    help="Number of sentences to sample per reference model (faithful only, default all)")
    fit.add_argument("--discount", type=float, default=0.75, 
                    help="Kneser-Ney discount parameter (faithful only, default 0.75)")
    fit.add_argument("--random-seed", type=int, default=42, 
                    help="Random seed for reproducibility (faithful only)")
    
    # Simplified implementation options
    fit.add_argument("--n-features", type=int, dest="n_features", default=2 ** 18,
                    help="Hash space size (simplified only)")
    fit.add_argument("--use-posnoise", action="store_true", 
                    help="Apply POSnoise preprocessing (simplified only)")
    fit.add_argument("--sentence-level", action="store_true", 
                    help="Use sentence-level features (simplified only)")
    
    # Common options
    fit.add_argument("--posnoise-model", default="en_core_web_sm", 
                    help="spaCy model for POSnoise")

    score = sub.add_parser("score", help="Score texts against model")
    score.add_argument("--model", required=True, help="Model file path")
    score.add_argument("--input", required=True, help="Input CSV with texts to score")
    score.add_argument("--output", required=True, help="Output CSV with scores")
    score.add_argument("--text-col", default="text", help="Name of text column")
    score.add_argument("--author-col", default="author", help="Name of author column")
    score.add_argument("--claim-col", default="claimed_author", 
                      help="Name of claimed author column")
    score.add_argument("--use-calibration", action="store_true",
                      help="Use calibrated scores (faithful only)")

    args = p.parse_args()
    if args.cmd == "fit":
        cmd_fit(args)
    elif args.cmd == "score":
        cmd_score(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
