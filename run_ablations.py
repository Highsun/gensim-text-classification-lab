"""One-factor-at-a-time FastText ablations, evaluated on validation data only."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from gensim.models import FastText
from gensim.models.fasttext_inner import compute_ngrams_bytes, ft_hash_bytes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from run_experiments import mean_vectors
from src.corpus import load_articles
from src.labels import STAGES, make_labels

BASELINE = {
    "vector_size": 64, "window": 5, "min_count": 2, "epochs": 5,
    "workers": 1, "seed": 42, "sg": 1, "negative": 5,
    "min_n": 2, "max_n": 4, "bucket": 100000,
}
VARIANTS = {
    "baseline": {},
    "no_subwords": {"min_n": 1, "max_n": 0},
    "ngrams_3_6": {"min_n": 3, "max_n": 6},
    "bucket_10000": {"bucket": 10000},
    "dimension_128": {"vector_size": 128},
    "window_2": {"window": 2},
    "epochs_10": {"epochs": 10},
    "min_count_5": {"min_count": 5},
    "cbow": {"sg": 0},
}


def validate_variants():
    for name, change in VARIANTS.items():
        if name == "baseline":
            assert not change
        elif len(change) not in (1, 2) or (len(change) == 2 and set(change) != {"min_n", "max_n"}):
            raise ValueError(f"{name} is not a one-factor-at-a-time change: {change}")


def validation_scores(vectors, labels, indices):
    scores = {}
    for stage in STAGES:
        allowed = set(stage)
        train = [i for i in indices["train"] if labels[i] in allowed]
        valid = [i for i in indices["validation"] if labels[i] in allowed]
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(vectors[train], np.asarray(labels)[train])
        predicted = classifier.predict(vectors[valid])
        scores[str(len(stage)) + "_classes"] = {
            "n_train": len(train), "n_validation": len(valid),
            "accuracy": float(accuracy_score(np.asarray(labels)[valid], predicted)),
            "macro_f1": float(f1_score(np.asarray(labels)[valid], predicted, average="macro")),
        }
    return scores


def bucket_statistics(model):
    if not model.wv.bucket:
        return {"distinct_ngrams": 0, "used_buckets": 0,
                "occupied_bucket_fraction": 0.0, "hash_collision_fraction": 0.0}
    ngrams = {ngram for word in model.wv.key_to_index
              for ngram in compute_ngrams_bytes(word, model.wv.min_n, model.wv.max_n)}
    used = len({ft_hash_bytes(ngram) % model.wv.bucket for ngram in ngrams})
    return {
        "distinct_ngrams": len(ngrams),
        "used_buckets": used,
        "occupied_bucket_fraction": used / model.wv.bucket,
        "hash_collision_fraction": 1 - used / len(ngrams) if ngrams else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/199801.txt"))
    parser.add_argument("--splits", type=Path, default=Path("outputs/default/splits.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ablations.json"))
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    args = parser.parse_args()
    validate_variants()

    articles = load_articles(args.data)
    selected, labels, _ = make_labels(articles)
    saved_splits = json.loads(args.splits.read_text(encoding="utf-8"))
    selected_ids = {article.article_id for article in selected}
    if set(saved_splits) != selected_ids:
        raise ValueError("Saved splits do not match the labeled documents; rerun run_experiments.py")
    indices = {part: [i for i, article in enumerate(selected) if saved_splits[article.article_id] == part]
               for part in ("train", "validation", "test")}
    if any(not values for values in indices.values()):
        raise ValueError("All train, validation and test splits must be nonempty")
    held_out = {article.article_id for article in selected if saved_splits[article.article_id] != "train"}
    embedding_docs = [list(article.body) for article in articles
                      if article.article_id not in held_out and len(article.body) >= 5]
    docs = [list(article.body) for article in selected]
    evaluation_words = [word for i in indices["validation"] for word in docs[i]]

    rows = []
    for name in args.variants:
        config = BASELINE | VARIANTS[name]
        begin = perf_counter()
        model = FastText(sentences=embedding_docs, **config)
        train_seconds = perf_counter() - begin
        vectors = mean_vectors(docs, model.wv, config["vector_size"])
        out_of_vocab = [word for word in evaluation_words if word not in model.wv.key_to_index]
        available_oov = sum(word in model.wv for word in out_of_vocab)
        row = {
            "name": name, "change_from_baseline": VARIANTS[name], "parameters": config,
            "train_seconds": round(train_seconds, 3), "vocabulary_size": len(model.wv),
            "estimated_training_memory_bytes": model.estimate_memory()["total"],
            "validation_oov_token_count": len(out_of_vocab),
            "validation_oov_vector_coverage": available_oov / len(out_of_vocab) if out_of_vocab else None,
            "bucket_statistics": bucket_statistics(model),
            "validation": validation_scores(vectors, labels, indices),
        }
        rows.append(row)
        print(f"{name}: 3-class F1={row['validation']['3_classes']['macro_f1']:.3f}, "
              f"5-class F1={row['validation']['5_classes']['macro_f1']:.3f}, "
              f"train={train_seconds:.1f}s", flush=True)
    output = {
        "protocol": "One factor at a time; same corpus and article splits; validation only; no test-set selection.",
        "split_file": str(args.splits), "selected_articles": len(selected),
        "class_counts": dict(Counter(labels)), "unsupervised_train_articles": len(embedding_docs),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
