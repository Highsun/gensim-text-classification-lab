"""Reproducible representation learning and weak-label classification pipeline."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.corpus import load_articles
from src.labels import KEYWORDS, STAGES, make_labels


def mean_vectors(docs, keyed_vectors, size):
    result = np.zeros((len(docs), size), dtype=np.float32)
    for row, doc in enumerate(docs):
        vectors = [keyed_vectors[word] for word in doc if word in keyed_vectors]
        if vectors:
            result[row] = np.mean(vectors, axis=0)
    return result


def score(name, model, matrix, labels, indices):
    train, val, test = indices
    x_train, x_val, x_test = (matrix[i] for i in indices)
    y_train, y_val, y_test = (np.asarray(labels)[i] for i in indices)
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    return {
        "method": name,
        "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        "validation_macro_f1": float(f1_score(y_val, val_pred, average="macro")),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_macro_f1": float(f1_score(y_test, test_pred, average="macro")),
        "test_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/199801.txt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/default"))
    parser.add_argument("--vector-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--bucket", type=int, default=100000)
    parser.add_argument("--min-n", type=int, default=2)
    parser.add_argument("--max-n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    articles = load_articles(args.data)
    selected, labels, excluded = make_labels(articles)
    if min(Counter(labels).values()) < 10:
        raise ValueError(f"Not enough documents per class: {Counter(labels)}")
    all_ids = np.arange(len(selected))
    train_val, test = train_test_split(all_ids, test_size=0.2, random_state=args.seed, stratify=labels)
    train, val = train_test_split(train_val, test_size=0.25, random_state=args.seed, stratify=np.asarray(labels)[train_val])
    split = {selected[i].article_id: part for part, ids in (("train", train), ("validation", val), ("test", test)) for i in ids}
    (args.output / "splits.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    # All articles except the held-out labeled articles can train unsupervised representations.
    held_out = {selected[i].article_id for i in np.concatenate((val, test))}
    embedding_docs = [list(a.body) for a in articles if a.article_id not in held_out and len(a.body) >= 5]
    docs = [list(a.body) for a in selected]
    text_docs = [" ".join(doc) for doc in docs]
    train_text = [text_docs[i] for i in train]
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=2, max_df=0.95, sublinear_tf=True)
    x_train = tfidf.fit_transform(train_text)
    x_all = tfidf.transform(text_docs)
    features = {"tfidf": x_all}

    common = dict(vector_size=args.vector_size, window=args.window, min_count=args.min_count,
                  epochs=args.epochs, workers=args.workers, seed=args.seed, sg=1)
    word2vec = models.Word2Vec(sentences=embedding_docs, **common)
    fasttext = models.FastText(sentences=embedding_docs, bucket=args.bucket,
                               min_n=args.min_n, max_n=args.max_n, **common)
    word2vec.save(str(args.output / "word2vec.model"))
    fasttext.save(str(args.output / "fasttext.model"))
    features["word2vec_mean"] = mean_vectors(docs, word2vec.wv, args.vector_size)
    features["fasttext_mean"] = mean_vectors(docs, fasttext.wv, args.vector_size)

    dictionary = corpora.Dictionary(docs[i] for i in train)
    dictionary.filter_extremes(no_below=args.min_count, no_above=0.95)
    bow = [dictionary.doc2bow(doc) for doc in docs]
    train_bow = [bow[i] for i in train]
    gensim_tfidf = models.TfidfModel(train_bow, dictionary=dictionary)
    lsi_size = min(args.vector_size, len(dictionary) - 1, len(train) - 1)
    if lsi_size >= 2:
        lsi = models.LsiModel(gensim_tfidf[train_bow], id2word=dictionary, num_topics=lsi_size)
        lsi_features = np.zeros((len(docs), lsi_size), dtype=np.float32)
        for row, doc_bow in enumerate(bow):
            for topic, weight in lsi[gensim_tfidf[doc_bow]]:
                lsi_features[row, topic] = weight
        features["gensim_lsi"] = lsi_features
    probe = next((w for w in ("经济", "北京", "中国") if w in fasttext.wv.key_to_index), None)
    info = {
        "articles_total": len(articles), "labeled": len(selected), "excluded": dict(excluded),
        "class_counts": dict(Counter(labels)), "split_counts": {p: dict(Counter(labels[i] for i in ids)) for p, ids in (("train", train), ("validation", val), ("test", test))},
        "unsupervised_train_articles": len(embedding_docs), "vocabulary": len(dictionary),
        "example_bow_nonzero": len(bow[0]), "example_tfidf_nonzero": len(gensim_tfidf[bow[0]]),
        "fasttext_vocab": len(fasttext.wv), "word2vec_vocab": len(word2vec.wv),
        "probe": probe,
        "fasttext_neighbors": fasttext.wv.most_similar(probe, topn=5) if probe else [],
        "word2vec_neighbors": word2vec.wv.most_similar(probe, topn=5) if probe else [],
        "fasttext_oov_vector_available": bool("经济学家们" in fasttext.wv),
        "word2vec_oov_vector_available": bool("经济学家们" in word2vec.wv),
    }
    results = {}
    for stage in STAGES:
        keep = set(stage)
        stage_indices = tuple(np.asarray([i for i in ids if labels[i] in keep]) for ids in (train, val, test))
        results[str(len(stage)) + "_classes"] = [
            score(name, LogisticRegression(max_iter=1000, random_state=args.seed), matrix, labels, stage_indices)
            for name, matrix in features.items()
        ]
    payload = {"parameters": vars(args) | {"data": str(args.data), "output": str(args.output)},
               "labels": {k: list(v) for k, v in KEYWORDS.items()}, "dataset": info, "results": results}
    (args.output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for stage, rows in results.items():
        print(stage, *(f"{r['method']}: macro-F1={r['test_macro_f1']:.3f}" for r in rows), sep=" | ")
    print("Saved", args.output / "metrics.json")


if __name__ == "__main__":
    main()
