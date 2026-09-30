# Gensim FastText implementation guide

This is a developer reading guide for the installed Gensim 4.3.3 implementation. It documents which components this repository calls and where to inspect their behavior. Run `python -m src.source_map` to print line numbers for the actual installed package; the Cython source is available as `gensim/models/fasttext_inner.pyx` in the Conda environment or [upstream repository](https://github.com/piskvorky/gensim/blob/4.3.3/gensim/models/fasttext_inner.pyx).

## Two related FastText tasks

| Reference | Objective | Mapping here |
| --- | --- | --- |
| [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) | Learn word representations from context, using character n-grams | `gensim.models.FastText` with `sg=1`; its vectors feed the classifier |
| [Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068/) | Learn a supervised text classifier, including word n-gram features and a label prediction layer | Only the idea of an aggregated document vector is used; scikit-learn trains the separate label classifier |

Gensim 4.3.3 rejects `word_ngrams != 1` in `FastText.__init__`. Its `min_n` and `max_n` refer to **character** n-grams within a word, not the **word** n-grams in the classification paper. Setting `hs=1` in Gensim changes the *word embedding training objective*; it does not add the classification paper's hierarchical label softmax. These are distinct mechanisms even though they share terminology.

## Source trace

1. `FastText.__init__` (`fasttext.py`) validates `word_ngrams`, sets up `FastTextKeyedVectors`, and delegates shared options to `Word2Vec.__init__`.
2. `Word2Vec.build_vocab` scans word counts and applies `min_count`. FastText's `FastTextKeyedVectors.recalc_char_ngram_buckets` creates per-word hashed subword indices. `FastText.estimate_memory` accounts for the vocabulary and `bucket × vector_size` subword storage.
3. `Word2Vec.train` schedules batches, epochs and learning-rate decay. `FastText._do_train_job` calls the compiled `train_batch_any` in `fasttext_inner.pyx`. With `sg=1, hs=0, negative=5`, the skip-gram negative-sampling path updates word, subword and output vectors.
4. `ft_ngram_hashes` (`fasttext.py`) calls `compute_ngrams_bytes` and `ft_hash_bytes` (`fasttext_inner.pyx`), then takes the hash modulo `bucket`. The Cython function adds `<` and `>` boundaries and walks UTF-8 character starts. This matters for Chinese: an n-gram length counts characters rather than raw UTF-8 bytes.
5. `FastTextKeyedVectors.adjust_vectors` composes stored in-vocabulary word and subword vectors after training. `__contains__` checks whether a vector can be returned. `get_vector` returns a stored composite vector for an in-vocabulary word and builds an OOV vector from hashed subwords otherwise.
6. `_save_specials` and `_load_specials` handle derived vector caches, hash compatibility and cache reconstruction on model serialization. The project saves a full `.model` so training may be resumed or vectors inspected later.

The [subword paper](https://arxiv.org/abs/1607.04606) describes a word through its character n-gram representations and a context-prediction objective. Gensim's optimized Cython routines include implementation-specific scaling and caches, so the formulas in the paper should be compared with the actual training and inference paths rather than read as line-for-line pseudocode. `FastTextKeyedVectors.get_vector` can return an OOV vector when hashes exist; that alone does **not** prove its semantic quality. A very small bucket can also collide many unrelated n-grams.

## Parameter map

| Parameter | Source role | Practical choice or check |
| --- | --- | --- |
| `vector_size` | Width of word, subword and output vectors | 64 is the project baseline; 128 tests capacity against memory and runtime |
| `window` | Maximum context distance in training | 5 is baseline; 2 tests more local contexts |
| `epochs` | Corpus passes and training schedule | 5 is baseline; 10 tests more optimization without changing data |
| `min_count` | Excludes rare whole words from vocabulary | 2 is baseline; 5 tests smaller vocabulary and changed OOV rate |
| `min_n`, `max_n` | Character subword lengths | 2–4 is baseline; 3–6 changes subword granularity; `min_n=1, max_n=0` disables subwords |
| `bucket` | Modulus for hashed subword IDs | 100000 is baseline; 10000 tests collision and memory tradeoffs |
| `sg` | `1` = skip-gram, `0` = CBOW for embedding training | Skip-gram is baseline; CBOW is a controlled architecture variant |
| `negative`, `hs` | Word embedding training objective | Baseline uses five negative samples and `hs=0`; neither is the supervised label loss |
| `sample`, `shrink_windows` | Frequent-word downsampling and random reduction of effective window | Kept at Gensim defaults across ablations to isolate the chosen factor |
| `workers`, `seed` | Thread scheduling and initialization | One worker and fixed seed reduce variation; use `PYTHONHASHSEED=0` for repeated runs |

## Evaluation boundary

`run_experiments.py` trains TF-IDF and LSI only on labeled training articles. Word2Vec and FastText train without labels on those training articles plus articles not selected for classification; labeled validation and test articles are excluded. The title supplies weak labels but is not a classifier feature. `run_ablations.py` reads the same saved article split, changes one parameter group at a time, and reports **validation** metrics; it does not use test results to choose settings. This is an exploratory comparison on one random split and weak labels, not a statistical claim that one setting is universally better.

The chosen Gensim version's LSI code needs SciPy below 1.14 because it accesses `scipy.sparse.sparsetools.csc_matvecs`. This compatibility constraint is pinned in `environment.yml` and `requirements.txt`.
