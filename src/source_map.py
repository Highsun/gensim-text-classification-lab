"""Print installed Gensim source locations for paper-to-code inspection."""

import inspect
from gensim.models import FastText
from gensim.models.fasttext import FastTextKeyedVectors


def main() -> None:
    targets = (FastText.build_vocab, FastText.train, FastText._do_train_job,
               FastTextKeyedVectors.recalc_char_ngram_buckets, FastTextKeyedVectors.get_vector)
    for target in targets:
        print(f"{target.__qualname__}: {inspect.getsourcefile(target)}:{inspect.getsourcelines(target)[1]}")
    print("Inner training loops are compiled Cython; inspect gensim/models/fasttext_inner.pyx in the Gensim source tree.")


if __name__ == "__main__":
    main()
