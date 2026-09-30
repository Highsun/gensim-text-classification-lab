"""Print installed Gensim source locations for paper-to-code inspection."""

import inspect
from pathlib import Path
import re
from gensim.models import FastText
from gensim.models.fasttext import FastTextKeyedVectors, ft_ngram_hashes


def main() -> None:
    targets = (FastText.__init__, FastText.build_vocab, FastText.train,
               FastText._do_train_job, FastText.estimate_memory,
               FastTextKeyedVectors.recalc_char_ngram_buckets,
               FastTextKeyedVectors.adjust_vectors,
               FastTextKeyedVectors.__contains__,
               FastTextKeyedVectors.get_vector,
               FastTextKeyedVectors._save_specials,
               FastTextKeyedVectors._load_specials, ft_ngram_hashes)
    for target in targets:
        print(f"{target.__qualname__}: {inspect.getsourcefile(target)}:{inspect.getsourcelines(target)[1]}")
    pyx = Path(inspect.getsourcefile(FastText)).with_name("fasttext_inner.pyx")
    if pyx.exists():
        for name in ("train_batch_any", "compute_ngrams_bytes", "ft_hash_bytes"):
            line = next((n for n, text in enumerate(pyx.read_text(encoding="utf-8").splitlines(), 1)
                         if re.match(rf"^(?:cpdef|def)\s+{name}\b", text)), None)
            if line:
                print(f"{name}: {pyx}:{line}")
    else:
        print("Cython source not installed; see gensim/models/fasttext_inner.pyx in the Gensim source repository.")


if __name__ == "__main__":
    main()
