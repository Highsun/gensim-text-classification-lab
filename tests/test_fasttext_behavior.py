import numpy as np
import pytest
from gensim.models import FastText, Word2Vec


def test_subwords_enable_oov_vector_but_word2vec_does_not():
    corpus = [["新闻", "记者", "报道"], ["记者", "采访", "新闻"]]
    common = dict(vector_size=8, min_count=1, epochs=2, workers=1, seed=42)
    fasttext = FastText(sentences=corpus, min_n=2, max_n=4, bucket=1000, **common)
    word2vec = Word2Vec(sentences=corpus, **common)
    unseen = "新闻记者"
    assert unseen not in fasttext.wv.key_to_index
    assert np.isfinite(fasttext.wv[unseen]).all()
    assert np.linalg.norm(fasttext.wv[unseen]) > 0
    with pytest.raises(KeyError):
        _ = word2vec.wv[unseen]


def test_disabling_subwords_disables_fasttext_oov_vector():
    model = FastText(sentences=[["新闻", "记者"]], vector_size=8, min_count=1,
                     min_n=1, max_n=0, epochs=1, workers=1, seed=42)
    assert model.wv.bucket == 0
    with pytest.raises(KeyError):
        _ = model.wv["新闻记者"]
