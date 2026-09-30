from src.corpus import content_tokens, load_articles, parse_piece
from src.labels import make_labels


def test_bracketed_entity_and_article_boundary(tmp_path):
    assert parse_piece("[国务院/nt") == ("国务院", "nt")
    assert parse_piece("侨办/j]nt") == ("侨办", "j]nt")
    data = tmp_path / "sample.txt"
    data.write_text(
        "19980101-01-001-001/m  足球/n  比赛/n\n"
        "19980101-01-001-002/m  [中国/ns  队/n]nt  赢/v  了/u  。/w\n"
        "19980101-01-002-001/m  银行/n  投资/v\n"
        "19980101-01-002-002/m  市场/n  变化/v  。/w\n",
        encoding="utf-8",
    )
    articles = load_articles(data)
    assert len(articles) == 2
    assert articles[0].title == "足球比赛"
    assert articles[0].body == ("中国", "队", "赢", "了")
    accepted, labels, _ = make_labels(articles, min_body_tokens=2)
    assert [a.article_id for a in accepted] == [a.article_id for a in articles]
    assert labels == ["sports", "economy"]


def test_punctuation_and_untagged_are_dropped():
    assert content_tokens("２００/m  ，/w  新闻/n  bad  ！/w") == ["新闻"]
