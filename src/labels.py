"""Auditable, title-derived weak topic labels; no gold labels are provided."""

from collections import Counter
from .corpus import Article

KEYWORDS = {
    "economy": ("经济", "企业", "市场", "金融", "银行", "投资", "贸易", "股市", "工业"),
    "sports": ("比赛", "足球", "篮球", "奥运", "运动员", "世界杯", "体育", "乒乓", "排球", "网球"),
    "culture": ("文化", "文学", "艺术", "音乐", "电影", "电视", "戏剧", "出版", "书法"),
    "science": ("科技", "科学", "科研", "电脑", "计算机", "航天", "卫星", "互联网"),
    "international": ("美国", "俄罗斯", "联合国", "日本", "韩国", "印度", "巴基斯坦", "法国", "英国"),
}
STAGES = ("economy", "sports", "culture"), tuple(KEYWORDS)


def make_labels(articles: list[Article], min_body_tokens: int = 30) -> tuple[list[Article], list[str], Counter]:
    accepted, labels = [], []
    excluded = Counter()
    for article in articles:
        if len(article.body) < min_body_tokens:
            excluded["short_body"] += 1
            continue
        hits = [label for label, words in KEYWORDS.items() if any(word in article.title for word in words)]
        if len(hits) != 1:
            excluded["ambiguous" if hits else "unlabeled"] += 1
            continue
        accepted.append(article)
        labels.append(hits[0])
    return accepted, labels, excluded
