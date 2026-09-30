"""Parse the segmented People's Daily 1998-01 corpus into articles."""

from dataclasses import dataclass
from pathlib import Path
import re

_ID = re.compile(r"^(\d{8}-\d{2}-\d{3})-(\d{3})/\S+$")
_PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)


@dataclass(frozen=True)
class Article:
    article_id: str
    title: str
    body: tuple[str, ...]


def parse_piece(piece: str) -> tuple[str, str] | None:
    """Return surface form and POS, including bracketed named entities."""
    if "/" not in piece:
        return None
    word, pos = piece.rsplit("/", 1)
    word = word.lstrip("[").rstrip("]")
    return (word, pos) if word else None


def content_tokens(line: str) -> list[str]:
    tokens = []
    for part in line.split():
        parsed = parse_piece(part)
        if parsed is None:
            continue
        word, pos = parsed
        if pos.lower().startswith("w") or _PUNCT.fullmatch(word):
            continue
        if word.isdigit():
            continue
        tokens.append(word)
    return tokens


def title_text(line: str) -> str:
    words = []
    for part in line.split():
        parsed = parse_piece(part)
        if parsed:
            words.append(parsed[0])
    return "".join(words)


def load_articles(path: Path) -> list[Article]:
    groups: dict[str, list[tuple[int, str]]] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            pieces = line.strip().split(maxsplit=1)
            if not pieces:
                continue
            match = _ID.fullmatch(pieces[0])
            if not match:
                raise ValueError(f"Invalid article ID on line {number}: {pieces[0]}")
            groups.setdefault(match[1], []).append((int(match[2]), pieces[1] if len(pieces) > 1 else ""))
    articles = []
    for article_id, rows in groups.items():
        rows.sort()
        title = title_text(rows[0][1])
        body = tuple(token for _, line in rows[1:] for token in content_tokens(line))
        articles.append(Article(article_id, title, body))
    return articles
