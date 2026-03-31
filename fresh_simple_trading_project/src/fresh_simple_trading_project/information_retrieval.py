from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from itertools import zip_longest
import logging
import re
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree

from .agents import NewsResearchAgent
from .llm import TextGenerationClient
from .models import NewsArticle, RetrievalResult

logger = logging.getLogger(__name__)

POSITIVE_KEYWORDS = {"beat", "growth", "surge", "upgrade", "approval", "profit", "buyback"}
NEGATIVE_KEYWORDS = {"downgrade", "lawsuit", "fraud", "miss", "recall", "probe", "bankruptcy"}
NEWS_SEARCH_FOCUS = [
    "symbol-specific news",
    "American economy developments",
    "overall U.S. stock market trend",
]
SYMBOL_NEWS_QUERY_TEMPLATE = "{symbol} stock market news"
MACRO_NEWS_QUERY = "American economy inflation interest rates jobs GDP Federal Reserve"
MARKET_NEWS_QUERY = "overall U.S. stock market trend S&P 500 Nasdaq Dow market breadth"
MACRO_KEYWORDS = {
    "american economy",
    "economy",
    "inflation",
    "interest rate",
    "rates",
    "jobs",
    "employment",
    "gdp",
    "federal reserve",
    "fed",
    "cpi",
    "pce",
}
MARKET_KEYWORDS = {
    "stock market",
    "s&p 500",
    "nasdaq",
    "dow",
    "market breadth",
    "equities",
    "wall street",
    "rally",
    "selloff",
}
MARKET_PROXY_SYMBOLS = ["SPY", "QQQ", "DIA", "IWM"]


class NewsSearchClient(Protocol):
    def search_news(self, query: str, limit: int = 10) -> list[NewsArticle]:
        ...


@dataclass
class StaticNewsSearchClient:
    articles: list[NewsArticle]

    def search_news(self, query: str, limit: int = 10) -> list[NewsArticle]:
        return list(self.articles[:limit])


@dataclass
class AlpacaNewsSearchClient:
    service: object
    max_age_days: int = 7

    def search_news(self, query: str, limit: int = 10) -> list[NewsArticle]:
        try:
            if query == MACRO_NEWS_QUERY:
                articles = self.service.fetch_news(
                    days=self.max_age_days,
                    limit=max(limit * 4, 25),
                )
                return _filter_articles_by_keywords(articles, MACRO_KEYWORDS, limit=limit)

            if query == MARKET_NEWS_QUERY:
                articles = self.service.fetch_news(
                    symbols=MARKET_PROXY_SYMBOLS,
                    days=self.max_age_days,
                    limit=max(limit * 4, 25),
                )
                return _filter_articles_by_keywords(articles, MARKET_KEYWORDS, limit=limit)

            symbol = _extract_symbol_from_query(query)
            if not symbol:
                return []
            return self.service.fetch_recent_news(symbol, days=self.max_age_days, limit=limit)
        except Exception as exc:  # pragma: no cover - external API behavior varies by environment.
            logger.warning("Alpaca news search failed for query %r: %s", query, exc)
            return []


@dataclass
class WebSearchNewsClient:
    endpoint: str = "https://news.google.com/rss/search"
    max_age_days: int = 7
    locale: str = "en-US"
    geography: str = "US"
    edition: str = "US:en"

    def search_news(self, query: str, limit: int = 10) -> list[NewsArticle]:
        params = urlencode(
            {
                "q": f"{query} when:{max(1, self.max_age_days)}d",
                "hl": self.locale,
                "gl": self.geography,
                "ceid": self.edition,
            }
        )
        try:
            with urlopen(f"{self.endpoint}?{params}", timeout=20) as response:  # nosec B310
                payload = response.read()
            root = ElementTree.fromstring(payload)
        except Exception as exc:  # pragma: no cover - external API behavior varies by environment.
            logger.warning("Live web news search failed for query %r: %s", query, exc)
            return []

        results: list[NewsArticle] = []
        for item in root.findall("./channel/item"):
            raw_title = str(item.findtext("title", "")).strip()
            headline, fallback_source = _split_google_news_title(raw_title)
            source_element = item.find("source")
            source = (
                str(source_element.text).strip()
                if source_element is not None and source_element.text
                else fallback_source
            )
            results.append(
                NewsArticle(
                    headline=headline,
                    summary=_clean_html(item.findtext("description", "")),
                    source=source,
                    url=str(item.findtext("link", "")).strip(),
                    published_at=_parse_rfc2822(item.findtext("pubDate")),
                )
            )
            if len(results) >= limit:
                break
        return [article for article in results if article.headline]


@dataclass
class CombinedNewsSearchClient:
    clients: list[NewsSearchClient]

    def search_news(self, query: str, limit: int = 10) -> list[NewsArticle]:
        merged: list[NewsArticle] = []
        seen: set[str] = set()
        for client in self.clients:
            for article in client.search_news(query, limit=limit):
                key = _article_key(article)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(article)
                if len(merged) >= limit:
                    return merged
        return merged


@dataclass
class InformationRetrievalModule:
    news_client: NewsSearchClient
    llm_client: TextGenerationClient | None = None
    news_agent: NewsResearchAgent | None = None
    max_article_age_days: int = 7

    def __post_init__(self) -> None:
        if self.news_agent is None and self.llm_client is not None:
            self.news_agent = NewsResearchAgent(llm_client=self.llm_client)

    def retrieve(self, symbol: str, limit: int = 10) -> RetrievalResult:
        queries = _build_news_queries(symbol)
        per_query_limit = max(3, limit)
        article_batches = [
            _filter_recent_articles(
                self.news_client.search_news(query, limit=per_query_limit),
                max_age_days=self.max_article_age_days,
            )
            for query in queries
        ]
        articles = _interleave_unique_articles(article_batches, max_items=limit)
        positive_hits = 0
        negative_hits = 0
        risk_flags: list[str] = []
        catalysts: list[str] = []
        headlines: list[str] = []

        for article in articles:
            body = f"{article.headline} {article.summary}".lower()
            positive_hits += sum(keyword in body for keyword in POSITIVE_KEYWORDS)
            negative_hits += sum(keyword in body for keyword in NEGATIVE_KEYWORDS)
            headlines.append(article.headline)
            if "earnings" in body and "earnings" not in catalysts:
                catalysts.append("earnings")
            if "upgrade" in body and "analyst upgrade" not in catalysts:
                catalysts.append("analyst upgrade")
            if "downgrade" in body and "analyst downgrade" not in catalysts:
                catalysts.append("analyst downgrade")

        total_hits = positive_hits + negative_hits
        sentiment = 0.0 if total_hits == 0 else round((positive_hits - negative_hits) / total_hits, 2)
        if negative_hits > positive_hits:
            risk_flags.append("Headline sentiment is net negative.")
        if any(keyword in " ".join(headlines).lower() for keyword in {"lawsuit", "fraud", "bankruptcy"}):
            risk_flags.append("At least one headline contains a high-risk keyword.")
        summary_note = self._summarize_with_llm(
            symbol=symbol,
            articles=articles[:limit],
            sentiment=sentiment,
            risk_flags=risk_flags,
            catalysts=catalysts,
            search_queries=queries,
        )

        return RetrievalResult(
            symbol=symbol,
            articles=articles,
            headline_summary=headlines[:limit],
            sentiment_score=sentiment,
            risk_flags=risk_flags,
            catalysts=catalysts,
            summary_note=summary_note,
        )

    def _summarize_with_llm(
        self,
        *,
        symbol: str,
        articles: list[NewsArticle],
        sentiment: float,
        risk_flags: list[str],
        catalysts: list[str],
        search_queries: list[str],
    ) -> str | None:
        if self.news_agent is None:
            return None

        return self.news_agent.summarize(
            symbol=symbol,
            articles=articles,
            sentiment=sentiment,
            risk_flags=risk_flags,
            catalysts=catalysts,
            search_queries=search_queries,
        )


def _build_news_queries(symbol: str) -> list[str]:
    return [
        SYMBOL_NEWS_QUERY_TEMPLATE.format(symbol=symbol),
        MACRO_NEWS_QUERY,
        MARKET_NEWS_QUERY,
    ]


def _interleave_unique_articles(article_batches: list[list[NewsArticle]], max_items: int) -> list[NewsArticle]:
    merged: list[NewsArticle] = []
    seen: set[str] = set()
    for row in zip_longest(*article_batches):
        for article in row:
            if article is None:
                continue
            key = _article_key(article)
            if key in seen:
                continue
            seen.add(key)
            merged.append(article)
            if len(merged) >= max_items:
                return merged
    return merged


def _filter_recent_articles(articles: list[NewsArticle], max_age_days: int) -> list[NewsArticle]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, max_age_days))
    recent: list[NewsArticle] = []
    for article in articles:
        published_at = _parse_published_at(article.published_at)
        if published_at is not None and published_at < cutoff:
            continue
        recent.append(article)
    return recent


def _parse_published_at(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _article_key(article: NewsArticle) -> str:
    headline = article.headline.strip().lower()
    source = article.source.strip().lower()
    if headline and source:
        return f"{headline} | {source}"
    url = article.url.strip()
    if url:
        return url.lower()
    return " | ".join(
        [
            headline,
            source,
            str(article.published_at or "").strip().lower(),
        ]
    )


def _extract_symbol_from_query(query: str) -> str:
    if not query.strip():
        return ""
    return query.strip().split()[0].upper()


def _filter_articles_by_keywords(
    articles: list[NewsArticle],
    keywords: set[str],
    *,
    limit: int,
) -> list[NewsArticle]:
    matches: list[NewsArticle] = []
    for article in articles:
        body = f"{article.headline} {article.summary}".lower()
        if any(keyword in body for keyword in keywords):
            matches.append(article)
            if len(matches) >= limit:
                return matches
    return matches or articles[:limit]


def _clean_html(raw: str | None) -> str:
    if raw is None:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)
    return " ".join(unescape(text).split())


def _split_google_news_title(title: str) -> tuple[str, str]:
    if " - " not in title:
        return title, ""
    headline, source = title.rsplit(" - ", 1)
    return headline.strip(), source.strip()


def _parse_rfc2822(raw: str | None) -> str | None:
    if raw is None:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    try:
        timestamp = parsedate_to_datetime(candidate)
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.isoformat().replace("+00:00", "Z")
