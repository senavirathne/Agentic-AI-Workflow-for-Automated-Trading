"""News retrieval, filtering, and summarization for the trading workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from itertools import zip_longest
import logging
import re
import requests
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree

from .agents import NewsResearchAgent
from .models import NewsArticle, RetrievalResult
from .storage import ResultStore
from .utils import _as_string

logger = logging.getLogger(__name__)

NEWS_SEARCH_FOCUS = [
    "symbol-specific news",
    "American economy developments",
    "overall U.S. stock market trend",
]
SYMBOL_NEWS_QUERY_TEMPLATE = "{symbol} stock market news"
MACRO_NEWS_QUERY = "American economy inflation interest rates jobs GDP Federal Reserve"
MARKET_NEWS_QUERY = "overall U.S. stock market trend S&P 500 Nasdaq Dow market breadth"
NEWS_AGENT_MAX_INPUT_CHARS = 12_000
NEWS_AGENT_SEARCH_LIMIT_PER_QUERY = 40
NEWS_AGENT_MAX_CRITICAL_ITEMS = 3
NEWS_AGENT_MAX_SUMMARY_CHARS = 320
NEWS_AGENT_MAX_CRITICAL_ITEM_CHARS = 180
ALPHA_VANTAGE_NEWS_FUNCTION = "NEWS_SENTIMENT"
ALPHA_VANTAGE_NEWS_MAX_LIMIT = 1000
ALPHA_VANTAGE_TIMEOUT_SECONDS = 30


class NewsSearchClient(Protocol):
    """Protocol for news providers that return article lists."""

    def search_news(
        self,
        query: str,
        limit: int = 10,
        *,
        input_size_chars: int | None = None,
    ) -> list[NewsArticle]:
        """Return news articles for the supplied query."""
        ...


@dataclass
class StaticNewsSearchClient:
    """Fixed news provider used by tests and local demos."""

    articles: list[NewsArticle]

    def search_news(self, query: str, limit: int = 10, *, input_size_chars: int | None = None) -> list[NewsArticle]:
        """Return the configured static article subset."""

        del input_size_chars
        return list(self.articles[:limit])


@dataclass
class AlphaVantageNewsSearchClient:
    """News provider backed by Alpha Vantage's sentiment/news endpoint."""

    api_key: str
    base_url: str = "https://www.alphavantage.co/query"
    max_age_days: int = 7
    http_get: object = requests.get
    _last_request_succeeded: bool = field(default=False, init=False, repr=False)
    _last_request_cacheable: bool = field(default=False, init=False, repr=False)

    def search_news(
        self,
        query: str,
        limit: int = 10,
        *,
        input_size_chars: int | None = None,
    ) -> list[NewsArticle]:
        """Search Alpha Vantage news and normalize the feed into articles."""

        self._last_request_succeeded = False
        self._last_request_cacheable = False
        try:
            params = self._build_query_params(query, limit=limit, input_size_chars=input_size_chars)
            if params is None:
                return []
            response = self.http_get(self.base_url, params=params, timeout=ALPHA_VANTAGE_TIMEOUT_SECONDS)
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            payload = response.json()
            if "Error Message" in payload:
                raise ValueError(payload["Error Message"])
            if "Note" in payload or "Information" in payload:
                self._last_request_cacheable = True
                logger.info(
                    "Alpha Vantage news request returned an informational/rate-limit response for query %r. "
                    "Reusing any available cached news.",
                    query,
                )
                return []
            self._last_request_succeeded = True
            self._last_request_cacheable = True
            return _parse_alpha_vantage_news_feed(payload, limit=limit)
        except Exception as exc:  # pragma: no cover - external API behavior varies by environment.
            logger.warning("Alpha Vantage news search failed for query %r: %s", query, exc)
            return []

    @property
    def last_request_succeeded(self) -> bool:
        """Report whether the most recent Alpha Vantage request completed successfully."""

        return self._last_request_succeeded

    @property
    def last_request_cacheable(self) -> bool:
        """Report whether the most recent Alpha Vantage response should suppress same-day refetches."""

        return self._last_request_cacheable

    def _build_query_params(
        self,
        query: str,
        *,
        limit: int,
        input_size_chars: int | None,
    ) -> dict[str, str | int] | None:
        params: dict[str, str | int] = {
            "function": ALPHA_VANTAGE_NEWS_FUNCTION,
            "apikey": self.api_key,
            "sort": "LATEST",
            "limit": min(
                ALPHA_VANTAGE_NEWS_MAX_LIMIT,
                max(limit, _target_article_count_for_input_size(input_size_chars)),
            ),
        }
        if query == MACRO_NEWS_QUERY:
            params["topics"] = "economy_macro,economy_monetary,economy_fiscal"
            return params
        if query == MARKET_NEWS_QUERY:
            params["topics"] = "financial_markets"
            return params

        symbol = _extract_symbol_from_query(query)
        if not symbol:
            return None
        params["tickers"] = symbol
        return params


@dataclass
class WebSearchNewsClient:
    """News provider backed by public RSS and web-search feeds."""

    endpoint: str = "https://news.google.com/rss/search"
    max_age_days: int = 7
    locale: str = "en-US"
    geography: str = "US"
    edition: str = "US:en"

    def search_news(self, query: str, limit: int = 10, *, input_size_chars: int | None = None) -> list[NewsArticle]:
        """Search Google News RSS and normalize the resulting feed items."""

        del input_size_chars
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
                    provider="web_search",
                )
            )
            if len(results) >= limit:
                break
        return [article for article in results if article.headline]


@dataclass
class CombinedNewsSearchClient:
    """Merge multiple news providers into one aggregated client."""

    clients: list[NewsSearchClient]

    def search_news(self, query: str, limit: int = 10, *, input_size_chars: int | None = None) -> list[NewsArticle]:
        """Query each client and merge unique articles in provider order."""

        merged: list[NewsArticle] = []
        seen: set[str] = set()
        for client in self.clients:
            for article in client.search_news(query, limit=limit, input_size_chars=input_size_chars):
                key = _combined_article_key(article)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(article)
        return merged


@dataclass
class InformationRetrievalModule:
    """Fetch articles and compress them into trading-ready context."""

    news_client: NewsSearchClient
    news_agent: NewsResearchAgent | None = None
    news_archive: ResultStore | None = None
    max_article_age_days: int = 7

    def retrieve(
        self,
        symbol: str,
        limit: int = 10,
        *,
        input_size_chars: int | None = None,
        published_at_lte: datetime | str | None = None,
    ) -> RetrievalResult:
        """Retrieve and summarize symbol, macro, and market news context.

        Args:
            symbol: Ticker symbol to gather news for.
            limit: Maximum number of display headlines to keep.
            input_size_chars: Optional prompt budget for article selection.
            published_at_lte: Optional upper bound for backtest-safe news timestamps.

        Returns:
            A normalized retrieval result for downstream workflow stages.
        """

        target_input_chars = input_size_chars or NEWS_AGENT_MAX_INPUT_CHARS
        queries = _build_news_queries(symbol)
        per_query_limit = max(limit, NEWS_AGENT_SEARCH_LIMIT_PER_QUERY, _target_article_count_for_input_size(target_input_chars))
        article_batches = [
            self._load_query_articles(
                symbol,
                query,
                limit=per_query_limit,
                input_size_chars=target_input_chars,
                published_at_lte=published_at_lte,
            )
            for query in queries
        ]
        candidate_articles = _sort_articles_newest_first(
            _interleave_unique_articles(
                article_batches,
                max_items=max(limit * 4, per_query_limit * len(queries)),
            )
        )
        articles = _select_articles_for_news_agent(
            candidate_articles,
            max_input_chars=target_input_chars,
        )
        headlines = [article.headline for article in articles[:limit]]
        news_summary = self._summarize_with_llm(
            symbol=symbol,
            articles=articles,
            search_queries=queries,
        )
        summary_note = _compress_text(
            news_summary.summary_note if news_summary is not None else None,
            max_chars=NEWS_AGENT_MAX_SUMMARY_CHARS,
        )
        if summary_note is None:
            summary_note = _fallback_news_summary(articles)
        critical_news = _compress_text_list(
            news_summary.critical_news if news_summary and news_summary.critical_news else [],
            max_items=NEWS_AGENT_MAX_CRITICAL_ITEMS,
            max_chars_per_item=NEWS_AGENT_MAX_CRITICAL_ITEM_CHARS,
        )
        risk_flags = news_summary.risk_flags if news_summary and news_summary.risk_flags else []
        catalysts = news_summary.catalysts if news_summary and news_summary.catalysts else []
        display_headlines = critical_news if critical_news else headlines[:NEWS_AGENT_MAX_CRITICAL_ITEMS]

        return RetrievalResult(
            symbol=symbol,
            articles=articles,
            headline_summary=display_headlines,
            sentiment_score=0.0,
            critical_news=critical_news,
            risk_flags=risk_flags,
            catalysts=catalysts,
            summary_note=summary_note,
        )

    def _load_query_articles(
        self,
        symbol: str,
        query: str,
        *,
        limit: int,
        input_size_chars: int,
        published_at_lte: datetime | str | None,
    ) -> list[NewsArticle]:
        fetch_bucket = _news_fetch_bucket(published_at_lte)
        live_articles = _filter_articles_published_at_lte(
            _sort_articles_newest_first(
                self._search_live_news(
                    symbol,
                    query,
                    limit=limit,
                    input_size_chars=input_size_chars,
                    fetch_bucket=fetch_bucket,
                )
            ),
            published_at_lte=published_at_lte,
        )
        if self.news_archive is None:
            return _filter_articles_for_decision_context(live_articles, symbol, query=query)

        if live_articles:
            self.news_archive.save_retrieved_news(symbol, query, live_articles)
        cached_articles = self.news_archive.load_retrieved_news(
            symbol,
            query,
            limit=max(limit * 4, limit),
            published_at_lte=published_at_lte,
        )
        if cached_articles:
            return _sort_articles_newest_first(_filter_articles_for_decision_context(cached_articles, symbol, query=query))
        return _filter_articles_for_decision_context(live_articles, symbol, query=query)

    def _search_live_news(
        self,
        symbol: str,
        query: str,
        *,
        limit: int,
        input_size_chars: int,
        fetch_bucket: str,
    ) -> list[NewsArticle]:
        if isinstance(self.news_client, CombinedNewsSearchClient):
            merged: list[NewsArticle] = []
            seen: set[str] = set()
            for client in self.news_client.clients:
                for article in self._search_live_news_client(
                    client,
                    symbol,
                    query,
                    limit=limit,
                    input_size_chars=input_size_chars,
                    fetch_bucket=fetch_bucket,
                ):
                    key = _combined_article_key(article)
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(article)
            return merged
        return self._search_live_news_client(
            self.news_client,
            symbol,
            query,
            limit=limit,
            input_size_chars=input_size_chars,
            fetch_bucket=fetch_bucket,
        )

    def _search_live_news_client(
        self,
        client: NewsSearchClient,
        symbol: str,
        query: str,
        *,
        limit: int,
        input_size_chars: int,
        fetch_bucket: str,
    ) -> list[NewsArticle]:
        if isinstance(client, AlphaVantageNewsSearchClient) and self.news_archive is not None:
            if self.news_archive.has_news_query_fetch(
                symbol,
                query,
                provider="alpha_vantage",
                fetch_bucket=fetch_bucket,
            ):
                return []

        articles = client.search_news(
            query,
            limit=limit,
            input_size_chars=input_size_chars,
        )

        if (
            isinstance(client, AlphaVantageNewsSearchClient)
            and self.news_archive is not None
            and client.last_request_cacheable
        ):
            self.news_archive.save_news_query_fetch(
                symbol,
                query,
                provider="alpha_vantage",
                fetch_bucket=fetch_bucket,
            )
        return articles

    def _summarize_with_llm(
        self,
        *,
        symbol: str,
        articles: list[NewsArticle],
        search_queries: list[str],
    ):
        if self.news_agent is None:
            return None

        return self.news_agent.summarize(
            symbol=symbol,
            articles=articles,
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


def _sort_articles_newest_first(articles: list[NewsArticle]) -> list[NewsArticle]:
    return sorted(
        articles,
        key=lambda article: _parse_published_at(article.published_at) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )


def _select_articles_for_news_agent(
    articles: list[NewsArticle],
    *,
    max_input_chars: int,
) -> list[NewsArticle]:
    if not articles:
        return []

    selected: list[NewsArticle] = []
    used_chars = 0
    for article in articles:
        article_text = _news_agent_article_line(article)
        article_chars = len(article_text)
        if selected and used_chars + article_chars > max_input_chars:
            break
        selected.append(article)
        used_chars += article_chars
    return selected


def _target_article_count_for_input_size(input_size_chars: int | None) -> int:
    if input_size_chars is None:
        return 50
    # Assume roughly 400 chars per article line in the LLM prompt.
    return max(25, min(ALPHA_VANTAGE_NEWS_MAX_LIMIT, input_size_chars // 400))


def _parse_alpha_vantage_news_feed(payload: dict, *, limit: int) -> list[NewsArticle]:
    feed = payload.get("feed")
    if not isinstance(feed, list):
        return []

    articles: list[NewsArticle] = []
    for item in feed:
        if not isinstance(item, dict):
            continue
        primary_ticker, primary_relevance = _extract_primary_ticker(item)
        articles.append(
            NewsArticle(
                headline=_as_string(item.get("title")) or "",
                summary=_as_string(item.get("summary")) or "",
                source=_as_string(item.get("source")) or _as_string(item.get("source_domain")) or "",
                url=_as_string(item.get("url")) or "",
                published_at=_parse_alpha_vantage_time_published(item.get("time_published")),
                provider="alpha_vantage",
                primary_ticker=primary_ticker,
                primary_ticker_relevance=primary_relevance,
            )
        )
        if len(articles) >= limit:
            break
    return [article for article in articles if article.headline]


def _news_agent_article_line(article: NewsArticle) -> str:
    return (
        f"- {article.headline} | {article.summary} | source={article.source} | "
        f"published_at={article.published_at}\n"
    )


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


def _parse_alpha_vantage_time_published(raw: object) -> str | None:
    candidate = _as_string(raw)
    if candidate is None:
        return None
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            parsed = datetime.strptime(candidate, fmt).replace(tzinfo=timezone.utc)
            return parsed.isoformat().replace("+00:00", "Z")
        except ValueError:
            continue
    return None


def _compress_text(value: str | None, *, max_chars: int) -> str | None:
    candidate = _as_string(value)
    if candidate is None:
        return None
    normalized = " ".join(candidate.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _compress_text_list(
    items: list[str],
    *,
    max_items: int,
    max_chars_per_item: int,
) -> list[str]:
    compressed: list[str] = []
    for item in items[:max_items]:
        value = _compress_text(item, max_chars=max_chars_per_item)
        if value:
            compressed.append(value)
    return compressed


def _fallback_news_summary(articles: list[NewsArticle]) -> str | None:
    """Create a compact fallback summary when no news-agent text is available."""

    if not articles:
        return None
    top_headlines = [article.headline.strip() for article in articles[:NEWS_AGENT_MAX_CRITICAL_ITEMS] if article.headline.strip()]
    if not top_headlines:
        return None
    return _compress_text(
        f"Latest available news: {'; '.join(top_headlines)}",
        max_chars=NEWS_AGENT_MAX_SUMMARY_CHARS,
    )


def _filter_articles_for_decision_context(articles: list[NewsArticle], symbol: str, *, query: str) -> list[NewsArticle]:
    symbol_query = SYMBOL_NEWS_QUERY_TEMPLATE.format(symbol=symbol.upper())
    if query != symbol_query:
        return list(articles)

    target_symbol = symbol.upper()
    filtered: list[NewsArticle] = []
    for article in articles:
        provider = article.provider.strip().lower()
        if provider != "alpha_vantage":
            filtered.append(article)
            continue
        if article.primary_ticker and article.primary_ticker.upper() == target_symbol:
            filtered.append(article)
    return filtered


def _filter_articles_published_at_lte(
    articles: list[NewsArticle],
    *,
    published_at_lte: datetime | str | None,
) -> list[NewsArticle]:
    if published_at_lte is None:
        return list(articles)
    cutoff = _parse_published_at(str(published_at_lte))
    if cutoff is None:
        return list(articles)
    filtered: list[NewsArticle] = []
    for article in articles:
        published_at = _parse_published_at(article.published_at)
        if published_at is None or published_at <= cutoff:
            filtered.append(article)
    return filtered


def _news_fetch_bucket(published_at_lte: datetime | str | None) -> str:
    """Group persisted news fetches by UTC calendar day."""

    if published_at_lte is not None:
        cutoff = _parse_published_at(str(published_at_lte))
        if cutoff is not None:
            return cutoff.date().isoformat()
    return datetime.now(timezone.utc).date().isoformat()


def _article_key(article: NewsArticle) -> str:
    url = article.url.strip()
    if url:
        return url.lower()
    headline = article.headline.strip().lower()
    source = article.source.strip().lower()
    if headline and source:
        return f"{headline} | {source}"
    return " | ".join(
        [
            headline,
            source,
            str(article.published_at or "").strip().lower(),
        ]
    )


def _combined_article_key(article: NewsArticle) -> str:
    provider = article.provider.strip().lower()
    return f"{provider}::{_article_key(article)}" if provider else _article_key(article)


def _extract_symbol_from_query(query: str) -> str:
    if not query.strip():
        return ""
    return query.strip().split()[0].upper()


def _extract_primary_ticker(item: dict[str, object]) -> tuple[str | None, float | None]:
    raw_ticker_sentiment = item.get("ticker_sentiment")
    if not isinstance(raw_ticker_sentiment, list) or not raw_ticker_sentiment:
        return None, None

    primary_item: dict[str, object] | None = None
    primary_relevance: float | None = None
    for candidate in raw_ticker_sentiment:
        if not isinstance(candidate, dict):
            continue
        try:
            relevance = float(candidate.get("relevance_score", 0) or 0)
        except (TypeError, ValueError):
            relevance = 0.0
        if primary_item is None or relevance > (primary_relevance or 0.0):
            primary_item = candidate
            primary_relevance = relevance

    if primary_item is None:
        return None, None
    ticker = _as_string(primary_item.get("ticker"))
    if ticker is None:
        return None, None
    return ticker.upper(), primary_relevance


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
