from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import fresh_simple_trading_project.information_retrieval as information_retrieval_module
from fresh_simple_trading_project.information_retrieval import (
    AlpacaNewsSearchClient,
    CombinedNewsSearchClient,
    InformationRetrievalModule,
    WebSearchNewsClient,
)
from fresh_simple_trading_project.models import NewsArticle


@dataclass
class RecordingNewsSearchClient:
    responses: dict[str, list[NewsArticle]]
    calls: list[tuple[str, int]] = field(default_factory=list)

    def search_news(self, query: str, limit: int = 10) -> list[NewsArticle]:
        self.calls.append((query, limit))
        return list(self.responses.get(query, []))


class RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, content: str) -> str | None:
        self.calls.append((system_prompt, content))
        return "Macro and market context summarized."


class StubHTTPResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "StubHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


@dataclass
class StubAlpacaService:
    responses_by_symbols: dict[str | None, list[NewsArticle]]
    calls: list[tuple[str | None, int, int]] = field(default_factory=list)

    def fetch_news(
        self,
        *,
        symbols: str | list[str] | None = None,
        days: int = 7,
        limit: int = 10,
    ) -> list[NewsArticle]:
        if isinstance(symbols, list):
            key = ",".join(symbols)
        else:
            key = symbols
        self.calls.append((key, days, limit))
        return list(self.responses_by_symbols.get(key, []))

    def fetch_recent_news(self, symbol: str, days: int = 7, limit: int = 10) -> list[NewsArticle]:
        self.calls.append((symbol, days, limit))
        return list(self.responses_by_symbols.get(symbol, []))


def test_retrieve_searches_symbol_macro_and_market_queries() -> None:
    client = RecordingNewsSearchClient(
        {
            "AAPL stock market news": [NewsArticle(headline="AAPL launches a new product", source="Symbol Wire")],
            "American economy inflation interest rates jobs GDP Federal Reserve": [
                NewsArticle(headline="American economy shows cooling inflation", source="Macro Wire")
            ],
            "overall U.S. stock market trend S&P 500 Nasdaq Dow market breadth": [
                NewsArticle(headline="S&P 500 and Nasdaq extend broad market rally", source="Market Wire")
            ],
        }
    )
    module = InformationRetrievalModule(client)

    result = module.retrieve("AAPL", limit=6)

    assert [query for query, _ in client.calls] == [
        "AAPL stock market news",
        "American economy inflation interest rates jobs GDP Federal Reserve",
        "overall U.S. stock market trend S&P 500 Nasdaq Dow market breadth",
    ]
    assert result.headline_summary == [
        "AAPL launches a new product",
        "American economy shows cooling inflation",
        "S&P 500 and Nasdaq extend broad market rally",
    ]


def test_retrieve_excludes_articles_older_than_one_week() -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    recent = (now - timedelta(days=2)).isoformat().replace("+00:00", "Z")
    stale = (now - timedelta(days=9)).isoformat().replace("+00:00", "Z")
    client = RecordingNewsSearchClient(
        {
            "AAPL stock market news": [
                NewsArticle(
                    headline="Stale AAPL article should be ignored",
                    source="Old Wire",
                    published_at=stale,
                ),
                NewsArticle(
                    headline="Fresh AAPL article remains eligible",
                    source="Recent Wire",
                    published_at=recent,
                ),
            ],
        }
    )
    module = InformationRetrievalModule(client, max_article_age_days=7)

    result = module.retrieve("AAPL", limit=6)

    assert result.headline_summary == ["Fresh AAPL article remains eligible"]
    assert [article.headline for article in result.articles] == ["Fresh AAPL article remains eligible"]


def test_combined_news_client_merges_live_and_static_results_without_duplicates() -> None:
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    live_client = RecordingNewsSearchClient(
        {
            "AAPL stock market news": [
                NewsArticle(
                    headline="Live headline",
                    source="Live Wire",
                    url="https://example.com/live",
                    published_at=recent,
                ),
                NewsArticle(
                    headline="Duplicate headline",
                    source="Live Wire",
                    url="https://example.com/duplicate",
                    published_at=recent,
                ),
            ]
        }
    )
    static_client = RecordingNewsSearchClient(
        {
            "AAPL stock market news": [
                NewsArticle(
                    headline="Duplicate headline",
                    source="Demo News",
                    url="https://example.com/duplicate",
                    published_at=recent,
                ),
                NewsArticle(
                    headline="Static fallback headline",
                    source="Demo News",
                    published_at=recent,
                ),
            ]
        }
    )
    client = CombinedNewsSearchClient([live_client, static_client])

    result = client.search_news("AAPL stock market news", limit=5)

    assert [article.headline for article in result] == [
        "Live headline",
        "Duplicate headline",
        "Static fallback headline",
    ]


def test_alpaca_news_client_uses_alpaca_for_symbol_macro_and_market_scopes() -> None:
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    service = StubAlpacaService(
        {
            "AAPL": [NewsArticle(headline="AAPL beats earnings estimates", source="Benzinga", published_at=recent)],
            None: [
                NewsArticle(headline="American economy cools as inflation slows", source="Benzinga", published_at=recent),
                NewsArticle(headline="Unrelated company acquisition headline", source="Benzinga", published_at=recent),
            ],
            "SPY,QQQ,DIA,IWM": [
                NewsArticle(headline="S&P 500 and Nasdaq extend stock market rally", source="Benzinga", published_at=recent),
            ],
        }
    )
    client = AlpacaNewsSearchClient(service, max_age_days=7)

    symbol_articles = client.search_news("AAPL stock market news", limit=5)
    macro_articles = client.search_news("American economy inflation interest rates jobs GDP Federal Reserve", limit=5)
    market_articles = client.search_news("overall U.S. stock market trend S&P 500 Nasdaq Dow market breadth", limit=5)

    assert [article.headline for article in symbol_articles] == ["AAPL beats earnings estimates"]
    assert [article.headline for article in macro_articles] == ["American economy cools as inflation slows"]
    assert [article.headline for article in market_articles] == ["S&P 500 and Nasdaq extend stock market rally"]
    assert service.calls[0] == ("AAPL", 7, 5)
    assert service.calls[1] == (None, 7, 25)
    assert service.calls[2] == ("SPY,QQQ,DIA,IWM", 7, 25)


def test_web_search_news_client_parses_live_search_feed(monkeypatch) -> None:
    rss_payload = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>American economy cools as inflation slows - Macro Wire</title>
      <link>https://news.google.com/rss/articles/example-1</link>
      <pubDate>Sun, 29 Mar 2026 06:00:00 GMT</pubDate>
      <description><![CDATA[<p>Federal Reserve and inflation coverage.</p>]]></description>
      <source url="https://example.com">Macro Wire</source>
    </item>
  </channel>
</rss>"""
    monkeypatch.setattr(
        information_retrieval_module,
        "urlopen",
        lambda *args, **kwargs: StubHTTPResponse(rss_payload),
    )
    client = WebSearchNewsClient(max_age_days=7)

    result = client.search_news("American economy inflation interest rates jobs GDP Federal Reserve", limit=5)

    assert [article.headline for article in result] == ["American economy cools as inflation slows"]
    assert result[0].source == "Macro Wire"
    assert result[0].summary == "Federal Reserve and inflation coverage."
    assert result[0].published_at == "2026-03-29T06:00:00Z"


def test_news_agent_prompt_mentions_macro_and_market_search_focus() -> None:
    client = RecordingNewsSearchClient(
        {
            "AAPL stock market news": [NewsArticle(headline="AAPL gets an analyst upgrade", source="Symbol Wire")],
            "American economy inflation interest rates jobs GDP Federal Reserve": [
                NewsArticle(headline="American economy adds jobs as inflation cools", source="Macro Wire")
            ],
            "overall U.S. stock market trend S&P 500 Nasdaq Dow market breadth": [
                NewsArticle(headline="Overall U.S. stock market trend remains bullish", source="Market Wire")
            ],
        }
    )
    llm = RecordingLLM()
    module = InformationRetrievalModule(client, llm_client=llm)

    result = module.retrieve("AAPL", limit=6)

    assert result.summary_note == "Macro and market context summarized."
    _, prompt = llm.calls[-1]
    assert "American economy developments" in prompt
    assert "overall U.S. stock market trend" in prompt
    assert "Queries used:" in prompt
