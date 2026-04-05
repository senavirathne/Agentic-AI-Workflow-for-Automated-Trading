from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import fresh_simple_trading_project.information_retrieval as information_retrieval_module
from fresh_simple_trading_project.agents import NewsResearchAgent
from fresh_simple_trading_project.information_retrieval import (
    AlphaVantageNewsSearchClient,
    CombinedNewsSearchClient,
    InformationRetrievalModule,
    WebSearchNewsClient,
)
from fresh_simple_trading_project.models import NewsArticle
from fresh_simple_trading_project.storage import InMemoryResultStore


@dataclass
class RecordingNewsSearchClient:
    responses: dict[str, list[NewsArticle]]
    calls: list[tuple[str, int]] = field(default_factory=list)

    def search_news(self, query: str, limit: int = 10, *, input_size_chars: int | None = None) -> list[NewsArticle]:
        del input_size_chars
        self.calls.append((query, limit))
        return list(self.responses.get(query, []))


class RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, content: str) -> str | None:
        self.calls.append((system_prompt, content))
        return "Macro and market context summarized."


class JSONRecordingLLM:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, content: str) -> str | None:
        self.calls.append((system_prompt, content))
        return self.payload


class StubHTTPResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "StubHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeAlphaVantageResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


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


def test_retrieve_keeps_recent_articles_first_when_older_history_exists() -> None:
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

    assert result.headline_summary == [
        "Fresh AAPL article remains eligible",
        "Stale AAPL article should be ignored",
    ]
    assert [article.headline for article in result.articles] == [
        "Fresh AAPL article remains eligible",
        "Stale AAPL article should be ignored",
    ]


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


def test_alpha_vantage_news_client_uses_tickers_topics_and_dynamic_limit_without_time_from() -> None:
    calls: list[dict] = []

    def fake_http_get(_url: str, *, params: dict, timeout: int) -> FakeAlphaVantageResponse:
        del timeout
        calls.append(dict(params))
        if params.get("tickers") == "AAPL":
            feed = [
                {
                    "title": "AAPL beats earnings estimates",
                    "summary": "AAPL reports strong quarterly results.",
                    "source": "Alpha Wire",
                    "url": "https://example.com/aapl",
                    "time_published": "20260402T013000",
                    "ticker_sentiment": [
                        {"ticker": "MSFT", "relevance_score": "0.41"},
                        {"ticker": "AAPL", "relevance_score": "0.87"},
                    ],
                }
            ]
        elif params.get("topics") == "financial_markets":
            feed = [
                {
                    "title": "S&P 500 extends market rally",
                    "summary": "Broad market strength continues.",
                    "source": "Alpha Wire",
                    "url": "https://example.com/market",
                    "time_published": "20260402T020000",
                }
            ]
        else:
            feed = [
                {
                    "title": "Federal Reserve holds rates steady",
                    "summary": "Macro policy remains restrictive.",
                    "source": "Alpha Wire",
                    "url": "https://example.com/macro",
                    "time_published": "20260402T021500",
                }
            ]
        return FakeAlphaVantageResponse({"feed": feed})

    client = AlphaVantageNewsSearchClient(
        api_key="ABCDEFGHI1234",
        max_age_days=7,
        http_get=fake_http_get,
    )

    symbol_articles = client.search_news("AAPL stock market news", limit=5, input_size_chars=12_000)
    macro_articles = client.search_news(
        "American economy inflation interest rates jobs GDP Federal Reserve",
        limit=5,
        input_size_chars=12_000,
    )
    market_articles = client.search_news(
        "overall U.S. stock market trend S&P 500 Nasdaq Dow market breadth",
        limit=5,
        input_size_chars=4_000,
    )

    assert [article.headline for article in symbol_articles] == ["AAPL beats earnings estimates"]
    assert [article.headline for article in macro_articles] == ["Federal Reserve holds rates steady"]
    assert [article.headline for article in market_articles] == ["S&P 500 extends market rally"]
    assert symbol_articles[0].provider == "alpha_vantage"
    assert symbol_articles[0].primary_ticker == "AAPL"
    assert symbol_articles[0].primary_ticker_relevance == 0.87
    assert macro_articles[0].primary_ticker is None
    assert calls[0]["tickers"] == "AAPL"
    assert calls[1]["topics"] == "economy_macro,economy_monetary,economy_fiscal"
    assert calls[2]["topics"] == "financial_markets"
    assert "time_from" not in calls[0]
    assert "time_from" not in calls[1]
    assert "time_from" not in calls[2]
    assert calls[0]["limit"] > calls[2]["limit"]


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
    module = InformationRetrievalModule(client, news_agent=NewsResearchAgent(llm_client=llm))

    result = module.retrieve("AAPL", limit=6)

    assert result.summary_note == "Macro and market context summarized."
    _, prompt = llm.calls[-1]
    assert "American economy developments" in prompt
    assert "overall U.S. stock market trend" in prompt
    assert "Queries used:" in prompt
    assert "Rule-based sentiment score" not in prompt


def test_retrieve_builds_fallback_summary_from_latest_available_news_when_agent_is_absent() -> None:
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

    assert result.summary_note == (
        "Latest available news: AAPL launches a new product; American economy shows cooling inflation; "
        "S&P 500 and Nasdaq extend broad market rally"
    )


def test_news_agent_uses_structured_llm_output_for_critical_news() -> None:
    client = RecordingNewsSearchClient(
        {
            "AAPL stock market news": [
                NewsArticle(headline="AAPL supplier disruption may hit iPhone shipments", source="Symbol Wire")
            ],
            "American economy inflation interest rates jobs GDP Federal Reserve": [
                NewsArticle(headline="Federal Reserve signals slower rate cuts", source="Macro Wire")
            ],
            "overall U.S. stock market trend S&P 500 Nasdaq Dow market breadth": [
                NewsArticle(headline="Nasdaq breadth weakens despite index gains", source="Market Wire")
            ],
        }
    )
    llm = JSONRecordingLLM(
        """
        {
          "summary_note": "Supply-chain pressure and a less dovish Fed are the key decision drivers.",
          "critical_news": [
            "AAPL supplier disruption could pressure near-term device shipments.",
            "Federal Reserve commentary points to tighter financial conditions for longer."
          ],
          "risk_flags": [
            "Supply-chain disruption risk remains active.",
            "Macro policy remains restrictive."
          ],
          "catalysts": [
            "Any resolution of supplier constraints would be positive."
          ]
        }
        """
    )
    module = InformationRetrievalModule(client, news_agent=NewsResearchAgent(llm_client=llm))

    result = module.retrieve("AAPL", limit=6)

    assert result.summary_note == "Supply-chain pressure and a less dovish Fed are the key decision drivers."
    assert result.critical_news == [
        "AAPL supplier disruption could pressure near-term device shipments.",
        "Federal Reserve commentary points to tighter financial conditions for longer.",
    ]
    assert result.headline_summary == result.critical_news
    assert result.risk_flags == [
        "Supply-chain disruption risk remains active.",
        "Macro policy remains restrictive.",
    ]
    assert result.catalysts == ["Any resolution of supplier constraints would be positive."]
    assert result.sentiment_score == 0.0


def test_news_agent_output_is_capped_to_three_compressed_critical_items() -> None:
    client = RecordingNewsSearchClient(
        {
            "AAPL stock market news": [NewsArticle(headline="AAPL article", source="Symbol Wire")],
            "American economy inflation interest rates jobs GDP Federal Reserve": [
                NewsArticle(headline="Economy article", source="Macro Wire")
            ],
            "overall U.S. stock market trend S&P 500 Nasdaq Dow market breadth": [
                NewsArticle(headline="Market article", source="Market Wire")
            ],
        }
    )
    llm = JSONRecordingLLM(
        """
        {
          "summary_note": "A very long summary that should still remain concise after compression because the retrieval layer now enforces a hard bound on the news handoff size before it reaches the decision agent.",
          "critical_news": [
            "AAPL: exceptionally long symbol-specific critical news item that should be trimmed because it is much longer than the compact format required for the decision-making agent to consume efficiently.",
            "Economy: exceptionally long macroeconomic critical news item that should also be trimmed because the news handoff must stay compact and precise.",
            "Market: exceptionally long stock-market critical news item that should also be trimmed for the same reason.",
            "Extra item that should be dropped entirely."
          ],
          "risk_flags": [],
          "catalysts": []
        }
        """
    )
    module = InformationRetrievalModule(client, news_agent=NewsResearchAgent(llm_client=llm))

    result = module.retrieve("AAPL", limit=6)

    assert len(result.critical_news) == 3
    assert all(len(item) <= 180 for item in result.critical_news)
    assert len(result.summary_note or "") <= 320
    assert result.headline_summary == result.critical_news


def test_news_agent_fills_input_budget_with_older_articles() -> None:
    recent = datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)
    articles = [
        NewsArticle(
            headline=f"Article {index}",
            summary="x" * 1500,
            source="Wire",
            published_at=(recent - timedelta(hours=index)).isoformat().replace("+00:00", "Z"),
        )
        for index in range(10)
    ]
    client = RecordingNewsSearchClient(
        {
            "AAPL stock market news": articles,
            "American economy inflation interest rates jobs GDP Federal Reserve": [],
            "overall U.S. stock market trend S&P 500 Nasdaq Dow market breadth": [],
        }
    )
    llm = RecordingLLM()
    module = InformationRetrievalModule(client, news_agent=NewsResearchAgent(llm_client=llm))

    result = module.retrieve("AAPL", limit=6)

    _, prompt = llm.calls[-1]
    assert "Article 0" in prompt
    assert "Article 1" in prompt
    assert "Article 6" in prompt
    assert "Article 7" not in prompt
    assert [article.headline for article in result.articles][:3] == ["Article 0", "Article 1", "Article 2"]


def test_retrieve_uses_cached_news_history_to_fill_input_budget() -> None:
    recent = datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)
    archive = InMemoryResultStore()
    archive.save_retrieved_news(
        "AAPL",
        "AAPL stock market news",
        [
            NewsArticle(
                headline=f"Cached article {index}",
                summary="y" * 1200,
                source="Archive Wire",
                published_at=(recent - timedelta(hours=index + 1)).isoformat().replace("+00:00", "Z"),
            )
            for index in range(4)
        ],
    )
    client = RecordingNewsSearchClient(
        {
            "AAPL stock market news": [
                NewsArticle(
                    headline="Live article",
                    summary="z" * 1200,
                    source="Live Wire",
                    published_at=recent.isoformat().replace("+00:00", "Z"),
                )
            ],
            "American economy inflation interest rates jobs GDP Federal Reserve": [],
            "overall U.S. stock market trend S&P 500 Nasdaq Dow market breadth": [],
        }
    )
    llm = RecordingLLM()
    module = InformationRetrievalModule(
        client,
        news_agent=NewsResearchAgent(llm_client=llm),
        news_archive=archive,
    )

    result = module.retrieve("AAPL", limit=6, input_size_chars=4_000)

    _, prompt = llm.calls[-1]
    assert "Live article" in prompt
    assert "Cached article 0" in prompt
    assert "Cached article 1" in prompt
    assert [article.headline for article in result.articles][:3] == [
        "Live article",
        "Cached article 0",
        "Cached article 1",
    ]


def test_retrieve_calls_alpha_vantage_news_once_per_day_and_reuses_stored_articles() -> None:
    archive = InMemoryResultStore()
    calls: list[dict] = []

    def fake_http_get(_url: str, *, params: dict, timeout: int) -> FakeAlphaVantageResponse:
        del timeout
        calls.append(dict(params))
        if params.get("tickers") == "AAPL":
            feed = [
                {
                    "title": "Fresh AAPL article",
                    "summary": "Alpha Vantage symbol-specific article.",
                    "source": "Alpha Wire",
                    "url": "https://example.com/aapl-fresh",
                    "time_published": "20260402T120000",
                    "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.91"}],
                }
            ]
        elif params.get("topics") == "economy_macro,economy_monetary,economy_fiscal":
            feed = [
                {
                    "title": "Fresh macro article",
                    "summary": "Macro backdrop update.",
                    "source": "Alpha Wire",
                    "url": "https://example.com/macro-fresh",
                    "time_published": "20260402T110000",
                }
            ]
        else:
            feed = [
                {
                    "title": "Fresh market article",
                    "summary": "Market breadth update.",
                    "source": "Alpha Wire",
                    "url": "https://example.com/market-fresh",
                    "time_published": "20260402T100000",
                }
            ]
        return FakeAlphaVantageResponse({"feed": feed})

    module = InformationRetrievalModule(
        AlphaVantageNewsSearchClient(
            api_key="ABCDEFGHI1234",
            max_age_days=7,
            http_get=fake_http_get,
        ),
        news_archive=archive,
    )

    first = module.retrieve("AAPL", limit=6, published_at_lte="2026-04-02T20:00:00Z")
    second = module.retrieve("AAPL", limit=6, published_at_lte="2026-04-02T20:00:00Z")

    assert len(calls) == 3
    assert [article.headline for article in first.articles] == [
        "Fresh AAPL article",
        "Fresh macro article",
        "Fresh market article",
    ]
    assert [article.headline for article in second.articles] == [
        "Fresh AAPL article",
        "Fresh macro article",
        "Fresh market article",
    ]


def test_retrieve_reuses_latest_cached_news_when_alpha_vantage_hits_free_key_limit() -> None:
    archive = InMemoryResultStore()
    queries = [
        "AAPL stock market news",
        "American economy inflation interest rates jobs GDP Federal Reserve",
        "overall U.S. stock market trend S&P 500 Nasdaq Dow market breadth",
    ]
    archive.save_retrieved_news(
        "AAPL",
        queries[0],
        [
            NewsArticle(
                headline="Cached symbol article",
                summary="Previously stored symbol news.",
                source="Archive Wire",
                url="https://example.com/cached-symbol",
                published_at="2026-04-01T12:00:00Z",
                provider="alpha_vantage",
                primary_ticker="AAPL",
                primary_ticker_relevance=0.94,
            )
        ],
    )
    archive.save_retrieved_news(
        "AAPL",
        queries[1],
        [
            NewsArticle(
                headline="Cached macro article",
                summary="Previously stored macro news.",
                source="Archive Wire",
                url="https://example.com/cached-macro",
                published_at="2026-04-01T11:00:00Z",
                provider="alpha_vantage",
            )
        ],
    )
    archive.save_retrieved_news(
        "AAPL",
        queries[2],
        [
            NewsArticle(
                headline="Cached market article",
                summary="Previously stored market news.",
                source="Archive Wire",
                url="https://example.com/cached-market",
                published_at="2026-04-01T10:00:00Z",
                provider="alpha_vantage",
            )
        ],
    )

    calls: list[dict] = []

    def fake_http_get(_url: str, *, params: dict, timeout: int) -> FakeAlphaVantageResponse:
        del timeout
        calls.append(dict(params))
        return FakeAlphaVantageResponse({"Note": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."})

    module = InformationRetrievalModule(
        AlphaVantageNewsSearchClient(
            api_key="ABCDEFGHI1234",
            max_age_days=7,
            http_get=fake_http_get,
        ),
        news_archive=archive,
    )

    first = module.retrieve("AAPL", limit=6, published_at_lte="2026-04-02T20:00:00Z")
    second = module.retrieve("AAPL", limit=6, published_at_lte="2026-04-02T20:00:00Z")

    assert len(calls) == 3
    assert [article.headline for article in first.articles] == [
        "Cached symbol article",
        "Cached macro article",
        "Cached market article",
    ]
    assert [article.headline for article in second.articles] == [
        "Cached symbol article",
        "Cached macro article",
        "Cached market article",
    ]


def test_retrieve_filters_cached_alpha_vantage_news_to_target_primary_ticker() -> None:
    recent = datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)
    archive = InMemoryResultStore()
    archive.save_retrieved_news(
        "AAPL",
        "AAPL stock market news",
        [
            NewsArticle(
                headline="AAPL primary article",
                summary="AAPL-specific update",
                source="Alpha Wire",
                url="https://example.com/aapl-primary",
                published_at=recent.isoformat().replace("+00:00", "Z"),
                provider="alpha_vantage",
                primary_ticker="AAPL",
                primary_ticker_relevance=0.92,
            ),
            NewsArticle(
                headline="MSFT primary article",
                summary="Cross-ticker article that should be removed",
                source="Alpha Wire",
                url="https://example.com/msft-primary",
                published_at=(recent - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
                provider="alpha_vantage",
                primary_ticker="MSFT",
                primary_ticker_relevance=0.95,
            ),
            NewsArticle(
                headline="Web market fallback",
                summary="Broad market context should remain available.",
                source="Market Wire",
                url="https://example.com/web-market",
                published_at=(recent - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
                provider="web_search",
            ),
        ],
    )
    client = RecordingNewsSearchClient(
        {
            "AAPL stock market news": [],
            "American economy inflation interest rates jobs GDP Federal Reserve": [],
            "overall U.S. stock market trend S&P 500 Nasdaq Dow market breadth": [],
        }
    )
    llm = RecordingLLM()
    module = InformationRetrievalModule(
        client,
        news_agent=NewsResearchAgent(llm_client=llm),
        news_archive=archive,
    )

    result = module.retrieve("AAPL", limit=6, input_size_chars=4_000)

    _, prompt = llm.calls[-1]
    assert "AAPL primary article" in prompt
    assert "MSFT primary article" not in prompt
    assert "Web market fallback" in prompt
    assert [article.headline for article in result.articles] == [
        "AAPL primary article",
        "Web market fallback",
    ]


def test_retrieve_filters_articles_to_backtest_news_cutoff() -> None:
    cutoff = "2026-04-01T20:00:00Z"
    archive = InMemoryResultStore()
    archive.save_retrieved_news(
        "AAPL",
        "AAPL stock market news",
        [
            NewsArticle(
                headline="Cached article before cutoff",
                summary="Eligible cached context.",
                source="Archive Wire",
                published_at="2026-04-01T19:30:00Z",
            ),
            NewsArticle(
                headline="Cached article after cutoff",
                summary="Should be excluded from the backtest prompt.",
                source="Archive Wire",
                published_at="2026-04-01T21:30:00Z",
            ),
        ],
    )
    client = RecordingNewsSearchClient(
        {
            "AAPL stock market news": [
                NewsArticle(
                    headline="Live article after cutoff",
                    summary="Should not be used.",
                    source="Live Wire",
                    published_at="2026-04-01T22:00:00Z",
                ),
                NewsArticle(
                    headline="Live article before cutoff",
                    summary="Still eligible.",
                    source="Live Wire",
                    published_at="2026-04-01T18:00:00Z",
                ),
            ],
            "American economy inflation interest rates jobs GDP Federal Reserve": [],
            "overall U.S. stock market trend S&P 500 Nasdaq Dow market breadth": [],
        }
    )
    llm = RecordingLLM()
    module = InformationRetrievalModule(
        client,
        news_agent=NewsResearchAgent(llm_client=llm),
        news_archive=archive,
    )

    result = module.retrieve("AAPL", limit=6, published_at_lte=cutoff)

    _, prompt = llm.calls[-1]
    assert "Cached article before cutoff" in prompt
    assert "Live article before cutoff" in prompt
    assert "Cached article after cutoff" not in prompt
    assert "Live article after cutoff" not in prompt
    assert [article.headline for article in result.articles] == [
        "Cached article before cutoff",
        "Live article before cutoff",
    ]
