"""Source failures, real wire-format fixtures, revision timing and cross-source dedup."""
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from fin_skills.collect import (Collector, Event, HttpClient, Response, Store, Watch,
                                collect_news, news_digest, news_watches)
from fin_skills.collect.news import GDELTSource
from fin_skills.tools import call_tool

NOW = "2026-09-21T12:00:00Z"


def event(**fields):
    row = dict(id="a", source="rss", kind="news", title="Example earnings report",
               url="https://example.com/a", observed_at=NOW, published_at=NOW)
    row.update(fields)
    return row


def test_digest_is_point_in_time_deduplicated_and_preserves_sources():
    events = [event(), event(id="b", source="gdelt", url="https://example.com/a?utm_source=x"),
              event(id="c", url="https://example.com/c", title="Trading halt"),
              event(id="d", observed_at="2026-09-22T00:00:00Z"),
              event(id="e", published_at="2026-09-10T00:00:00Z"),
              event(id="f", published_at=None), event(id="g", published_at="2026-09-22T00:00:00Z")]
    report = news_digest(events, as_of=NOW)
    assert report["unique_articles"] == 2
    assert len(next(r for r in report["articles"] if r["url"].endswith("/a"))["provenance"]) == 2
    assert report["excluded"]["future"] == 2
    assert report["excluded"]["stale"] == report["excluded"]["missing_date"] == 1
    assert len(report["risk_matches"]) == 1


def test_latest_revision_asof_does_not_keep_old_risk_or_future_correction():
    old = event(title="Trading halt", observed_at="2026-09-21T09:00:00Z",
                published_at="2026-09-21T08:00:00Z")
    corrected = event(title="Correction: normal operation", observed_at="2026-09-21T11:00:00Z",
                      published_at="2026-09-21T08:00:00Z")
    assert news_digest([old, corrected], as_of=NOW)["risk_matches"] == []
    assert news_digest([old, corrected], as_of="2026-09-21T10:00:00Z")["risk_matches"]


def test_gdelt_index_time_is_not_publication_and_keywords_support_chinese():
    row = event(title="公司破产公告", published_at=None, data={"indexed_at": NOW})
    result = call_tool("summarize_news", {"events": [row], "as_of": NOW, "keywords": ["公司"]})
    assert result["articles"][0]["date_basis"] == "index_observation"
    assert result["articles"][0]["provenance"][0]["published_at"] is None
    assert result["risk_matches"][0]["terms"] == ["破产"]
    assert news_digest([row], as_of=NOW, keywords=["other"])["status"] == "unknown"


def test_url_identity_preserves_article_queries_and_bad_rows_visible():
    result = news_digest([event(url="https://example.com/?article=1"),
                          event(id="b", url="https://example.com/?article=2"),
                          event(id="c", observed_at="not-a-date")], as_of=NOW)
    assert result["unique_articles"] == 2
    assert result["excluded"]["invalid"] == 1


def test_gdelt_wire_shape_cap_warning_and_query_encoding():
    calls = []
    def sender(url, headers, timeout, max_bytes):
        calls.append(url)
        return Response(200, json.dumps({"articles": [{"title": "Example", "url": "https://example.com/a",
            "seendate": "20260921T100000Z", "language": "English", "sourcecountry": "US"}]}).encode())
    client = HttpClient(sender=sender, min_interval=0)
    batch = GDELTSource(client).fetch(Watch("q", "gdelt", '"interest rates" OR inflation',
                                            options={"max_records": 1}), {})
    assert batch.warnings and batch.events[0].published_at is None
    assert batch.events[0].data["indexed_at"] == "2026-09-21T10:00:00+00:00"
    assert parse_qs(urlsplit(calls[0]).query)["query"] == ['"interest rates" OR inflation']
    with pytest.raises(ValueError, match="timespan"):
        GDELTSource(client).fetch(Watch("q", "gdelt", "test", options={"timespan": "5y"}), {})


def test_multisource_partial_failure_keeps_successes_and_persistent_watches(tmp_path):
    def sender(url, headers, timeout, max_bytes):
        if "ecb" in url:
            return Response(503, b"down")
        return Response(200, b'<rss><channel><item><title>Announcement</title>'
                         b'<link>https://example.com/a</link><pubDate>Mon, 21 Sep 2026 10:00:00 GMT</pubDate>'
                         b'</item></channel></rss>')
    client = HttpClient(sender=sender, min_interval=0, attempts=1)
    report = collect_news(client=client)
    assert len(report["events"]) == 1
    assert [r["status"] for r in report["sources"]] == ["ok", "error"]
    with Store(str(tmp_path / "news.sqlite")) as store:
        collector = Collector(store, client=client)
        for watch in news_watches(feeds=["fed"]):
            collector.add(watch)
        assert collector.once(force=True)["new_records"] == 1
        assert collector.once(force=True)["new_records"] == 0


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_freshness_rejected(value):
    with pytest.raises(ValueError):
        news_digest([], as_of=NOW, max_age_hours=value)
