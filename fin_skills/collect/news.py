"""Multi-source news search and point-in-time, provenance-preserving news digests."""
from datetime import datetime, timezone
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .feeds import RSSSource
from .http import HttpClient, public_url
from .model import Batch, Event, Watch, utc, utcnow


NEWS_FEEDS = {
    "fed": "https://www.federalreserve.gov/feeds/press_all.xml",
    "ecb": "https://www.ecb.europa.eu/rss/press.html",
}
DEFAULT_RISK_TERMS = ("trading halt", "bankruptcy", "default on", "停牌", "破产", "违约")


def _stamp(value):
    return datetime.fromisoformat(utc(value))


def canonical_url(value):
    """Drop only known tracking keys; preserve article-identifying query parameters."""
    p = urlsplit(value)
    clean = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in ("fbclid", "gclid")]
    url = urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path or "/", urlencode(sorted(clean)), ""))
    return public_url(url)


def _terms(values, name):
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of strings")
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{name} must contain nonempty strings")
    return tuple(v.strip().casefold() for v in values)


def _matches(term, text):
    # English word boundaries avoid matching e.g. "halt" inside "asphalt".
    return bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)) if term.isascii() else term in text


def news_digest(events, *, as_of, max_age_hours=72.0, keywords=(), risk_terms=None):
    """Only use revisions observed AND published by as_of; missing dates remain visible.

    Exact canonical URLs are grouped, not headlines. Counts are coverage, not sentiment.
    GDELT seen times are not represented as publisher publication timestamps.
    """
    stamp = _stamp(as_of)
    if isinstance(max_age_hours, bool) or not isinstance(max_age_hours, (int, float)) or not 0 < max_age_hours < float("inf"):
        raise ValueError("max_age_hours must be finite and positive")
    keywords = _terms(keywords, "keywords")
    risks = _terms(DEFAULT_RISK_TERMS if risk_terms is None else risk_terms, "risk_terms")
    excluded = {k: 0 for k in ("future", "stale", "missing_date", "irrelevant", "not_news", "invalid")}
    known = []
    for event in events:
        row = event.to_dict() if isinstance(event, Event) else dict(event)
        try:
            if row.get("kind") != "news":
                excluded["not_news"] += 1
                continue
            observed = _stamp(row["observed_at"])
            published = _stamp(row["published_at"]) if row.get("published_at") else None
            seen = row.get("data", {}).get("indexed_at")
            dated = published or (_stamp(seen) if seen else None)
            if observed > stamp or (dated is not None and dated > stamp):
                excluded["future"] += 1
                continue
            url = canonical_url(row["url"])
            # Keep latest eligible revision per watch/event before applying content filters.
            known.append((observed, str(row.get("watch_id", row["source"])),
                          str(row["id"]), url, dated, row))
        except (KeyError, TypeError, ValueError, AttributeError):
            excluded["invalid"] += 1
    revisions = {}
    for item in sorted(known, key=lambda x: (x[0], x[1], x[2])):
        revisions[item[1:3]] = item
    groups = {}
    for observed, source, identity, url, dated, row in revisions.values():
        if dated is None:
            excluded["missing_date"] += 1
            continue
        if (stamp - dated).total_seconds() > max_age_hours * 3600:
            excluded["stale"] += 1
            continue
        title = str(row.get("title", ""))
        content = (title + " " + str(row.get("data", {}).get("text", ""))).casefold()
        if keywords and not any(_matches(k, content) for k in keywords):
            excluded["irrelevant"] += 1
            continue
        matched = [k for k in risks if _matches(k, content)]
        provenance = {"source": row["source"], "watch_id": source, "id": identity,
                      "observed_at": observed.isoformat(), "published_at": row.get("published_at"),
                      "indexed_at": row.get("data", {}).get("indexed_at")}
        if url not in groups:
            groups[url] = {"title": title, "url": url, "date": dated.isoformat(),
                           "date_basis": "publication" if row.get("published_at") else "index_observation",
                           "risk_terms": matched, "provenance": [provenance]}
        else:
            groups[url]["provenance"].append(provenance)
            groups[url]["risk_terms"] = sorted(set(groups[url]["risk_terms"] + matched))
    articles = sorted(groups.values(), key=lambda row: (row["date"], row["url"]), reverse=True)
    return {"as_of": stamp.isoformat(), "status": "available" if articles else "unknown",
            "articles": articles, "unique_articles": len(articles), "excluded": excluded,
            "risk_matches": [{"url": row["url"], "terms": row["risk_terms"]}
                             for row in articles if row["risk_terms"]],
            "risk_basis": "keyword screening, not verified events or sentiment",
            "content_is_untrusted": True}


class GDELTSource:
    """DOC 2.0 article discovery. Metadata only; bounded results are not exhaustive."""
    def __init__(self, client):
        self.client = client

    def fetch(self, watch, state):
        if not isinstance(watch.target, str) or not 1 <= len(watch.target.strip()) <= 500:
            raise ValueError("GDELT query must contain 1..500 characters")
        unknown = set(watch.options) - {"timespan", "max_records"}
        if unknown:
            raise ValueError(f"unknown GDELT options: {sorted(unknown)}")
        timespan = watch.options.get("timespan", "1d")
        if timespan not in ("15min", "1h", "6h", "12h", "1d", "3d", "7d"):
            raise ValueError("timespan must be 15min, 1h, 6h, 12h, 1d, 3d or 7d")
        limit = watch.options.get("max_records", 100)
        if type(limit) is not int or not 1 <= limit <= 250:
            raise ValueError("max_records must be an integer in 1..250")
        url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode({
            "query": watch.target, "mode": "artlist", "format": "json", "sort": "datedesc",
            "timespan": timespan, "maxrecords": limit})
        payload = self.client.get(url).json()
        if not isinstance(payload, dict) or not isinstance(payload.get("articles"), list):
            raise ValueError("GDELT response must contain an articles array")
        records, observed, warnings = [], utcnow(), []
        for row in payload["articles"][:limit]:
            try:
                article_url = canonical_url(row["url"])
                seen = row.get("seendate")
                indexed = (datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
                           if seen else None)
                records.append(Event(article_url, "gdelt", "news", row["title"], article_url, observed,
                    data={"indexed_at": indexed, "language": row.get("language"),
                          "domain": row.get("domain"), "source_country": row.get("sourcecountry"),
                          "metadata_only": True, "content_is_untrusted": True}))
            except (KeyError, TypeError, ValueError):
                warnings.append("invalid GDELT article metadata skipped")
        if len(payload["articles"]) >= limit:
            warnings.append("result cap reached; narrow the query/time window; coverage may be incomplete")
        return Batch(records, {"last_query": watch.target, "timespan": timespan,
                               "coverage": "bounded snapshot, not an exhaustive archive"}, warnings)


def news_watches(*, query=None, feeds=("fed", "ecb"), interval_seconds=900):
    """Prepare a persistent multi-source watchlist without performing network I/O."""
    if isinstance(feeds, str) or len(set(feeds)) != len(feeds):
        raise ValueError("feeds must be a sequence of distinct preset names")
    watches = [Watch("news-" + name, "rss", NEWS_FEEDS[name], interval_seconds) for name in feeds]
    if query:
        watches.append(Watch("news-search", "gdelt", query, interval_seconds))
    return watches


def collect_news(*, query=None, feeds=("fed", "ecb"), client=None):
    """One explicit network call per source (plus HTTP retries), isolated source errors.

    Use news_watches + Collector for persistence, checkpointing and a continuous schedule.
    """
    client = client or HttpClient()
    events, reports = [], []
    for watch in news_watches(query=query, feeds=feeds):
        try:
            backend = GDELTSource(client) if watch.source == "gdelt" else RSSSource(client)
            batch = backend.fetch(watch, {})
            events.extend({**event.to_dict(), "watch_id": watch.id} for event in batch.events)
            reports.append({"source": watch.id, "status": "partial" if batch.warnings else "ok",
                            "records": len(batch.events), "warnings": batch.warnings})
        except Exception as exc:  # isolate remote source failures, as Collector.once does
            reports.append({"source": watch.id, "status": "error", "error": str(exc),
                            "retry_after_seconds": getattr(exc, "retry_after", None)})
    return {"events": events, "sources": reports, "observed_at": utcnow(),
            "content_is_untrusted": True}
