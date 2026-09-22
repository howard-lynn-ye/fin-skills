"""Persistent polling runner. No daemon is started by importing or constructing it."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict

from .disclosures import HouseSource, SECSource
from .feeds import BlueskySource, PageSource, RSSSource
from .http import HttpClient
from .model import Watch
from .news import GDELTSource

SOURCES = {
    'gdelt': (GDELTSource, 'GDELT DOC news search; target=query, options=timespan/max_records; metadata only'),
    'rss': (RSSSource, 'RSS/Atom news feeds; conditional GET; public source timestamps'),
    'page': (PageSource, 'Static public HTML and bounded same-origin crawling; robots.txt respected'),
    'sec': (SECSource, 'Form 4 transaction XML and 13F holdings by CIK; SEC_IDENTITY required'),
    'house': (HouseSource, 'US House PTR index and PDF transactions; explicit year, pypdf required'),
    'bluesky': (BlueskySource, 'Public author posts through Bluesky API; posts are not trade evidence'),
}


def sources():
    return [{'name': k, 'description': v[1]} for k, v in SOURCES.items()]


class Collector:
    """Runs each watch independently. One failed source cannot erase another's results.

    State survives process restarts in ``Store``. Alerts are stored transactionally and
    acknowledged only AFTER the caller's delivery callback returns successfully.
    Delivery is at-least-once; use the event sequence as the downstream idempotency key.
    """

    def __init__(self, store, *, client=None, backends=None, clock=time.time):
        self.store, self.client, self.clock = store, client or HttpClient(), clock
        self.backends = backends or {name: cls(self.client) for name, (cls, _) in SOURCES.items()}

    def add(self, watch: Watch):
        if watch.source not in self.backends:
            raise ValueError(f'unknown source {watch.source!r}; choose {sorted(self.backends)}')
        self.store.put_watch(watch)

    def once(self, *, force=False, watch_ids=None):
        chosen = set(watch_ids) if watch_ids is not None else None
        watches = self.store.watches()
        if chosen is not None and chosen - {w.id for w in watches}:
            raise ValueError('watch_ids contains an unknown watch')
        reports, added = [], []
        for watch in watches:
            if not watch.enabled or (chosen is not None and watch.id not in chosen):
                continue
            if not force and not self.store.due(watch.id, self.clock()):
                continue
            try:
                batch = self.backends[watch.source].fetch(watch, self.store.state(watch.id))
                next_due = self.clock() + max(watch.interval_seconds, batch.retry_after or 0)
                new = self.store.save(watch, batch, next_due=next_due)
                added.extend(new)
                reports.append({'watch_id': watch.id, 'status': 'partial' if batch.warnings else 'ok',
                                'new_records': len(new), 'warnings': batch.warnings,
                                'remaining': batch.state.get('remaining', 0), 'next_due': next_due})
            except Exception as exc:  # noqa: BLE001 - isolate failing sources and persist status
                failures = next(s['failures'] for s in self.store.status() if s['id'] == watch.id)
                delay = max(min(watch.interval_seconds * 2 ** min(failures, 8), 3600),
                            getattr(exc, 'retry_after', None) or 0)
                self.store.failure(watch.id, f'{type(exc).__name__}: {exc}', self.clock() + delay)
                reports.append({'watch_id': watch.id, 'status': 'error', 'error': str(exc),
                                'retry_in_seconds': delay})
        return {'watches': reports, 'new_records': len(added), 'events': added,
                'content_is_untrusted': True}

    def deliver(self, callback, *, limit=100):
        """Deliver pending records; failures leave them pending for the next attempt."""
        rows = self.store.events(pending=True, limit=limit)
        for row in rows:
            callback(row)
            self.store.acknowledge([row['seq']])
        return len(rows)

    def run(self, *, stop=None, cycles=None, on_cycle=None):
        """Poll until stopped. ``cycles`` bounds a smoke run; None means continuous.

        ``on_cycle`` receives status and new events, but does not acknowledge the outbox.
        Use deliver() for a sink that must retry failed deliveries after a restart.
        """
        if cycles is not None and (not isinstance(cycles, int) or cycles < 1):
            raise ValueError('cycles must be a positive integer or None')
        stop = stop or threading.Event()
        completed = 0
        while not stop.is_set():
            report = self.once()
            if on_cycle:
                on_cycle(report)
            completed += 1
            if cycles is not None and completed >= cycles:
                break
            enabled = {w.id for w in self.store.watches() if w.enabled}
            due = [s['next_due'] for s in self.store.status() if s['id'] in enabled]
            delay = min(60, max(.25, min(due) - self.clock())) if due else 1
            stop.wait(delay)
        return completed


def configure(store, specifications):
    collector = Collector(store)
    watches = [Watch(**spec) for spec in specifications]
    if any(w.source not in collector.backends for w in watches):
        raise ValueError('unknown collection source')
    store.put_watches(watches)
    return [asdict(w) for w in store.watches()]
