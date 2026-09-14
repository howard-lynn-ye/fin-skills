#!/usr/bin/env python3
"""Opt-in network smoke check; no persistent service or external notifications.

    python scripts/check_collection_live.py
    python scripts/check_collection_live.py --house-year 2026 --house-name Pelosi \
        --house-since 2026-08-01
    python scripts/check_collection_live.py --sec-cik 1067983

SEC requires the caller's SEC_IDENTITY. House parsing requires the collect extra.
Records live only in a temporary SQLite database; output is a content-free summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fin_skills.collect import Collector, Store, Watch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--house-year', type=int)
    parser.add_argument('--house-name', default='Pelosi')
    parser.add_argument('--house-since', default='')
    parser.add_argument('--sec-cik')
    args = parser.parse_args()
    watches = [Watch('fed', 'rss', 'https://www.federalreserve.gov/feeds/press_all.xml')]
    if args.house_year:
        watches.append(Watch('house', 'house', args.house_name, options={
            'year': args.house_year, 'since': args.house_since, 'max_filings': 2}))
    if args.sec_cik:
        watches.append(Watch('sec', 'sec', args.sec_cik, options={'max_filings': 2}))
    with tempfile.TemporaryDirectory(prefix='fin-skills-collection-') as directory:  # noqa: SIM117
        with Store(Path(directory) / 'events.sqlite3') as store:
            collector = Collector(store)
            for watch in watches:
                collector.add(watch)
            first = collector.once(force=True)
            second = collector.once(force=True)
            summary = {
                'first_poll': first['watches'], 'second_poll': second['watches'],
                'records_by_kind': dict(Counter(row['kind'] for row in store.events(limit=1000))),
                'note': 'New records on the second poll can be genuine source updates; inspect per-watch counts.',
                'temporary_database_removed_on_exit': True,
            }
            print(json.dumps(summary, ensure_ascii=True, indent=2))
            return int(any(w['status'] == 'error' for w in first['watches'] + second['watches']))


if __name__ == '__main__':
    raise SystemExit(main())
