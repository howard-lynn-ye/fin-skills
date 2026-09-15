"""Command line collection and local alert consumption (JSON output)."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from . import Collector, Store, configure, sources


def main(argv=None):
    parser = argparse.ArgumentParser(description='Collect public information into a local SQLite watchlist')
    parser.add_argument('--database', default='fin-skills-events.sqlite3')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('sources')
    init = sub.add_parser('configure')
    init.add_argument('config', help='JSON file containing a watches array (no credentials)')
    sub.add_parser('watches')
    sub.add_parser('status')
    once = sub.add_parser('once')
    once.add_argument('--force', action='store_true')
    run = sub.add_parser('run')
    run.add_argument('--cycles', type=int, default=None, help='omit for continuous polling; Ctrl+C stops')
    run.add_argument('--stdout-alerts', action='store_true', help='deliver pending alerts to stdout')
    events = sub.add_parser('events')
    events.add_argument('--watch')
    events.add_argument('--after', type=int, default=0)
    events.add_argument('--limit', type=int, default=100)
    events.add_argument('--pending', action='store_true')
    events.add_argument('--latest', action='store_true')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    def emit(value):
        print(json.dumps(value, ensure_ascii=True, allow_nan=False), flush=True)
    if args.command == 'sources':
        emit(sources())
        return 0
    try:
        with Store(args.database) as store:
            collector = Collector(store)
            if args.command == 'configure':
                with open(args.config, encoding='utf-8') as f:
                    config = json.load(f)
                configure(store, config['watches'])
                emit({'watches': [asdict(w) for w in store.watches()]})
            elif args.command == 'watches':
                emit([asdict(w) for w in store.watches()])
            elif args.command == 'status':
                emit(store.status())
            elif args.command == 'once':
                report = collector.once(force=args.force)
                emit(report)
                return int(any(w['status'] != 'ok' for w in report['watches']))
            elif args.command == 'events':
                emit(store.events(watch_id=args.watch, after=args.after, limit=args.limit,
                                  pending=args.pending, latest=args.latest))
            elif args.command == 'run':
                def on_cycle(report):
                    emit({'type': 'cycle', **report})
                    if args.stdout_alerts:
                        collector.deliver(lambda event: emit({'type': 'alert', **event}))
                collector.run(cycles=args.cycles, on_cycle=on_cycle)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, KeyError) as exc:
        emit({'error': f'{type(exc).__name__}: {exc}'})
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
