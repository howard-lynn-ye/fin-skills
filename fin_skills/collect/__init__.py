"""Public information collectors, persistent watchlists and disclosure tracking.

    from fin_skills.collect import Store, Watch, Collector

    with Store('events.sqlite3') as store:
        collector = Collector(store)
        collector.add(Watch('fed', 'rss', 'https://www.federalreserve.gov/feeds/press_all.xml'))
        result = collector.once()
        print(result['new_records'])
        # collector.run()              # explicitly start continuous polling
        # collector.deliver(your_sink)  # local callback; acknowledge only on success

No sockets, files, processes or vendor imports are opened by importing this package.
Publication, execution, report period and observation dates are distinct. Social posts
and changes between institutional holdings snapshots are not proof of an executed trade.
"""
from .http import FetchError, HttpClient, Response
from .model import Batch, Event, Watch
from .parsers import form4, holdings_changes, house_transactions, thirteen_f
from .runtime import Collector, configure, sources
from .store import Store
from .news import collect_news, news_digest, news_watches

__all__ = [
           'collect_news', 'news_digest', 'news_watches',
           'Batch',
           'Collector',
           'Event',
           'FetchError',
           'HttpClient',
           'Response',
           'Store',
           'Watch',
           'configure',
           'form4',
           'holdings_changes',
           'house_transactions',
           'sources',
           'thirteen_f',
]
