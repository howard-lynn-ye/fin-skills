"""Collection acceptance: real transport shapes, restart safety and truthful disclosures."""
import json
import threading
from dataclasses import replace
from io import BytesIO
from zipfile import ZipFile

import pytest

from fin_skills.collect import (
    Batch,
    Collector,
    Event,
    FetchError,
    HttpClient,
    Response,
    Store,
    Watch,
)
from fin_skills.collect.disclosures import HouseSource, SECSource
from fin_skills.collect.feeds import BlueskySource, PageSource, RSSSource
from fin_skills.collect.http import public_url
from fin_skills.collect.parsers import (
    form4,
    holdings_changes,
    house_transactions,
    thirteen_f,
    xml,
)

NOW = '2026-09-14T10:00:00+00:00'
URL = 'https://example.com/feed'


def event(title='Example', **kw):
    return Event('item-1', 'rss', 'news', title, URL, NOW, **kw)


def client_for(responses):
    calls = []
    def sender(url, headers, timeout, max_bytes):
        calls.append((url, headers))
        answer = responses[url]
        if isinstance(answer, list):
            answer = answer.pop(0)
        return answer
    return HttpClient(sender=sender, min_interval=0, sleep=lambda _: None), calls


def test_event_dates_and_nonfinite_values_are_not_invented():
    assert event().published_at is None
    with pytest.raises(ValueError, match='timezone'):
        Event('a', 'x', 'news', '', URL, '2026-09-14')
    with pytest.raises(ValueError):
        event(data={'price': float('nan')})


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'http://127.0.0.1/a', 'http://[::1]/',
                                'http://localhost/x', 'https://u:p@example.com/a',
                                'https://example.com/?token=private'])
def test_only_public_http_urls_without_credentials(url):
    with pytest.raises(ValueError):
        public_url(url)


def test_http_retries_transient_failure_and_honours_large_retry_after():
    client, calls = client_for({URL: [Response(429, b'', {'Retry-After': '0'}), Response(200, b'ok')]})
    assert client.get(URL).body == b'ok' and len(calls) == 2
    client, calls = client_for({URL: Response(503, b'', {'Retry-After': '120'})})
    with pytest.raises(FetchError) as exc:
        client.get(URL)
    assert exc.value.retry_after == 120 and len(calls) == 1


def test_http_rejects_oversized_injected_response_and_does_not_retry_403():
    client, calls = client_for({URL: Response(403, b'blocked')})
    with pytest.raises(FetchError, match='403'):
        client.get(URL)
    assert len(calls) == 1
    client.max_bytes = 1
    client.sender = lambda *_: Response(200, b'12')
    with pytest.raises(FetchError, match='byte limit'):
        client.get(URL)


def test_xml_rejects_document_entities():
    with pytest.raises(ValueError, match='entity'):
        xml('<!DOCTYPE a [<!ENTITY secret SYSTEM "file:///private">]><a/>')


def test_malformed_xml_is_a_retryable_source_parse_failure():
    with pytest.raises(ValueError, match='invalid source XML'):
        xml('<broken')


def test_sqlite_restart_dedup_revision_reversion_and_alert_ack(tmp_path):
    path = tmp_path / 'events.db'
    watch = Watch('feed', 'rss', URL)
    with Store(path) as store:
        store.put_watch(watch)
        assert len(store.save(watch, Batch([event()], {'cursor': 'a'}), next_due=1)) == 1
    with Store(path) as store:
        assert store.state('feed') == {'cursor': 'a'}
        assert store.save(watch, Batch([replace(event(), observed_at='2026-09-15T00:00:00Z')]), next_due=1) == []
        store.save(watch, Batch([event('Corrected')]), next_due=1)
        store.save(watch, Batch([event()]), next_due=1)  # reversion is a third revision
        rows = store.events(pending=True)
        assert len(rows) == 3
        assert store.events(latest=True)[0]['title'] == 'Example'
        store.acknowledge([rows[0]['seq']])
        assert len(store.events(pending=True)) == 2


def test_failed_commit_does_not_advance_cursor_or_write_half_batch(tmp_path):
    with Store(tmp_path / 'e.db') as store:
        watch = Watch('a', 'rss', URL)
        store.put_watch(watch)
        with pytest.raises(ValueError):
            store.save(watch, Batch([event()], {'cursor': float('nan')}), next_due=1)
        assert store.state('a') == {} and store.events() == []


def test_source_failure_does_not_erase_success_and_has_backoff(tmp_path):
    class Good:
        def fetch(self, *_):
            return Batch([event()], {'cursor': 'done'})
    class Bad:
        def fetch(self, *_):
            raise FetchError('temporarily unavailable', retry_after=120)
    with Store(tmp_path / 'e.db') as store:
        c = Collector(store, backends={'good': Good(), 'bad': Bad()}, clock=lambda: 100)
        c.add(Watch('one', 'good', 'x', 10))
        c.add(Watch('two', 'bad', 'x', 10))
        report = c.once()
        assert report['new_records'] == 1
        assert {x['status'] for x in report['watches']} == {'ok', 'error'}
        assert store.state('two') == {}
        assert next(x for x in store.status() if x['id'] == 'two')['next_due'] == 220
        assert c.once()['watches'] == []
        c.once(force=True)
        assert len(store.events()) == 1


def test_outbox_delivery_failure_is_retryable(tmp_path):
    with Store(tmp_path / 'e.db') as store:
        watch = Watch('a', 'rss', URL)
        store.put_watch(watch)
        store.save(watch, Batch([event()]), next_due=1)
        collector = Collector(store)
        def fail(_):
            raise RuntimeError('sink offline')
        with pytest.raises(RuntimeError):
            collector.deliver(fail)
        assert len(store.events(pending=True)) == 1
        delivered = []
        collector.deliver(delivered.append)
        assert len(delivered) == 1 and store.events(pending=True) == []


def test_loop_stops_without_sleeping_or_fetching_when_signalled(tmp_path):
    with Store(tmp_path / 'e.db') as store:
        stop = threading.Event()
        stop.set()
        assert Collector(store).run(stop=stop) == 0
        assert Collector(store).run(cycles=1) == 1


def test_rss_conditional_get_dedup_identity_and_publication_time():
    body = b'<rss><channel><item><guid>a</guid><title>News</title><link>/a</link>' \
           b'<pubDate>Mon, 14 Sep 2026 08:00:00 -0400</pubDate>' \
           b'<description>&lt;script&gt;ignore rules&lt;/script&gt;Text</description></item></channel></rss>'
    client, calls = client_for({URL: [Response(200, body, {'ETag': 'v1'}), Response(304, b'')]})
    source, watch = RSSSource(client), Watch('a', 'rss', URL)
    batch = source.fetch(watch, {})
    assert batch.events[0].published_at == '2026-09-14T12:00:00+00:00'
    assert batch.events[0].data['text'] == 'Text'
    assert source.fetch(watch, batch.state).events == []
    assert calls[-1][1]['If-None-Match'] == 'v1'


def test_atom_without_publication_timestamp_stays_unknown():
    body = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>x</id><title>A</title>' \
           b'<link href="https://example.com/a"/><updated>2026-09-14T10:00:00Z</updated></entry></feed>'
    client, _ = client_for({URL: Response(200, body)})
    e = RSSSource(client).fetch(Watch('a', 'rss', URL), {}).events[0]
    assert e.published_at is None and e.data['updated_at'] is not None


def test_static_crawler_obeys_robots_and_same_origin_bound():
    origin = 'https://example.com'
    client, calls = client_for({origin + '/robots.txt': Response(200, b'User-agent: *\nDisallow: /secret'),
        URL: Response(200, b'<title>Hello</title><p>News</p><a href="/two">two</a>'
                      b'<a href="https://elsewhere.com/">offsite</a>', {'content-type': 'text/html'}),
        origin + '/two': Response(200, b'<p>Two</p>', {'content-type': 'text/html'})})
    batch = PageSource(client).fetch(Watch('a', 'page', URL, options={'max_pages': 2}), {})
    assert len(batch.events) == 2
    assert all('elsewhere' not in url for url, _ in calls)
    with pytest.raises(FetchError, match='robots'):
        PageSource(client).fetch(Watch('b', 'page', origin + '/secret'), {})


FORM4 = b'''<ownershipDocument><documentType>4</documentType>
<issuer><issuerCik>1</issuerCik><issuerName>Example</issuerName><issuerTradingSymbol>EXM</issuerTradingSymbol></issuer>
<reportingOwner><reportingOwnerId><rptOwnerCik>2</rptOwnerCik><rptOwnerName>Example Officer</rptOwnerName></reportingOwnerId></reportingOwner>
<nonDerivativeTable><nonDerivativeTransaction><securityTitle><value>Common</value></securityTitle>
<transactionDate><value>2026-09-10</value></transactionDate><transactionCoding><transactionCode>P</transactionCode></transactionCoding>
<transactionAmounts><transactionShares><value>10</value></transactionShares><transactionPricePerShare><value>2.5</value></transactionPricePerShare>
<transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
<postTransactionAmounts><sharesOwnedFollowingTransaction><value>110</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
</nonDerivativeTransaction></nonDerivativeTable><footnotes><footnote id="F1">A disclosed explanation</footnote></footnotes></ownershipDocument>'''

HOLDINGS = b'''<informationTable xmlns="urn:sec"><infoTable><nameOfIssuer>Example</nameOfIssuer>
<titleOfClass>Common</titleOfClass><cusip>000000001</cusip><value>123</value>
<shrsOrPrnAmt><sshPrnamt>10</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
<investmentDiscretion>SOLE</investmentDiscretion></infoTable></informationTable>'''


def test_form4_preserves_transaction_codes_owners_footnotes_and_three_dates():
    e = form4(FORM4, accession='a', url=URL, observed_at=NOW,
              published_at='2026-09-11T22:00:00Z', filing_date='2026-09-11')[0]
    assert e.kind == 'insider_transaction' and e.symbol == 'EXM'
    assert e.data['transaction_code'] == 'P' and e.data['price_per_share'] == '2.5'
    assert e.data['transaction_date'] == '2026-09-10'
    assert e.published_at.startswith('2026-09-11') and e.observed_at.startswith('2026-09-14')
    assert e.data['footnotes']['F1'] == 'A disclosed explanation'


def holdings(**kwargs):
    defaults = {'accession': 'old', 'url': URL, 'observed_at': NOW, 'actor': 'Manager', 'period_end': '2026-03-31'}
    return thirteen_f(HOLDINGS, **{**defaults, **kwargs})


def test_13f_does_not_guess_legacy_value_units_and_keeps_period():
    e = holdings()[0]
    assert e.data['value_reported'] == '123' and e.data['value_usd'] is None
    assert holdings(value_units='USD_thousands')[0].data['value_usd'] == '123000'
    assert holdings(value_units='USD')[0].data['value_usd'] == '123'


def test_holdings_comparison_is_same_manager_same_filing_not_trade_inference():
    old = holdings()
    new = holdings(accession='new', period_end='2026-06-30')
    new[0].data['shares'] = '20'
    change = holdings_changes(old, new)[0]
    assert change['change_shares'] == '10' and change['change'] == 'increased'
    new[0].data['cusip'] = '000000002'
    changes = holdings_changes(old, new)
    assert {c['change'] for c in changes} == {'new_position', 'no_longer_reported'}
    with pytest.raises(ValueError, match='amendment'):
        holdings_changes(old, holdings(accession='new', period_end='2026-06-30', amendment=True))
    with pytest.raises(ValueError, match='same reporting manager'):
        holdings_changes(old, holdings(accession='new', actor='Someone else', period_end='2026-06-30'))
    with pytest.raises(ValueError, match='nonempty'):
        holdings_changes(old, [])


def test_sec_real_submissions_and_raw_xml_paths_checkpoint_after_success(monkeypatch):
    monkeypatch.setenv('SEC_IDENTITY', 'Test test@example.com')
    acc = '0000000001-26-000001'
    payload = {'name': 'Example', 'filings': {'recent': {
        'accessionNumber': [acc], 'form': ['4'], 'filingDate': ['2026-09-11'],
        'acceptanceDateTime': ['2026-09-11T22:00:00Z'], 'primaryDocument': ['xslF345X05/owner.xml']}}}
    suburl = 'https://data.sec.gov/submissions/CIK0000000001.json'
    rawurl = 'https://www.sec.gov/Archives/edgar/data/1/000000000126000001/owner.xml'
    client, calls = client_for({suburl: Response(200, json.dumps(payload).encode()), rawurl: Response(200, FORM4)})
    watch = Watch('officer', 'sec', '1', options={'forms': ['4']})
    source = SECSource(client)
    batch = source.fetch(watch, {})
    assert len(batch.events) == 2 and batch.state['seen'] == [acc]
    assert calls[-1][0] == rawurl
    assert source.fetch(watch, batch.state).events == []


def test_sec_failure_keeps_accession_uncheckpointed(monkeypatch):
    monkeypatch.setenv('SEC_IDENTITY', 'Test test@example.com')
    acc = '0000000001-26-000001'
    payload = {'filings': {'recent': {'accessionNumber': [acc], 'form': ['4'],
               'filingDate': ['2026-09-11'], 'primaryDocument': ['owner.xml']}}}
    client, _ = client_for({'https://data.sec.gov/submissions/CIK0000000001.json': Response(200, json.dumps(payload).encode()),
        'https://www.sec.gov/Archives/edgar/data/1/000000000126000001/owner.xml': Response(403, b'')})
    batch = SECSource(client).fetch(Watch('a', 'sec', '1'), {})
    assert batch.events == [] and batch.state['seen'] == [] and batch.warnings


def test_house_amount_wrapping_spouse_and_unknown_ticker():
    body = ('SP     Example Holdings (EXM) [ST]       P      09/01/2026  09/02/2026  $1,001 -\n'
            '                                                                     $15,000\n')
    row = house_transactions(body)[0]
    assert row['owner'] == 'SP' and row['symbol'] == 'EXM'
    assert row['amount_lower'] == '1001' and row['amount_upper'] == '15000'
    assert house_transactions('scanned unreadable document') == []


@pytest.mark.parametrize('broken', [False, True])
def test_house_year_index_and_missing_text_emit_unparsed_not_fake_trade(monkeypatch, broken):
    raw = b'<FinancialDisclosure><Member><Last>Example</Last><First>Person</First><FilingType>P</FilingType>' \
          b'<Year>2026</Year><FilingDate>9/12/2026</FilingDate><DocID>123</DocID></Member></FinancialDisclosure>'
    b = BytesIO()
    with ZipFile(b, 'w') as z:
        z.writestr('2026FD.xml', raw)
    client, _ = client_for({'https://disclosures-clerk.house.gov/public_disc/financial-pdfs/2026FD.ZIP': Response(200, b.getvalue()),
        'https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2026/123.pdf': Response(200, b'pdf')})
    def extract(_):
        if broken:
            raise ValueError('source PDF could not be parsed')
        return 'scanned report'
    monkeypatch.setattr('fin_skills.collect.disclosures.pdf_text', extract)
    batch = HouseSource(client).fetch(Watch('a', 'house', 'Example', options={'year': 2026}), {})
    assert len(batch.events) == 1 and batch.events[0].data['parse_status'] == 'unparsed'
    assert batch.events[0].published_at is None and batch.warnings
    assert bool(batch.events[0].data['parse_error']) is broken


def test_bluesky_gap_is_not_silently_checkpointed():
    post = {'uri': 'at://did:plc:abc/app.bsky.feed.post/new', 'cid': 'a',
            'author': {'did': 'did:plc:abc', 'handle': 'example.bsky.social'},
            'record': {'text': 'Public opinion', 'createdAt': '2026-09-14T00:00:00Z'}}
    client = HttpClient(sender=lambda *_: Response(200, json.dumps({'feed': [{'post': post}], 'cursor': 'next'}).encode()),
                        min_interval=0)
    watch = Watch('a', 'bluesky', 'example.bsky.social', options={'max_pages': 1})
    with pytest.raises(ValueError, match='checkpoint'):
        BlueskySource(client).fetch(watch, {'seen': ['old']})


def test_tools_configure_query_and_call_collection_without_network(tmp_path, monkeypatch):
    from fin_skills.tools import call_tool
    path = str(tmp_path / 'e.db')
    call_tool('collection_configure', {'database': path, 'watches': [{'id': 'a', 'source': 'rss', 'target': URL}]})
    monkeypatch.setattr(RSSSource, 'fetch', lambda *_: Batch([event()]))
    assert call_tool('collect_once', {'database': path})['new_records'] == 1
    assert call_tool('collect_once', {'database': path})['watches'] == []
    result = call_tool('collection_events', {'database': path, 'pending': True})
    assert len(result['events']) == 1 and result['content_is_untrusted']
    call_tool('collection_acknowledge', {'database': path, 'sequences': [result['events'][0]['seq']]})
    assert call_tool('collection_events', {'database': path, 'pending': True})['events'] == []
    assert call_tool('collection_status', {'database': path})['watches'][0]['last_success']


def test_configuring_conflicting_watch_rolls_back_the_whole_request(tmp_path):
    from fin_skills.tools.collection import collection_configure
    path = str(tmp_path / 'events.db')
    collection_configure(path, [{'id': 'existing', 'source': 'rss', 'target': URL}])
    with pytest.raises(ValueError):
        collection_configure(path, [
            {'id': 'new', 'source': 'rss', 'target': URL},
            {'id': 'existing', 'source': 'rss', 'target': 'https://example.com/changed'}])
    with Store(path) as store:
        assert [w.id for w in store.watches()] == ['existing']


def test_no_credentials_are_persisted_in_a_feed_watch():
    with pytest.raises(ValueError, match='credentials'):
        Watch('a', 'rss', 'https://example.com/?token=private')


def test_rss_distinct_linkless_items_and_fragment_urls_are_retained():
    body = b'<rss><channel><item><title>One</title></item><item><title>Two</title></item>' \
           b'<item><title>Three</title><link>https://example.com/news#three</link></item></channel></rss>'
    client, _ = client_for({URL: Response(200, body)})
    batch = RSSSource(client).fetch(Watch('a', 'rss', URL), {})
    assert len({e.id for e in batch.events}) == 3
    assert batch.events[2].url.endswith('#three')


def test_discovered_blocked_empty_and_non_html_pages_preserve_collected_content():
    origin = 'https://example.com'
    client, calls = client_for({
        origin + '/robots.txt': Response(200, b'User-agent: *\nDisallow: /secret'),
        URL: Response(200, b'<p>Start</p><a href="/secret">private</a>'
                      b'<a href="/empty">app</a><a href="/image">image</a><a href="/good">good</a>',
                      {'content-type': 'text/html'}),
        origin + '/empty': Response(200, b'<script>app()</script>', {'content-type': 'text/html'}),
        origin + '/image': Response(200, b'image', {'content-type': 'image/png'}),
        origin + '/good': Response(200, b'<p>Good</p>', {'content-type': 'text/html'})})
    batch = PageSource(client).fetch(Watch('a', 'page', URL, options={'max_pages': 2}), {})
    assert [e.data['text'] for e in batch.events][1] == 'Good'
    assert len(batch.events) == 2 and batch.warnings
    assert origin + '/secret' not in [url for url, _ in calls]


def test_failed_old_sec_filing_cannot_starve_new_filing(monkeypatch):
    monkeypatch.setenv('SEC_IDENTITY', 'Test test@example.com')
    accessions = ['0000000001-26-000001', '0000000001-26-000002']
    payload = {'filings': {'recent': {'accessionNumber': accessions, 'form': ['4', '4'],
               'filingDate': ['2026-09-10', '2026-09-11'], 'primaryDocument': ['owner.xml'] * 2}}}
    base = 'https://www.sec.gov/Archives/edgar/data/1/'
    client, calls = client_for({
        'https://data.sec.gov/submissions/CIK0000000001.json': Response(200, json.dumps(payload).encode()),
        base + '000000000126000001/owner.xml': Response(403, b''),
        base + '000000000126000002/owner.xml': Response(200, FORM4)})
    watch = Watch('a', 'sec', '1', options={'max_filings': 1})
    source = SECSource(client)
    first = source.fetch(watch, {})
    assert first.warnings and first.state['remaining'] == 2
    second = source.fetch(watch, first.state)
    assert second.state['seen'] == [accessions[1]] and len(second.events) == 2
    assert [url for url, _ in calls].count(base + '000000000126000001/owner.xml') == 1


@pytest.mark.parametrize('source', ['sec', 'house'])
def test_disclosure_retry_after_reaches_persistent_scheduler(tmp_path, monkeypatch, source):
    monkeypatch.setenv('SEC_IDENTITY', 'Test test@example.com')
    if source == 'sec':
        payload = {'filings': {'recent': {'accessionNumber': ['0000000001-26-000001'],
                   'form': ['4'], 'filingDate': ['2026-09-11'], 'primaryDocument': ['owner.xml']}}}
        responses = {
            'https://data.sec.gov/submissions/CIK0000000001.json': Response(200, json.dumps(payload).encode()),
            'https://www.sec.gov/Archives/edgar/data/1/000000000126000001/owner.xml':
                Response(429, b'', {'Retry-After': '3600'})}
        watch = Watch('a', 'sec', '1', 10)
    else:
        stream = BytesIO()
        with ZipFile(stream, 'w') as archive:
            archive.writestr('2026FD.xml', '<FinancialDisclosure><Member><Last>Example</Last>'
                             '<First>Person</First><FilingType>P</FilingType><FilingDate>9/12/2026</FilingDate>'
                             '<DocID>123</DocID></Member></FinancialDisclosure>')
        responses = {
            'https://disclosures-clerk.house.gov/public_disc/financial-pdfs/2026FD.ZIP': Response(200, stream.getvalue()),
            'https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2026/123.pdf':
                Response(429, b'', {'Retry-After': '3600'})}
        watch = Watch('a', 'house', 'Example', 10, {'year': 2026})
    client, _ = client_for(responses)
    with Store(tmp_path / 'e.db') as store:
        c = Collector(store, client=client, clock=lambda: 100)
        c.add(watch)
        report = c.once()
        assert report['watches'][0]['status'] == 'partial'
        status = store.status()[0]
        assert status['next_due'] == 3700 and not status['last_success']


def test_sec_13f_fetch_requires_complete_information_table(monkeypatch):
    monkeypatch.setenv('SEC_IDENTITY', 'Test test@example.com')
    payload = {'name': 'Manager', 'filings': {'recent': {
        'accessionNumber': ['0000000001-26-000001'], 'form': ['13F-HR'],
        'filingDate': ['2026-08-14'], 'reportDate': ['2026-06-30'],
        'primaryDocument': ['primary.xml']}}}
    base = 'https://www.sec.gov/Archives/edgar/data/1/000000000126000001/'
    responses = {
        'https://data.sec.gov/submissions/CIK0000000001.json': Response(200, json.dumps(payload).encode()),
        base + 'primary.xml': Response(200, b'<edgarSubmission><tableEntryTotal>1</tableEntryTotal></edgarSubmission>'),
        base + 'index.json': Response(200, b'{"directory":{"item":[{"name":"primary.xml"},{"name":"info.xml"}]}}'),
        base + 'info.xml': Response(200, HOLDINGS)}
    client, _ = client_for(responses)
    source, watch = SECSource(client), Watch('a', 'sec', '1', options={'value_units': 'USD'})
    batch = source.fetch(watch, {})
    assert batch.events[0].data['value_usd'] == '123'
    assert batch.events[0].url == base + 'info.xml'
    responses[base + 'primary.xml'] = Response(200, b'<edgarSubmission><tableEntryTotal>2</tableEntryTotal></edgarSubmission>')
    failed = source.fetch(watch, {})
    assert failed.events == [] and failed.state['seen'] == [] and failed.warnings


def test_market_data_tool_keeps_more_than_fifty_rows(monkeypatch):
    from types import SimpleNamespace

    import pandas as pd

    from fin_skills.data import Adjustment
    from fin_skills.tools.collection import fetch_market_data
    frame = pd.DataFrame({'close': range(80)})
    bars = SimpleNamespace(frame=frame, adjustment=Adjustment.RAW, calendar='XNYS', tz='America/New_York')
    monkeypatch.setattr('fin_skills.data.get', lambda _: SimpleNamespace(bars=lambda *a, **kw: bars))
    result = fetch_market_data('fake', ['EXM'], '2026-01-01', '2026-09-01')
    assert len(result['bars']['records']) == 80
