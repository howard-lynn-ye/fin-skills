"""Real SEC and US House retrieval. Public disclosure is not real-time execution data."""
from __future__ import annotations

import os
import re
import time
from datetime import datetime
from io import BytesIO
from zipfile import ZipFile

from .http import FetchError
from .model import Batch, Event, utc, utcnow
from .parsers import form4, house_transactions, pdf_text, text, thirteen_f, xml


def _iso_date(value):
    for fmt in ('%m/%d/%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(value, fmt).date().isoformat()  # noqa: DTZ007 - civil date only
        except ValueError:
            pass
    raise ValueError(f'invalid source date {value!r}')


def _rows(columns):
    for i, accession in enumerate(columns.get('accessionNumber', [])):
        yield {key: values[i] for key, values in columns.items()
               if isinstance(values, list) and len(values) > i}


class SECSource:
    """Watch a CIK for actual Form 4 transactions or complete 13F information tables.

    The reporting owner CIK can be used for an individual; an issuer CIK collects all
    filings in its submissions history. ``owner_name`` filters parsed owners when an
    issuer is watched. Older history is opt-in and bounded; incomplete backfills fail.
    SEC_IDENTITY (or EDGAR_IDENTITY) must identify the caller with a contact email.
    """

    def __init__(self, client):
        self.client = client

    def fetch(self, watch, state):
        identity = os.environ.get('SEC_IDENTITY') or os.environ.get('EDGAR_IDENTITY')
        if not identity or not re.search(r'\S+@\S+\.\S+', identity):
            raise ValueError('set SEC_IDENTITY to your name/application and contact email')
        headers = {'User-Agent': identity}
        cik = str(watch.target).upper().removeprefix('CIK')
        if not cik.isdigit() or not 0 < int(cik) < 10**10:
            raise ValueError('SEC target must be a numeric CIK')
        cik = str(int(cik))
        payload = self.client.get(f'https://data.sec.gov/submissions/CIK{int(cik):010d}.json', headers).json()
        forms = watch.options.get('forms', ['4', '4/A', '13F-HR', '13F-HR/A'])
        if not isinstance(forms, list) or not forms or set(forms) - {'4', '4/A', '13F-HR', '13F-HR/A'}:
            raise ValueError('supported forms: 4, 4/A, 13F-HR, 13F-HR/A')
        recent = list(_rows(payload.get('filings', {}).get('recent', {})))
        start = watch.options.get('since', '')
        end = watch.options.get('until', '')
        if start:
            start = _iso_date(start)
        if end:
            end = _iso_date(end)
        if watch.options.get('include_archives', False):
            archives = [a for a in payload.get('filings', {}).get('files', [])
                        if (not start or a.get('filingTo', '') >= start)
                        and (not end or a.get('filingFrom', '') <= end)]
            if len(archives) > 50:
                raise ValueError('historical range exceeds 50 archives; narrow since/until')
            for archive in archives:
                filename = archive['name']
                if not re.fullmatch(r'CIK\d+-submissions-\d+\.json', filename):
                    raise ValueError('unexpected SEC archive filename')
                recent.extend(_rows(self.client.get('https://data.sec.gov/submissions/' + filename, headers).json()))
        seen = set(state.get('seen', []))
        candidates = {r['accessionNumber']: r for r in recent if r.get('form') in forms
                      and (not start or r.get('filingDate', '') >= start)
                      and (not end or r.get('filingDate', '') <= end)
                      and r['accessionNumber'] not in seen}
        maximum = int(watch.options.get('max_filings', 10))
        if not 1 <= maximum <= 100:
            raise ValueError('max_filings must be 1..100')
        retry = dict(state.get('retry', {}))
        eligible = [r for r in candidates.values()
                    if retry.get(r['accessionNumber'], {}).get('at', 0) <= time.time()]
        chosen = sorted(eligible, key=lambda r: (retry.get(r['accessionNumber'], {}).get('attempts', 0),
                                                r['filingDate'], r['accessionNumber']))[:maximum]
        events, warnings = [], []
        retry_after = None
        for row in chosen:
            accession = row['accessionNumber']
            if not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession):
                raise ValueError('invalid SEC accession number')
            base = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace("-", "")}/'
            primary = row.get('primaryDocument', '').split('/')[-1]
            if not primary or not re.fullmatch(r'[\w.\-]+', primary):
                raise ValueError('invalid SEC primary document filename')
            url = base + primary
            observed = utcnow()
            accepted = row.get('acceptanceDateTime')
            # SEC submissions stamps are UTC when Z is present; unknown naive stamps
            # stay unknown rather than being assigned an invented timezone.
            try:
                published = utc(accepted) if accepted else None
            except ValueError:
                published = None
            common = {'accession': accession, 'url': url, 'observed_at': observed,
                          'published_at': published, 'filing_date': row['filingDate']}
            try:
                if row['form'].startswith('4'):
                    parsed = form4(self.client.get(url, headers).body, **common)
                    owner = watch.options.get('owner_name', '').casefold()
                    if owner:
                        parsed = [e for e in parsed if owner in e.actor.casefold()]
                else:
                    cover = xml(self.client.get(url, headers).body)
                    index = self.client.get(base + 'index.json', headers).json()
                    parsed = None
                    documents = [d['name'] for d in index.get('directory', {}).get('item', [])
                                 if d.get('name', '').lower().endswith('.xml') and d['name'] != primary]
                    if len(documents) > 30:
                        raise ValueError('too many XML documents in filing directory')
                    for filename in documents:
                        if not re.fullmatch(r'[\w.\-]+', filename):
                            continue
                        body = self.client.get(base + filename, headers).body
                        if xml(body).tag != 'informationTable':
                            continue
                        period = row.get('reportDate') or text(cover, './/reportCalendarOrQuarter')
                        if not period:
                            raise ValueError('13F reporting period is absent')
                        common['url'] = base + filename
                        parsed = thirteen_f(body, **common, actor=payload.get('name', cik),
                                            period_end=_iso_date(period),
                                            value_units=watch.options.get('value_units', 'as_filed'),
                                            amendment=row['form'].endswith('/A'))
                        expected = text(cover, './/tableEntryTotal')
                        if expected and int(expected) != len(parsed):
                            raise ValueError('13F information table count differs from cover summary')
                        break
                    if parsed is None:
                        raise ValueError('no XML information table found (legacy filing or format change)')
                events.extend(parsed)
                events.append(Event(accession, 'sec', 'filing', f'{payload.get("name", cik)} {row["form"]}',
                                    url, observed, published, payload.get('name', cik), '',
                                    {'form': row['form'], 'filing_date': row['filingDate'],
                                     'period_end': row.get('reportDate'), 'parse_status': 'structured',
                                     'parsed_records': len(parsed), 'accession': accession}))
                seen.add(accession)
                retry.pop(accession, None)
            except (FetchError, ValueError) as exc:
                # Do not checkpoint a failed filing: it is retried on the next poll.
                warnings.append(f'{accession}: {exc}')
                attempts = retry.get(accession, {}).get('attempts', 0) + 1
                wait = max(min(watch.interval_seconds * 2 ** min(attempts, 8), 86400),
                           getattr(exc, 'retry_after', None) or 0)
                retry[accession] = {'attempts': attempts, 'at': time.time() + wait}
                if isinstance(exc, FetchError) and exc.retry_after:
                    retry_after = exc.retry_after
                    break
        return Batch(events, {'seen': sorted(seen), 'retry': retry,
                              'remaining': len(set(candidates) - seen)}, warnings, retry_after)


class HouseSource:
    """Fetch the Clerk's annual XML index and readable PTR PDFs for an exact surname.

    Reports with unrecognized/scanned tables are retained as unparsed filing alerts.
    No Senate login/click-through or third-party paid feed is silently substituted.
    The source's use restrictions are recorded in each record's metadata.
    """

    def __init__(self, client):
        self.client = client

    def fetch(self, watch, state):
        year = watch.options.get('year')
        if not isinstance(year, int) or not 2008 <= year <= 2100:
            raise ValueError('House watch requires an explicit filing year (2008..2100)')
        response = self.client.get(f'https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP')
        with ZipFile(BytesIO(response.body)) as archive:
            names = [i for i in archive.infolist() if i.filename.lower() == f'{year}fd.xml']
            if len(names) != 1 or names[0].file_size > 20_000_000:
                raise ValueError('annual archive has no bounded XML index')
            root = xml(archive.read(names[0]))
        if root.tag != 'FinancialDisclosure':
            raise ValueError('unexpected House index root')
        seen, events, warnings = set(state.get('seen', [])), [], []
        retry = dict(state.get('retry', {}))
        retry_after = None
        rows = []
        for member in root.findall('Member'):
            if text(member, 'Last').casefold() != watch.target.casefold() or text(member, 'FilingType') != 'P':
                continue
            first = watch.options.get('first_name', '')
            if first and text(member, 'First').casefold() != first.casefold():
                continue
            doc_id = text(member, 'DocID')
            if not doc_id.isdigit():
                raise ValueError('invalid House document id')
            filing_date = _iso_date(text(member, 'FilingDate'))
            if doc_id in seen or filing_date < watch.options.get('since', ''):
                continue
            rows.append((filing_date, doc_id, member))
        maximum = int(watch.options.get('max_filings', 10))
        if not 1 <= maximum <= 100:
            raise ValueError('max_filings must be 1..100')
        eligible = [r for r in rows if retry.get(r[1], {}).get('at', 0) <= time.time()]
        for filing_date, doc_id, member in sorted(eligible, key=lambda r: (
                retry.get(r[1], {}).get('attempts', 0), r[0], r[1]))[:maximum]:
            actor = ' '.join((text(member, 'First'), text(member, 'Last')))
            url = f'https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf'
            observed = utcnow()
            parse_error = None
            try:
                body = self.client.get(url).body
            except FetchError as exc:
                warnings.append(f'{doc_id}: {exc}')
                attempts = retry.get(doc_id, {}).get('attempts', 0) + 1
                retry[doc_id] = {'attempts': attempts, 'at': time.time() + max(
                    min(watch.interval_seconds * 2 ** min(attempts, 8), 86400), exc.retry_after or 0)}
                if exc.retry_after:
                    retry_after = exc.retry_after
                    break
                continue
            try:
                content = pdf_text(body)
                transactions = house_transactions(content)
            except ValueError as exc:
                content, transactions, parse_error = '', [], str(exc)
            data = {'filing_date': filing_date, 'doc_id': doc_id, 'year': year,
                    'parse_status': 'extracted_reviewable' if transactions else 'unparsed',
                    'parsed_records': len(transactions), 'source_text': content, 'parse_error': parse_error,
                    'content_is_untrusted': True,
                    'use_restrictions_url': 'https://disclosures-clerk.house.gov/FinancialDisclosure/ViewSearch',
                    'published_time_unknown': True}
            events.append(Event(doc_id, 'house', 'congressional_filing', f'{actor}: periodic transaction report',
                                url, observed, None, actor, '', data))
            for i, row in enumerate(transactions):
                events.append(Event(f'{doc_id}:transaction:{i}', 'house', 'congressional_transaction',
                                    f'{actor}: {row["transaction_type"]} {row["asset"]}', url,
                                    observed, None, actor, row['symbol'], {**row, 'doc_id': doc_id,
                                    'filing_date': filing_date, 'published_time_unknown': True}))
            if not transactions:
                warnings.append(f'{doc_id}: no readable transaction rows; original text retained for review')
            seen.add(doc_id)
            retry.pop(doc_id, None)
        return Batch(events, {'seen': sorted(seen), 'retry': retry,
                              'remaining': len({r[1] for r in rows} - seen)}, warnings, retry_after)
