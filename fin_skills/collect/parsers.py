"""Pure, bounded disclosure parsers. Unknown amounts and dates are never guessed."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from io import BytesIO

from .model import Event


def xml(body):
    raw = body.encode() if isinstance(body, str) else body
    if len(raw) > 20_000_000 or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError("oversized XML or document/entity declarations are not accepted")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError('invalid source XML') from exc
    for node in root.iter():
        node.tag = node.tag.split('}')[-1]
    return root


def text(node, path, default=""):
    value = node.findtext(path)
    return value.strip() if value else default


def number(value):
    if value is None or not str(value).strip():
        return None
    try:
        n = Decimal(str(value).replace(',', '').replace('$', '').strip())
    except InvalidOperation as exc:
        raise ValueError(f"invalid reported number: {value!r}") from exc
    if not n.is_finite():
        raise ValueError("reported numbers must be finite")
    return str(n)


def form4(body, *, accession, url, observed_at, published_at=None, filing_date=None):
    root = xml(body)
    if root.tag != 'ownershipDocument':
        raise ValueError("expected ownershipDocument XML")
    owners = [{'cik': text(n, 'reportingOwnerId/rptOwnerCik'),
               'name': text(n, 'reportingOwnerId/rptOwnerName'),
               'is_director': text(n, 'reportingOwnerRelationship/isDirector'),
               'is_officer': text(n, 'reportingOwnerRelationship/isOfficer'),
               'officer_title': text(n, 'reportingOwnerRelationship/officerTitle')}
              for n in root.findall('reportingOwner')]
    actor = '; '.join(x['name'] for x in owners)
    issuer = text(root, 'issuer/issuerName')
    symbol = text(root, 'issuer/issuerTradingSymbol')
    notes = {n.get('id'): ''.join(n.itertext()).strip() for n in root.findall('footnotes/footnote')}
    events = []
    for derivative, path in ((False, 'nonDerivativeTable/nonDerivativeTransaction'),
                             (True, 'derivativeTable/derivativeTransaction')):
        for i, node in enumerate(root.findall(path)):
            code = text(node, 'transactionCoding/transactionCode')
            data = {
                'accession': accession, 'form': text(root, 'documentType'),
                'filing_date': filing_date, 'transaction_date': text(node, 'transactionDate/value'),
                'owners': owners, 'issuer': issuer, 'issuer_cik': text(root, 'issuer/issuerCik'),
                'security': text(node, 'securityTitle/value'), 'derivative': derivative,
                'transaction_code': code,
                'acquired_disposed': text(node, 'transactionAmounts/transactionAcquiredDisposedCode/value'),
                'shares': number(text(node, 'transactionAmounts/transactionShares/value')),
                'price_per_share': number(text(node, 'transactionAmounts/transactionPricePerShare/value')),
                'shares_after': number(text(node, 'postTransactionAmounts/sharesOwnedFollowingTransaction/value')),
                'direct_indirect': text(node, 'ownershipNature/directOrIndirectOwnership/value'),
                'conversion_or_exercise_price': number(text(node, 'conversionOrExercisePrice/value')),
                'footnotes': notes, 'parse_status': 'structured',
                'not_a_realtime_execution_feed': True,
            }
            events.append(Event(f'{accession}:{"D" if derivative else "N"}:{i}', 'sec',
                                'insider_transaction', f'{actor}: {symbol} code {code}', url,
                                observed_at, published_at, actor, symbol, data))
    return events


def thirteen_f(body, *, accession, url, observed_at, actor, period_end,
               published_at=None, filing_date=None, value_units='as_filed', amendment=False):
    if value_units not in ('USD', 'USD_thousands', 'as_filed'):
        raise ValueError("13F value_units must be USD, USD_thousands, or as_filed")
    root = xml(body)
    if root.tag != 'informationTable':
        raise ValueError("expected informationTable XML")
    events = []
    for i, node in enumerate(root.findall('infoTable')):
        value = number(text(node, 'value'))
        shares = number(text(node, 'shrsOrPrnAmt/sshPrnamt'))
        data = {'accession': accession, 'period_end': period_end, 'filing_date': filing_date,
                'issuer': text(node, 'nameOfIssuer'), 'class': text(node, 'titleOfClass'),
                'cusip': text(node, 'cusip'), 'put_call': text(node, 'putCall'),
                'shares': shares, 'share_type': text(node, 'shrsOrPrnAmt/sshPrnamtType'),
                'value_reported': value, 'value_units': value_units,
                'value_usd': (str(Decimal(value) * (1000 if value_units == 'USD_thousands' else 1))
                              if value is not None and value_units != 'as_filed' else None),
                'discretion': text(node, 'investmentDiscretion'),
                'other_manager': text(node, 'otherManager'), 'amendment': amendment,
                'parse_status': 'structured', 'snapshot_not_trade': True}
        events.append(Event(f'{accession}:holding:{i}', 'sec', 'institutional_holding',
                            f'{actor}: {data["issuer"]} ({data["cusip"]})', url,
                            observed_at, published_at, actor, '', data))
    return events


def holdings_changes(previous, current):
    """Compare COMPLETE same-manager snapshots. Differences are not executed trades.

    Amendments need explicit reconstruction first; additions-only amendments cannot
    replace a full portfolio. Corporate actions can also change share counts.
    """
    old, new = list(previous), list(current)
    if not old or not new:
        raise ValueError("both snapshots must be nonempty; empty is not evidence of liquidation")
    def unpack(rows):
        return [r.to_dict() if isinstance(r, Event) else r for r in rows]
    old, new = unpack(old), unpack(new)
    all_rows = old + new
    if any(r['kind'] != 'institutional_holding' for r in all_rows):
        raise ValueError("only 13F holding snapshots can be compared")
    if len({r['actor'] for r in all_rows}) != 1:
        raise ValueError("snapshots must have the same reporting manager")
    for rows in (old, new):
        if len({r['data']['accession'] for r in rows}) != 1:
            raise ValueError("each snapshot must contain exactly one filing")
        if any(r['data'].get('amendment') for r in rows):
            raise ValueError("reconstruct amendments against the original filing before comparison")
    if old[0]['data']['period_end'] >= new[0]['data']['period_end']:
        raise ValueError("the current reporting period must follow the previous period")
    def aggregate(rows):
        out = defaultdict(Decimal)
        for r in rows:
            d = r['data']
            if d.get('shares') is None:
                raise ValueError("a missing share count cannot be treated as zero")
            key = tuple(d.get(k, '') for k in ('cusip', 'class', 'put_call', 'share_type'))
            out[key] += Decimal(d['shares'])
        return dict(out)
    a, b = aggregate(old), aggregate(new)
    return [dict(zip(('cusip', 'class', 'put_call', 'share_type'), key),
                 previous_shares=str(a.get(key, Decimal(0))), current_shares=str(b.get(key, Decimal(0))),
                 change_shares=str(b.get(key, Decimal(0)) - a.get(key, Decimal(0))),
                 change=('new_position' if key not in a else 'no_longer_reported' if key not in b
                         else 'increased' if b[key] > a[key] else 'decreased'),
                 inference='reported_snapshot_difference_not_an_execution')
            for key in sorted(set(a) | set(b)) if a.get(key, Decimal(0)) != b.get(key, Decimal(0))]


def pdf_text(body):
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise ImportError('House transaction parsing requires pip install "fin-skills[collect]"') from exc
    try:
        reader = PdfReader(BytesIO(body))
        if len(reader.pages) > 100:
            raise ValueError("disclosure PDF exceeds 100-page extraction limit")
        return '\n'.join(page.extract_text(extraction_mode='layout') or ''
                         for page in reader.pages).replace('\x00', '')
    except PdfReadError as exc:
        raise ValueError('source PDF could not be parsed') from exc


def house_transactions(body_text):
    """Extract rows only when the PDF's transaction/date/amount columns are readable.

    Retain the exact source row and uncertain asset/ticker fields. Scanned or changed
    layouts produce an explicit partial/unparsed document, never fabricated trades.
    """
    # The Clerk's generated PDFs have fixed columns; identify rows by their date pair,
    # then consume wrapped amounts/assets until the next row or the footnote section.
    pattern = re.compile(r'(?P<code>P|S(?:\s*\(partial\)|\s*\(full\))?|E)\s+'
                         r'(?P<date>\d{2}/\d{2}/\d{4})\s+'
                         r'(?P<notice>\d{2}/\d{2}/\d{4})\s+(?P<amount>\$[\d,]+(?:\s*-.*)?)', re.IGNORECASE)
    lines = body_text.splitlines()
    found = [(i, pattern.search(line)) for i, line in enumerate(lines) if pattern.search(line)]
    rows = []
    for k, (i, m) in enumerate(found):
        stop = found[k + 1][0] if k + 1 < len(found) else min(i + 12, len(lines))
        raw = '\n'.join(lines[max(0, i - 1):stop])
        amount_text = lines[i][m.start('amount'):]
        for following in lines[i + 1:min(i + 4, stop)]:
            # Layout extraction can shift a wrapped amount a few spaces. Only
            # consume an otherwise empty line containing money, or its amount column.
            continuation = following.strip()
            if re.fullmatch(r'\$[\d,]+(?:\s*[YN])?', continuation):
                amount_text += ' ' + continuation
            else:
                amount_text += ' ' + following[m.start('amount'):]
        money = re.findall(r'\$([\d,]+)', amount_text)
        prefix = lines[i][:m.start()].strip()
        owner_match = re.match(r'(?:\d+\s+)?(SP|JT|DC)\s+', prefix)
        owner = owner_match[1] if owner_match else ''
        asset = prefix[owner_match.end():].strip() if owner_match else re.sub(r'^\d+\s+', '', prefix)
        ticker = re.search(r'\(([A-Z][A-Z0-9.\-]{0,9})\)', asset)
        rows.append({'asset': asset, 'symbol': ticker[1] if ticker else '', 'owner': owner,
                     'transaction_type': m['code'], 'transaction_date': m['date'],
                     'notification_date': m['notice'],
                     'amount_lower': number(money[0]) if money else None,
                     'amount_upper': number(money[1]) if len(money) > 1 else None,
                     'amount_is_range': True, 'source_row': raw,
                     'parse_status': 'extracted_reviewable'})
    return rows
