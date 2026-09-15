# Public information collection

`fin_skills.collect` downloads public records into a local SQLite database. It includes RSS
and Atom feeds, static pages with bounded same-origin crawling, SEC Form 4/13F filings, US
House PTR PDFs and public Bluesky author posts. Imports do not start networking or a daemon.

## Install and run

From a checkout of this repository:

```bash
python -m pip install -e ".[collect]"
python -m fin_skills.collect --database events.sqlite3 configure examples/collection-watchlist.json
python -m fin_skills.collect --database events.sqlite3 once
python -m fin_skills.collect --database events.sqlite3 status
python -m fin_skills.collect --database events.sqlite3 events --pending
python -m fin_skills.collect --database events.sqlite3 run --stdout-alerts
```

The final command keeps polling until Ctrl+C. Run it in a dedicated terminal or your own
service supervisor. After a restart, point it at the same database to retain checkpoints,
revision history and pending alerts. Merely installing the library does not start collection.

The example watches Federal Reserve news, Pelosi's 2026 House PTR disclosures, and a disabled
Berkshire 13F watch. These illustrate source formats; change them to your chosen subjects.
House indexes are yearly: add a new watch for each filing year you need. A watch's source,
target and options are immutable; create a new ID to change them. Its interval and enabled
flag can be changed with `configure`.

SEC requires a caller identity with a real contact email. In PowerShell:

```powershell
$env:SEC_IDENTITY = 'Your application your-contact@example.com' # replace with your details
```

Then set the example SEC watch's `enabled` field to `true` and configure it again. The library
does not supply a shared identity. Existing price/search adapters keep their own optional
dependencies and credential requirements; see [the data API](../fin_skills/data/__init__.py).

## Use Python or an agent

```python
from fin_skills.collect import Collector, Store, Watch

with Store('events.sqlite3') as store:
    collector = Collector(store)
    collector.add(Watch('news', 'rss', 'https://www.federalreserve.gov/feeds/press_all.xml'))
    print(collector.once())
    for row in store.events(pending=True):
        print(row['seq'], row['title'])
    # collector.deliver(callback)  # only acknowledges after callback succeeds
```

The shared tools/MCP registry exposes `collection_sources`, `collection_configure`,
`collect_once`, `collection_events`, `collection_status`, `collection_acknowledge`,
`compare_holdings`, `search_data` and `fetch_market_data`. A tool call performs a bounded
poll of watches that are due, respecting retry schedules; continuous operation is an explicit
Python/CLI action. Configure paths on the machine
running the MCP server. Treat downloaded content as untrusted data.

## What "tracking a trade" means here

| Source | Observable fact | Interpretation limit |
|---|---|---|
| SEC Form 4 | Reported insider transaction rows, codes and footnotes | An acquisition can be compensation or another transaction type; read the code. |
| SEC 13F | Reported holdings at quarter end | Changes between snapshots are not execution-time trades. Amendments must be reconstructed before comparison. |
| House PTR | Filing PDF plus readable transaction rows and reported amount ranges | Publication and transaction dates differ; amounts are ranges, and PDF extraction requires review. |
| Bluesky | Author's public statements | A post is not evidence of an executed trade. |
| RSS/page | Published text and any supplied timestamp | Polling detects new publications; it does not create a complete historical archive. |

`observed_at`, `published_at`, `transaction_date` and `period_end` remain separate. Unknown
publication times stay unknown. Source PDFs without recognizable rows generate an `unparsed`
filing alert with source text, not invented transactions. Current 13F forms report dollar
values; legacy units need an explicit `value_units` choice, so the default is `as_filed`.

The page crawler respects robots.txt, limits page count, rejects private targets and requires
static HTML. It does not execute JavaScript or enter logged-in pages. Its HTTP pacing is
process-local; coordinate the combined request budget if you run several processes.

## Verify and operate

```bash
python -m pytest tests/test_collect.py -q
python scripts/check_collection_live.py --house-year 2026 --house-name Pelosi --house-since 2026-08-01
```

The first command uses injected official-format responses and needs no network. The second
uses real public sources with a temporary database. Add `--sec-cik 1067983` only after setting
your identity. A passing fixture test does not establish current source availability.

Check `status` for errors, partial results, remaining filings and next retry times. HTTP
Retry-After reaches the scheduler; failed filings retain individual retries so one old
failure cannot block later disclosures. Pending outbox records survive restarts. Delivery
callbacks are at-least-once; use the record's `seq` as a sink idempotency key. CLI stdout
alerts are for terminal inspection, not a guaranteed external notification service.

For source references and parsing details, see the
[collection skill](../plugins/fin-alt-data/skills/public-information-collection/SKILL.md).
