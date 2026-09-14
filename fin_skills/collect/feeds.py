"""RSS/Atom, static public pages and public Bluesky author feeds."""
from __future__ import annotations

from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from .http import FetchError, public_url
from .model import Batch, Event, digest, utc, utcnow
from .parsers import text, xml


def _date(value):
    if not value:
        return None
    try:
        return utc(value)
    except ValueError:
        try:
            stamp = parsedate_to_datetime(value)
            return utc(stamp) if stamp.tzinfo is not None else None
        except (ValueError, TypeError):
            return None


class _Readable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.links = [], []
        self.skip, self.title, self.in_title = 0, '', False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript', 'svg'):
            self.skip += 1
        if tag == 'title':
            self.in_title = True
        if tag == 'a':
            self.links.extend(v for k, v in attrs if k == 'href' and v)
        if tag in ('p', 'div', 'article', 'section', 'br', 'h1', 'h2', 'li'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript', 'svg'):
            self.skip = max(0, self.skip - 1)
        if tag == 'title':
            self.in_title = False

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)
            if self.in_title:
                self.title += data

    def readable(self):
        return '\n'.join(' '.join(s.split()) for s in ''.join(self.parts).splitlines() if s.strip())


def readable_html(value):
    p = _Readable()
    p.feed(value)
    return p


def _headers(state):
    return {header: state[key] for key, header in (('etag', 'If-None-Match'),
            ('modified', 'If-Modified-Since')) if state.get(key)}


def _state(response, state):
    return {**state, 'etag': response.headers.get('etag'),
            'modified': response.headers.get('last-modified')}


class RSSSource:
    def __init__(self, client):
        self.client = client

    def fetch(self, watch, state):
        response = self.client.get(watch.target, _headers(state))
        if response.status == 304:
            return Batch(state=state)
        root = xml(response.body)
        if root.tag not in ('rss', 'feed', 'RDF'):
            raise ValueError("source returned neither RSS nor Atom")
        entries = root.findall('.//item') if root.tag != 'feed' else root.findall('entry')
        events, observed = [], utcnow()
        for node in entries:
            link = text(node, 'link')
            if not link:
                link = next((n.get('href') for n in node.findall('link')
                             if n.get('rel', 'alternate') == 'alternate'), '')
            has_link = bool(link)
            link = urljoin(watch.target, link) if link else watch.target
            public_url(link.split('#')[0])
            title = text(node, 'title')
            description = text(node, 'description') or text(node, 'summary') or text(node, 'content')
            published = text(node, 'pubDate') or text(node, 'published') or text(node, 'date')
            identity = text(node, 'guid') or text(node, 'id') or (
                link if has_link else 'rss:' + digest([title, published, description]))
            events.append(Event(identity, 'rss', 'news', title, link, observed,
                                _date(published), text(node, 'author/name') or text(node, 'creator'), '',
                                {'text': readable_html(description).readable(),
                                 'source_date': published, 'updated_at': _date(text(node, 'updated')),
                                 'content_is_untrusted': True}))
        return Batch(events, _state(response, state))


class PageSource:
    """A bounded same-origin static crawler. It does not execute JavaScript or bypass login."""
    def __init__(self, client):
        self.client = client

    def fetch(self, watch, state):
        public_url(watch.target)
        origin = urlsplit(watch.target)
        robots_url = f'{origin.scheme}://{origin.netloc}/robots.txt'
        try:
            response = self.client.get(robots_url)
            robot = RobotFileParser()
            robot.parse(response.body.decode('utf-8', errors='replace').splitlines())
        except FetchError as exc:
            if exc.status not in (404, 410):
                raise
            robot = RobotFileParser()
            robot.parse([])
        pages = int(watch.options.get('max_pages', 1))
        if not 1 <= pages <= 20:
            raise ValueError("max_pages must be 1..20")
        queue, visited, events, warnings = [watch.target], set(), [], []
        retry_after = None
        while queue and len(events) < pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            if not robot.can_fetch(self.client.user_agent, url):
                if url == watch.target:
                    raise FetchError("robots.txt disallows the requested page")
                continue
            crawl_delay = robot.crawl_delay(self.client.user_agent) or robot.crawl_delay('*')
            if crawl_delay:
                if crawl_delay > 30:
                    raise FetchError("robots crawl delay requires a slower dedicated schedule", retry_after=crawl_delay)
                self.client.sleep(crawl_delay)
            try:
                response = self.client.get(url)
            except FetchError as exc:
                if url == watch.target:
                    raise
                warnings.append(f'discovered page could not be read: {exc}')
                if exc.retry_after:
                    retry_after = exc.retry_after
                    break
                continue
            if 'html' not in response.headers.get('content-type', '').lower():
                if url == watch.target:
                    raise ValueError("page source requires text/html")
                continue
            parsed = readable_html(response.body.decode('utf-8', errors='replace'))
            content = parsed.readable()
            if not content:
                if url == watch.target:
                    raise ValueError("page has no readable static text; it may require JavaScript")
                warnings.append(f'{url}: no readable static text; may require JavaScript')
                continue
            events.append(Event(url, 'page', 'web_page', parsed.title or url, url, utcnow(),
                                data={'text': content, 'content_is_untrusted': True}))
            for href in parsed.links:
                target = urljoin(url, href).split('#')[0]
                destination = urlsplit(target)
                if (destination.scheme, destination.netloc) == (origin.scheme, origin.netloc) and target not in visited:
                    try:
                        public_url(target)
                    except ValueError:
                        continue
                    if len(queue) < 200:
                        queue.append(target)
        return Batch(events, {'last_crawl_pages': len(events)}, warnings, retry_after)


class BlueskySource:
    def __init__(self, client):
        self.client = client

    def fetch(self, watch, state):
        seen = set(state.get('seen', []))
        cursor, events = None, []
        max_pages = int(watch.options.get('max_pages', 5))
        if not 1 <= max_pages <= 20:
            raise ValueError('max_pages must be 1..20')
        reached = False
        for _ in range(max_pages):
            query = {'actor': watch.target, 'limit': 100, 'filter': 'posts_no_replies'}
            if cursor:
                query['cursor'] = cursor
            payload = self.client.get('https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?'
                                      + urlencode(query)).json()
            for entry in payload.get('feed', []):
                post = entry['post']
                if entry.get('reason'):       # repost isn't an assertion by its author
                    continue
                identity = post['uri']
                if identity in seen:
                    reached = True
                record, author = post.get('record', {}), post.get('author', {})
                url = f'https://bsky.app/profile/{author["did"]}/post/{identity.rsplit("/", 1)[-1]}'
                events.append(Event(identity, 'bluesky', 'social_post', record.get('text', '')[:180],
                                    url, utcnow(), _date(record.get('createdAt')), author.get('handle', ''), '',
                                    {'text': record.get('text', ''), 'indexed_at': post.get('indexedAt'),
                                     'cid': post.get('cid'), 'content_is_untrusted': True,
                                     'not_evidence_of_a_trade': True}))
            cursor = payload.get('cursor')
            if reached or not cursor:
                break
        if seen and cursor and not reached:
            raise ValueError('author feed page limit reached before checkpoint; raise max_pages to avoid a gap')
        return Batch(events, {'seen': list(dict.fromkeys([e.id for e in events] + list(seen)))[:2000]})
