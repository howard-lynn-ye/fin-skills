"""Bounded, paced HTTP reads with conditional GET and retryable error reporting."""
from __future__ import annotations

import ipaddress
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import ClassVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class FetchError(RuntimeError):
    def __init__(self, message, *, status=None, retry_after=None):
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


def public_url(url: str, *, resolve: bool = False) -> str:
    """Reject credentials, local addresses and non-HTTP schemes, including redirects."""
    p = urlsplit(url)
    if p.scheme not in ("http", "https") or not p.hostname or p.username or p.password:
        raise ValueError("a public HTTP(S) URL without embedded credentials is required")
    if p.fragment:
        raise ValueError("URLs must not contain a fragment")
    if any(word in p.query.lower() for word in ("api_key=", "apikey=", "token=", "secret=")):
        raise ValueError("credentials must not appear in URLs")
    addresses = []
    try:
        addresses = [ipaddress.ip_address(p.hostname)]
    except ValueError:
        if p.hostname.lower() == "localhost" or p.hostname.lower().endswith(".localhost"):
            raise ValueError("local addresses are not collection sources")
        if resolve:
            addresses = [ipaddress.ip_address(x[4][0]) for x in
                         socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))]
    if any(not a.is_global for a in addresses):
        raise ValueError("private/reserved addresses are not collection sources")
    return url


@dataclass
class Response:
    status: int
    body: bytes
    headers: dict = field(default_factory=dict)

    def json(self):
        return json.loads(self.body)


class _Redirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl, resolve=True)
        if urlsplit(newurl).netloc != urlsplit(req.full_url).netloc:
            raise FetchError("cross-host redirect refused; use the canonical source URL")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _send(url, headers, timeout, max_bytes):
    public_url(url, resolve=True)
    try:
        response = build_opener(_Redirect()).open(Request(url, headers=headers), timeout=timeout)
    except HTTPError as exc:
        response = exc
    with response:
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise FetchError("response exceeds configured byte limit")
        return Response(response.status, body, {k.lower(): v for k, v in response.headers.items()})


class HttpClient:
    """No request happens on construction. ``sender`` permits offline transport tests.

    Pacing is shared by host within this process. Operators with multiple processes
    must additionally coordinate their account/IP budget. SEC uses a conservative
    one request/second default here, below its published maximum.
    """

    _lock = threading.Lock()
    _last: ClassVar[dict[str, float]] = {}

    def __init__(self, user_agent="fin-skills/0.1 (+https://github.com/howard-lynn-ye/fin-skills)",
                 *, sender=None, timeout=25, max_bytes=8_000_000, attempts=3,
                 min_interval=1.0, sleep=time.sleep, clock=time.monotonic):
        if not 0 < timeout <= 120 or not 0 < max_bytes <= 50_000_000 or not 1 <= attempts <= 5:
            raise ValueError("invalid HTTP timeout, size or attempt limit")
        if min_interval < 0:
            raise ValueError("min_interval cannot be negative")
        self.user_agent, self.sender = user_agent, sender or _send
        self.timeout, self.max_bytes, self.attempts = timeout, max_bytes, attempts
        self.min_interval, self.sleep, self.clock = min_interval, sleep, clock

    def get(self, url, headers=None):
        public_url(url)
        host = urlsplit(url).hostname
        headers = {"User-Agent": self.user_agent, "Accept": "*/*", **(headers or {})}
        for attempt in range(self.attempts):
            with self._lock:
                delay = self.min_interval - (self.clock() - self._last.get(host, -1e20))
                if delay > 0:
                    self.sleep(delay)
                self._last[host] = self.clock()
            try:
                r = self.sender(url, headers, self.timeout, self.max_bytes)
                r.headers = {k.lower(): v for k, v in r.headers.items()}
                if len(r.body) > self.max_bytes:
                    raise FetchError("response exceeds configured byte limit")
                if r.status in (200, 304):
                    return r
                retry = r.headers.get("retry-after")
                try:
                    delay = max(0, float(retry))
                except (TypeError, ValueError):
                    try:
                        delay = max(0, parsedate_to_datetime(retry).timestamp() - time.time())
                    except (TypeError, ValueError, OverflowError):
                        delay = 2 ** attempt
                if r.status not in (429, 500, 502, 503, 504):
                    raise FetchError(f"{host} returned HTTP {r.status}", status=r.status)
                error = FetchError(f"{host} returned HTTP {r.status}", status=r.status,
                                   retry_after=delay)
            except (URLError, TimeoutError, OSError) as exc:
                delay = 2 ** attempt
                error = FetchError(f"{host}: transport failed ({type(exc).__name__})")
            if attempt + 1 == self.attempts or delay > 30:
                raise error
            self.sleep(delay)
        raise AssertionError("unreachable")
