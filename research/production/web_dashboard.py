#!/usr/bin/env python3
"""Interactive Web Dashboard Server for fin-skills Multi-Asset Advisor.

Serves the interactive web UI and provides real-time API endpoints for market quotes
and allocation recalculations.

Zero external web framework dependencies (uses Python standard library http.server).

Usage:
    python3 research/production/web_dashboard.py --port 8088
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Add repo root to Python path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.production.daily_advisor import (fetch_latest_market_snapshot,
                                              generate_advisor_ticket)

HTML_PATH = Path(__file__).resolve().parent / "dashboard.html"


class AdvisorDashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html", "/dashboard", "/app"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if HTML_PATH.exists():
                with open(HTML_PATH, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"<h1>Error: dashboard.html not found</h1>")
            return

        if self.path.startswith("/api/snapshot"):
            try:
                snapshot = fetch_latest_market_snapshot()
                latest_date = list(snapshot.values())[0]["date"]
                payload = {
                    "status": "ok",
                    "date": latest_date,
                    "snapshot": snapshot,
                }
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return

        # Fallback to default static file serving
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/calculate":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                post_body = self.rfile.read(content_len)
                params = json.loads(post_body.decode("utf-8"))
                capital = float(params.get("capital", 100000.0))
                profile = str(params.get("profile", "conservative"))

                ticket = generate_advisor_ticket(capital=capital, profile=profile)

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(ticket, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()


def run_server(port: int = 8088):
    server_address = ("", port)
    httpd = HTTPServer(server_address, AdvisorDashboardHandler)
    hostname = os.environ.get("FIN_SKILLS_DASHBOARD_HOST") or socket.gethostname()
    print("=" * 80)
    print(f" 🚀 fin-skills 交互式实盘决策看板已成功启动！")
    print(f" 本地直连访问:  http://localhost:{port}/")
    print(f" 本机网络访问:  http://{hostname}:{port}/")
    print("=" * 80)
    print(" 按 Ctrl+C 停止服务。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n正在安全关闭 Web 服务...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start fin-skills Web Dashboard")
    parser.add_argument("--port", type=int, default=8088, help="Port to listen on (default: 8088)")
    args = parser.parse_args()
    run_server(args.port)
