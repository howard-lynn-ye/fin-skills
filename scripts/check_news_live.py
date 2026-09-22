"""Explicit network smoke check. Prints compact source health, never full articles."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fin_skills.collect import HttpClient, collect_news, news_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Optional GDELT query; omitted means official RSS only")
    parser.add_argument("--output", type=Path, help="Write a NEW JSON receipt; refuses overwrite")
    args = parser.parse_args()
    result = collect_news(query=args.query, client=HttpClient(timeout=10, attempts=1))
    digest = news_digest(result["events"], as_of=result["observed_at"])
    receipt = {"checked_at": result["observed_at"], "query": args.query,
               "sources": result["sources"], "records": len(result["events"]),
               "fresh_unique_articles": digest["unique_articles"],
               "exclusions": digest["excluded"], "scope": "connectivity and parsing, not coverage or efficacy"}
    body = json.dumps(receipt, indent=2, ensure_ascii=True, allow_nan=False)
    print(body)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(body + "\n")
    return int(any(s["status"] != "ok" for s in result["sources"]))


if __name__ == "__main__":
    raise SystemExit(main())
