"""Explicitly refresh the frozen ECB FX fixture; normal benchmarks never use the network."""
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
ROOT = Path(__file__).resolve().parent / "data"


def main():
    raw = urllib.request.urlopen(URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        df = pd.read_csv(archive.open("eurofxref-hist.csv"), usecols=["Date", "USD", "JPY", "GBP", "CHF"])
    df = df[(df.Date >= "2005-01-01") & (df.Date <= "2025-12-31")].sort_values("Date")
    if df.empty or df.isna().any().any() or df.Date.duplicated().any():
        raise ValueError("unexpected ECB fixture; inspect upstream before refreshing")
    ROOT.mkdir(exist_ok=True)
    path = ROOT / "ecb_fx.csv"
    df.to_csv(path, index=False, lineterminator="\n")
    manifest = {"source": "European Central Bank", "url": URL,
        "terms_url": "https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html",
        "attribution": "Source: European Central Bank. Reference rates are freely available from the ECB.",
        "transformations": "Selected Date/USD/JPY/GBP/CHF and 2005-2025 inclusive; sorted ascending; values not adjusted.",
        "units": "foreign currency units per euro; reference rates, not executable quotes",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "upstream_zip_sha256": hashlib.sha256(raw).hexdigest(),
        "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(df), "start": df.Date.iloc[0], "end": df.Date.iloc[-1]}
    (ROOT / "ecb_fx.provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
