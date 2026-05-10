"""
index_scraper.py
================
Pulls S&P 500 and Russell 2000 tickers + weights from iShares (BlackRock).

  S&P 500      → iShares Core S&P 500 ETF  (IVV)
  Russell 2000 → iShares Russell 2000 ETF  (IWM)

Usage:
    pip install requests pandas
    python index_scraper.py

Outputs:
    sp500.csv       — ticker, name, weight_pct, sector, asset_class, ...
    russell2000.csv — ticker, name, weight_pct, sector, asset_class, ...

Notes:
    - iShares CSVs have a few metadata rows above the real header; the
      parser skips them automatically.
    - Cash, futures, and other non-equity rows are dropped.
    - If you get 403 errors, visit ishares.com in a browser first to
      establish a session cookie, then re-run. Alternatively, add a
      Referer header (see SESSION setup below).
"""

import io
import logging
import time
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── HTTP session ──────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.ishares.com/",
        "Connection": "keep-alive",
    }
)

TIMEOUT = 30  # seconds

# ── iShares fund registry ─────────────────────────────────────────────────────
ISHARES_FUNDS = {
    "IVV": {
        "name": "iShares Core S&P 500 ETF",
        "index": "S&P 500",
        "path": "239726/ishares-core-sp-500-etf",
        "output": "sp500.csv",
    },
    "IJH": {
        "name": "iShares Core S&P Mid-Cap ETF",
        "index": "S&P Mid-Cap 400",
        "path": "239763/ishares-core-sp-mid-cap-etf",
        "output": "sp400_midcap.csv",
    },
    "IWM": {
        "name": "iShares Russell 2000 ETF",
        "index": "Russell 2000",
        "path": "239710/ishares-russell-2000-etf",
        "output": "russell2000.csv",
    },
}

ISHARES_BASE = "https://www.ishares.com/us/products"


def _csv_url(etf: str) -> str:
    path = ISHARES_FUNDS[etf]["path"]
    return (
        f"{ISHARES_BASE}/{path}/1467271812596.ajax"
        f"?fileType=csv&fileName={etf}_holdings&dataType=fund"
    )


# ── Column normalisation ──────────────────────────────────────────────────────
# iShares column names vary slightly between funds; map all known variants.
_COL_RENAME = {
    "ticker":          "ticker",
    "name":            "name",
    "weight_(%)":      "weight_pct",
    "weight":          "weight_pct",
    "sector":          "sector",
    "asset_class":     "asset_class",
    "market_value":    "market_value",
    "notional_value":  "notional_value",
    "shares":          "shares",
    "price":           "price",
    "location":        "location",
    "exchange":        "exchange",
    "currency":        "currency",
    "fx_rate":         "fx_rate",
    "market_currency": "market_currency",
    "accrual_date":    "accrual_date",
    "cusip":           "cusip",
    "isin":            "isin",
    "sedol":           "sedol",
}


def fetch_ishares(etf: str) -> Optional[pd.DataFrame]:
    """
    Download and parse the holdings CSV for an iShares ETF.

    Returns a clean DataFrame (equity positions only) with normalised
    column names, or None if the request or parse fails.
    """
    meta = ISHARES_FUNDS[etf]
    url = _csv_url(etf)
    log.info("Fetching %s (%s) ...", meta["name"], etf)
    log.info("  URL: %s", url)

    # ── Download ──────────────────────────────────────────────────────────────
    try:
        resp = SESSION.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("  Download failed: %s", exc)
        return None

    # ── Locate the real header row ────────────────────────────────────────────
    # iShares prepends a few metadata lines before the CSV data.
    # The real header is the first line that starts with "Ticker".
    lines = resp.text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith(("Ticker", '"Ticker"'))),
        None,
    )
    if header_idx is None:
        log.error("  Could not locate header row in %s CSV.", etf)
        return None

    # ── Parse CSV ─────────────────────────────────────────────────────────────
    csv_body = "\n".join(lines[header_idx:])
    try:
        df = pd.read_csv(io.StringIO(csv_body))
    except Exception as exc:
        log.error("  CSV parse error: %s", exc)
        return None

    # ── Normalise columns ─────────────────────────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in _COL_RENAME.items() if k in df.columns})

    # ── Drop trailing metadata rows iShares appends at the bottom ─────────────
    df = df[df["ticker"].notna() & (df["ticker"].str.strip() != "")]

    # ── Keep equity positions only ────────────────────────────────────────────
    if "asset_class" in df.columns:
        df = df[df["asset_class"].str.lower().str.contains("equity", na=False)]

    # ── Coerce numeric columns ────────────────────────────────────────────────
    for col in ("weight_pct", "market_value", "shares", "price", "fx_rate",
                "notional_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Attach metadata ───────────────────────────────────────────────────────
    df["etf"] = etf
    df["index"] = meta["index"]

    df = df.reset_index(drop=True)
    log.info("  %d equity holdings parsed.", len(df))
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path(".")

    for etf, meta in ISHARES_FUNDS.items():
        df = fetch_ishares(etf)

        if df is None or df.empty:
            log.error("No data returned for %s -- skipping.", etf)
            continue

        out_path = out_dir / meta["output"]
        df.to_csv(out_path, index=False)
        log.info("  Saved -> %s  (%d rows)", out_path, len(df))

        # Quick sanity check: top 10 by weight
        if "weight_pct" in df.columns:
            top10 = (
                df[["ticker", "name", "weight_pct"]]
                .dropna(subset=["weight_pct"])
                .sort_values("weight_pct", ascending=False)
                .head(10)
            )
            log.info("  Top 10 by weight:\n%s", top10.to_string(index=False))

        time.sleep(2)  # be polite between requests

    log.info("Done.")


if __name__ == "__main__":
    main()
