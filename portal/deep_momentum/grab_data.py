import pickle
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-script/1.0)"}


def get_sp500_tickers() -> list[str]:
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    table = pd.read_html(resp.text)[0]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


def fetch_sp500_data(period_days: int = 365 * 3) -> dict[str, pd.DataFrame]:
    tickers = get_sp500_tickers()
    end = datetime.today()
    start = end - timedelta(days=period_days)

    print(f"Fetching {len(tickers)} tickers from {start.date()} to {end.date()}...")

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    adj_close = raw["Close"]

    # Intraday return: (Close - Open) / Open
    intraday_returns = (raw["Close"] - raw["Open"]) / raw["Open"]

    return {
        "adj_close": adj_close,
        "Open": raw["Open"],
        "Close": raw["Close"],
        "intraday_returns": intraday_returns,
        "tickers": tickers,
        "start": start,
        "end": end,
    }


if __name__ == "__main__":
    output_path = "sp500_data.pkl"
    data = fetch_sp500_data()

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\nSaved to {output_path}")
    print(f"  adj_close shape:       {data['adj_close'].shape}")
    print(f"  intraday_returns shape: {data['intraday_returns'].shape}")
    print(f"  Tickers loaded:        {data['adj_close'].shape[1]}")
