"""Sentiment data fetcher.

Sources:
  1. Crypto Fear & Greed Index (alternative.me) — always available, no key.
  2. EODHD news API + VADER compound score — requires EODHD_API_KEY in .env.

Both sources produce a daily score joined on date.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)

_FNG_URL = "https://api.alternative.me/fng/"
_EODHD_NEWS_URL = "https://eodhistoricaldata.com/api/news"


class SentimentDownloader:
    """Downloads and merges daily sentiment signals.

    Args:
        eodhd_api_key: EODHD API key (empty string = skip EODHD).
        request_delay: Seconds between EODHD paginated calls.
    """

    def __init__(self, eodhd_api_key: str = "", request_delay: float = 1.0) -> None:
        self.eodhd_api_key = eodhd_api_key
        self.request_delay = request_delay

    # ── Public interface ──────────────────────────────────────────────────────

    def download(self, days: int = 90) -> pd.DataFrame:
        """Download and merge all daily sentiment signals.

        Args:
            days: Number of calendar days to retrieve.

        Returns:
            DataFrame indexed by date (UTC, daily frequency) with columns:
            - fear_greed (0–100)
            - fear_greed_label (str)
            - eodhd_vader_score (float, NaN if key not available)
        """
        logger.info("Downloading sentiment data for last %d days", days)

        fng = self._download_fear_greed(days)

        if self.eodhd_api_key:
            vader = self._download_eodhd_sentiment(days)
            df = fng.join(vader, how="left")
        else:
            logger.warning(
                "EODHD_API_KEY not set — skipping news sentiment. "
                "Only Fear & Greed Index will be used."
            )
            df = fng
            df["eodhd_vader_score"] = float("nan")

        df = df.sort_index()
        logger.info(
            "Sentiment ready: %d days | Fear&Greed coverage: %d | EODHD coverage: %d",
            len(df),
            df["fear_greed"].notna().sum(),
            df["eodhd_vader_score"].notna().sum(),
        )
        return df

    # ── Fear & Greed ──────────────────────────────────────────────────────────

    def _download_fear_greed(self, days: int) -> pd.DataFrame:
        """Fetch Crypto Fear & Greed Index from alternative.me.

        Args:
            days: Number of past days to retrieve.

        Returns:
            DataFrame indexed by date with columns: fear_greed, fear_greed_label.
        """
        logger.info("Fetching Fear & Greed Index (last %d days) ...", days)
        try:
            resp = requests.get(
                _FNG_URL,
                params={"limit": days, "format": "json"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Fear & Greed API failed: {exc}") from exc

        records = []
        for item in data.get("data", []):
            dt = datetime.fromtimestamp(int(item["timestamp"]), tz=timezone.utc).date()
            records.append(
                {
                    "date": dt,
                    "fear_greed": int(item["value"]),
                    "fear_greed_label": item["value_classification"],
                }
            )

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date").sort_index()
        logger.info("Fear & Greed: %d daily scores retrieved", len(df))
        return df

    # ── EODHD + VADER ─────────────────────────────────────────────────────────

    def _download_eodhd_sentiment(self, days: int) -> pd.DataFrame:
        """Fetch BTC news from EODHD and score with VADER.

        Retrieves news articles in paginated batches, applies VADER sentiment
        analysis to each article's title + description, and aggregates to a
        daily average compound score.

        Args:
            days: Number of past days to retrieve.

        Returns:
            DataFrame indexed by date with column: eodhd_vader_score.
        """
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
        except ImportError:
            logger.error("vaderSentiment not installed. Run: pip install vaderSentiment")
            return pd.DataFrame(columns=["eodhd_vader_score"])

        analyzer = SentimentIntensityAnalyzer()

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days)

        logger.info(
            "Fetching EODHD news for BTC: %s → %s", start_date, end_date
        )

        articles = self._fetch_eodhd_news(
            symbol="BTC.CC",
            from_date=start_date.isoformat(),
            to_date=end_date.isoformat(),
        )

        if not articles:
            logger.warning("No EODHD articles retrieved — check API key and symbol.")
            return pd.DataFrame(columns=["eodhd_vader_score"])

        records = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('content', '')}"
            score = analyzer.polarity_scores(text)["compound"]
            pub_date = article.get("date", "")[:10]  # YYYY-MM-DD
            records.append({"date": pub_date, "compound": score})

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        daily = df.groupby("date")["compound"].mean().rename("eodhd_vader_score")
        daily = daily.sort_index()

        logger.info(
            "EODHD sentiment: %d articles → %d daily scores",
            len(articles),
            len(daily),
        )
        return daily.to_frame()

    def _fetch_eodhd_news(
        self, symbol: str, from_date: str, to_date: str
    ) -> list[dict]:
        """Paginated EODHD news fetch.

        Args:
            symbol: Asset symbol (e.g. 'BTC.CC').
            from_date: ISO date string (YYYY-MM-DD).
            to_date: ISO date string (YYYY-MM-DD).

        Returns:
            List of article dicts with at minimum 'title', 'date', 'content'.
        """
        articles: list[dict] = []
        offset = 0
        limit = 50  # Max per EODHD call

        while True:
            params = {
                "api_token": self.eodhd_api_key,
                "s": symbol,
                "from": from_date,
                "to": to_date,
                "limit": limit,
                "offset": offset,
                "fmt": "json",
            }
            try:
                resp = requests.get(_EODHD_NEWS_URL, params=params, timeout=20)
                resp.raise_for_status()
                batch = resp.json()
            except requests.RequestException as exc:
                logger.error("EODHD API call failed (offset=%d): %s", offset, exc)
                break

            if not isinstance(batch, list) or not batch:
                break  # No more data

            articles.extend(batch)
            logger.debug("  EODHD: fetched %d articles (offset=%d)", len(batch), offset)

            if len(batch) < limit:
                break  # Last page

            offset += limit
            time.sleep(self.request_delay)

        return articles
