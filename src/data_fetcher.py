"""
FINRA data fetching and caching module.
Handles scraping from FINRA's daily short sale reports with two-tier caching.
"""

import streamlit as st
import pandas as pd
import requests
from io import StringIO
from pathlib import Path
from datetime import datetime
import time
from .utils import get_trading_days, cache_not_expired


# Constants
FINRA_URL_TEMPLATE = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
REQUEST_DELAY = 0.25  # Seconds between requests
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def fetch_single_day(symbol, date_str):
    """
    Fetch FINRA short sale data for a specific date and symbol.

    Args:
        symbol: Stock ticker symbol
        date_str: Date in YYYYMMDD format

    Returns:
        DataFrame with short sale data or None if fetch fails
    """
    url = FINRA_URL_TEMPLATE.format(date=date_str)

    try:
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            # Parse pipe-delimited data
            df = pd.read_csv(
                StringIO(response.text),
                sep='|',
                dtype={
                    'Date': str,
                    'Symbol': str,
                    'ShortVolume': 'Int64',
                    'ShortExemptVolume': 'Int64',
                    'TotalVolume': 'Int64',
                    'Market': str
                }
            )

            # Filter for target symbol
            df = df[df['Symbol'] == symbol.upper()]

            return df if len(df) > 0 else None

        elif response.status_code == 404:
            return None  # Data not available (holiday, weekend, future date)
        else:
            return None

    except Exception as e:
        # Silently fail - will be reported in aggregate
        return None


def scrape_finra_data(symbol, trading_dates, progress_bar=None, status_text=None):
    """
    Scrape FINRA data for all trading dates.

    Args:
        symbol: Stock ticker symbol
        trading_dates: List of dates in YYYYMMDD format
        progress_bar: Optional Streamlit progress bar
        status_text: Optional Streamlit text element for status updates

    Returns:
        Combined DataFrame with all data
    """
    all_data = []
    successful = 0
    failed = 0

    for i, date_str in enumerate(trading_dates):
        if status_text:
            status_text.text(f"Fetching {symbol} data for {date_str}...")

        df = fetch_single_day(symbol, date_str)

        if df is not None and len(df) > 0:
            all_data.append(df)
            successful += 1
        else:
            failed += 1

        # Update progress
        if progress_bar:
            progress_bar.progress((i + 1) / len(trading_dates))

        # Rate limiting
        time.sleep(REQUEST_DELAY)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_finra_data_cached(symbol, start_date, end_date):
    """
    Fetch FINRA data with two-tier caching.

    Uses:
    1. Streamlit's in-memory cache (session-level)
    2. File-based cache (persistent across sessions)

    Args:
        symbol: Stock ticker symbol
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format

    Returns:
        DataFrame with FINRA short sale data
    """
    cache_file = CACHE_DIR / f"{symbol}_{start_date}_{end_date}.parquet"

    # Check file cache first
    if cache_file.exists() and cache_not_expired(cache_file, hours=24):
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            # If cache read fails, re-fetch
            pass

    # Generate trading dates
    trading_dates = get_trading_days(start_date, end_date)

    # Fetch from FINRA with progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        data = scrape_finra_data(symbol, trading_dates, progress_bar, status_text)

        # Save to file cache
        if not data.empty:
            try:
                data.to_parquet(cache_file)
            except Exception:
                # If cache write fails, continue anyway
                pass

        return data

    finally:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()


def clear_cache_for_symbol(symbol):
    """
    Clear all cached data for a specific symbol.

    Args:
        symbol: Stock ticker symbol
    """
    cache_files = CACHE_DIR.glob(f"{symbol}_*.parquet")
    for file in cache_files:
        try:
            file.unlink()
        except Exception:
            pass


def clear_old_cache(days=7):
    """
    Clear cache files older than specified days.

    Args:
        days: Number of days after which to delete cache files
    """
    from datetime import timedelta

    cutoff = datetime.now() - timedelta(days=days)

    cache_files = CACHE_DIR.glob("*.parquet")
    for file in cache_files:
        try:
            file_time = datetime.fromtimestamp(file.stat().st_mtime)
            if file_time < cutoff:
                file.unlink()
        except Exception:
            pass


def get_cache_info():
    """
    Get information about cached data.

    Returns:
        Dictionary with cache statistics
    """
    cache_files = list(CACHE_DIR.glob("*.parquet"))

    if not cache_files:
        return {
            'total_files': 0,
            'total_size_mb': 0,
            'oldest_file': None,
            'newest_file': None
        }

    total_size = sum(f.stat().st_size for f in cache_files)

    file_times = [(f, datetime.fromtimestamp(f.stat().st_mtime)) for f in cache_files]
    file_times.sort(key=lambda x: x[1])

    return {
        'total_files': len(cache_files),
        'total_size_mb': round(total_size / (1024 * 1024), 2),
        'oldest_file': file_times[0][1].strftime('%Y-%m-%d %H:%M') if file_times else None,
        'newest_file': file_times[-1][1].strftime('%Y-%m-%d %H:%M') if file_times else None
    }
