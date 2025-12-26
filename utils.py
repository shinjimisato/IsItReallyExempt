"""
Utility functions for FINRA short sale analysis.
Includes trading calendar, date helpers, and formatting functions.
"""

import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def get_trading_days(start_date, end_date):
    """
    Get list of trading days between start_date and end_date using NYSE calendar.

    Args:
        start_date: Start date (datetime or string 'YYYYMMDD')
        end_date: End date (datetime or string 'YYYYMMDD')

    Returns:
        List of date strings in YYYYMMDD format
    """
    # Convert to datetime if strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y%m%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y%m%d')

    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index.tolist()

    # Convert to YYYYMMDD format strings
    trading_dates = [d.strftime('%Y%m%d') for d in trading_days]

    return trading_dates


def get_date_range_from_months(months_back):
    """
    Calculate start and end dates from months_back parameter.

    Args:
        months_back: Number of months to go back from today

    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=months_back)
    return start_date, end_date


def format_number(num):
    """
    Format number with commas for thousands.

    Args:
        num: Number to format

    Returns:
        Formatted string (e.g., "1,234,567")
    """
    if pd.isna(num):
        return "N/A"
    return f"{num:,.0f}"


def format_percentage(num, decimals=2):
    """
    Format number as percentage.

    Args:
        num: Number to format
        decimals: Number of decimal places

    Returns:
        Formatted string (e.g., "45.67%")
    """
    if pd.isna(num):
        return "N/A"
    return f"{num:.{decimals}f}%"


def format_date(date_obj):
    """
    Format datetime object as string.

    Args:
        date_obj: datetime object or pd.Timestamp

    Returns:
        Formatted date string (YYYY-MM-DD)
    """
    if pd.isna(date_obj):
        return "N/A"
    if isinstance(date_obj, str):
        return date_obj
    return date_obj.strftime('%Y-%m-%d')


def cache_not_expired(cache_file, hours=24):
    """
    Check if cache file is still valid (not expired).

    Args:
        cache_file: Path to cache file
        hours: Cache validity period in hours

    Returns:
        True if cache is still valid, False if expired
    """
    if not cache_file.exists():
        return False

    file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
    return file_age < timedelta(hours=hours)


def validate_symbol(symbol):
    """
    Validate and clean a stock symbol.

    Args:
        symbol: Stock symbol string

    Returns:
        Cleaned symbol (uppercase, stripped)
    """
    if not symbol or not isinstance(symbol, str):
        return None

    cleaned = symbol.strip().upper()

    # Basic validation - symbols are typically 1-5 characters
    if len(cleaned) < 1 or len(cleaned) > 5:
        return None

    # Check for valid characters (letters only, no numbers/special chars)
    if not cleaned.isalpha():
        return None

    return cleaned
