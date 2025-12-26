"""
Data preprocessing and validation module.
Converts raw FINRA data into analysis-ready format.
"""

import pandas as pd
import numpy as np


def preprocess_data(df):
    """
    Clean and preprocess raw FINRA data.

    Adds derived metrics:
    - ShortPercent: Short volume as % of total volume
    - ShortExemptPercent: Short exempt volume as % of total volume
    - ShortExemptRatio: Short exempt as % of short volume
    - NonShortVolume: Total volume minus short volume

    Also adds time-based grouping columns for analysis.

    Args:
        df: Raw FINRA DataFrame

    Returns:
        Preprocessed DataFrame with derived metrics
    """
    if df.empty:
        return df

    df = df.copy()

    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

    # Calculate derived metrics
    df['ShortPercent'] = (df['ShortVolume'] / df['TotalVolume'] * 100).round(2)
    df['ShortExemptPercent'] = (df['ShortExemptVolume'] / df['TotalVolume'] * 100).round(4)

    # Handle division by zero for ShortExemptRatio
    df['ShortExemptRatio'] = np.where(
        df['ShortVolume'] > 0,
        (df['ShortExemptVolume'] / df['ShortVolume'] * 100).round(4),
        0
    )

    df['NonShortVolume'] = df['TotalVolume'] - df['ShortVolume']

    # Add time-based columns for grouping
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['WeekNumber'] = df['Date'].dt.isocalendar().week
    df['YearWeek'] = df['Date'].dt.strftime('%Y-W%W')
    df['Month'] = df['Date'].dt.to_period('M')
    df['YearMonth'] = df['Date'].dt.strftime('%Y-%m')

    # Sort by date and symbol
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

    return df


def validate_data(df):
    """
    Validate data integrity and return issues found.

    Checks for:
    - ShortVolume > TotalVolume (should never happen)
    - ShortExemptVolume > ShortVolume (data error)
    - Missing required columns

    Args:
        df: Preprocessed DataFrame

    Returns:
        Dictionary with validation results
    """
    issues = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    if df.empty:
        issues['valid'] = False
        issues['errors'].append("DataFrame is empty")
        return issues

    # Check required columns
    required_cols = ['Date', 'Symbol', 'ShortVolume', 'ShortExemptVolume', 'TotalVolume']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        issues['valid'] = False
        issues['errors'].append(f"Missing required columns: {', '.join(missing_cols)}")
        return issues

    # Check for invalid short volume
    invalid_short = df[df['ShortVolume'] > df['TotalVolume']]
    if len(invalid_short) > 0:
        issues['valid'] = False
        issues['errors'].append(
            f"Found {len(invalid_short)} rows where ShortVolume > TotalVolume"
        )

    # Check for invalid exempt volume
    invalid_exempt = df[df['ShortExemptVolume'] > df['ShortVolume']]
    if len(invalid_exempt) > 0:
        issues['warnings'].append(
            f"Found {len(invalid_exempt)} rows where ShortExemptVolume > ShortVolume"
        )

    # Check for missing data
    null_counts = df[required_cols].isnull().sum()
    null_cols = null_counts[null_counts > 0]

    if len(null_cols) > 0:
        issues['warnings'].append(
            f"Found null values: {null_cols.to_dict()}"
        )

    return issues


def merge_multi_symbol_data(data_dict):
    """
    Merge data from multiple symbols for comparison.

    Args:
        data_dict: Dictionary of {symbol: DataFrame}

    Returns:
        Merged DataFrame with Symbol column preserved
    """
    if not data_dict:
        return pd.DataFrame()

    all_dfs = []
    for symbol, df in data_dict.items():
        df_copy = df.copy()
        df_copy['Symbol'] = symbol
        all_dfs.append(df_copy)

    return pd.concat(all_dfs, ignore_index=True)


def calculate_day_over_day_changes(df):
    """
    Calculate day-over-day percentage changes for volume metrics.

    Args:
        df: Preprocessed DataFrame (single symbol)

    Returns:
        DataFrame with added change columns
    """
    df = df.copy().sort_values('Date')

    df['ShortVolumePctChange'] = df['ShortVolume'].pct_change() * 100
    df['TotalVolumePctChange'] = df['TotalVolume'].pct_change() * 100
    df['ShortExemptVolumePctChange'] = df['ShortExemptVolume'].pct_change() * 100

    return df


def filter_by_date_range(df, start_date, end_date):
    """
    Filter DataFrame by date range.

    Args:
        df: DataFrame with Date column
        start_date: Start date (datetime or string)
        end_date: End date (datetime or string)

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    df = df.copy()

    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    # Convert filter dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    return df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
