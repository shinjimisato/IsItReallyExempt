"""
Analysis functions for FINRA short sale data.
Includes statistical analysis, anomaly detection, and multi-symbol comparison.
"""

import pandas as pd
import numpy as np


def generate_summary_stats(df):
    """
    Generate comprehensive summary statistics.

    Args:
        df: Preprocessed DataFrame (single symbol)

    Returns:
        Dictionary with summary statistics
    """
    if df.empty:
        return {}

    stats = {
        'trading_days': len(df),
        'date_range': {
            'start': df['Date'].min(),
            'end': df['Date'].max()
        },
        'volume': {
            'total': int(df['TotalVolume'].sum()),
            'avg': float(df['TotalVolume'].mean()),
            'median': float(df['TotalVolume'].median()),
            'std': float(df['TotalVolume'].std())
        },
        'short_volume': {
            'total': int(df['ShortVolume'].sum()),
            'avg': float(df['ShortVolume'].mean()),
            'median': float(df['ShortVolume'].median()),
            'std': float(df['ShortVolume'].std())
        },
        'short_percent': {
            'avg': float(df['ShortPercent'].mean()),
            'median': float(df['ShortPercent'].median()),
            'min': float(df['ShortPercent'].min()),
            'max': float(df['ShortPercent'].max()),
            'std': float(df['ShortPercent'].std())
        },
        'short_exempt': {
            'total': int(df['ShortExemptVolume'].sum()),
            'avg': float(df['ShortExemptVolume'].mean()),
            'days_with_exempt': int(len(df[df['ShortExemptVolume'] > 0])),
            'avg_exempt_ratio': float(df['ShortExemptRatio'].mean())
        }
    }

    return stats


def get_top_days(df, metric='ShortVolume', n=20):
    """
    Get top N days by specified metric.

    Args:
        df: Preprocessed DataFrame
        metric: Column to sort by
        n: Number of top days to return

    Returns:
        DataFrame with top N days
    """
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    # Build column list, ensuring no duplicates
    base_cols = ['Date', 'Symbol', 'ShortVolume', 'ShortExemptVolume',
                 'TotalVolume', 'ShortPercent', 'ShortExemptRatio']

    # Add metric column if it's not already in base_cols
    if metric not in base_cols:
        cols = ['Date', 'Symbol', metric] + [c for c in base_cols if c not in ['Date', 'Symbol']]
    else:
        cols = base_cols

    top_days = df.nlargest(n, metric)[cols].copy()

    return top_days


def analyze_by_day_of_week(df):
    """
    Analyze short volume patterns by day of week.

    Args:
        df: Preprocessed DataFrame

    Returns:
        DataFrame with day-of-week aggregations
    """
    if df.empty:
        return pd.DataFrame()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    daily_avg = df.groupby('DayOfWeek').agg({
        'ShortVolume': ['mean', 'sum', 'count'],
        'ShortExemptVolume': ['mean', 'sum'],
        'TotalVolume': ['mean', 'sum'],
        'ShortPercent': 'mean',
        'ShortExemptRatio': 'mean'
    }).round(2)

    daily_avg.columns = ['_'.join(col).strip() for col in daily_avg.columns.values]

    # Reindex to get proper day order
    daily_avg = daily_avg.reindex(day_order)

    return daily_avg


def analyze_by_week(df):
    """
    Analyze short volume by week.

    Args:
        df: Preprocessed DataFrame

    Returns:
        DataFrame with weekly aggregations
    """
    if df.empty:
        return pd.DataFrame()

    weekly = df.groupby('YearWeek').agg({
        'Date': ['min', 'max'],
        'ShortVolume': ['sum', 'mean'],
        'ShortExemptVolume': ['sum', 'mean'],
        'TotalVolume': ['sum', 'mean'],
        'ShortPercent': 'mean',
        'ShortExemptRatio': 'mean'
    }).round(2)

    weekly.columns = ['_'.join(col).strip() for col in weekly.columns.values]
    weekly = weekly.sort_index()

    return weekly


def analyze_by_month(df):
    """
    Analyze short volume by month.

    Args:
        df: Preprocessed DataFrame

    Returns:
        DataFrame with monthly aggregations
    """
    if df.empty:
        return pd.DataFrame()

    monthly = df.groupby('YearMonth').agg({
        'Date': 'count',
        'ShortVolume': ['sum', 'mean', 'std'],
        'ShortExemptVolume': ['sum', 'mean'],
        'TotalVolume': ['sum', 'mean'],
        'ShortPercent': ['mean', 'std'],
        'ShortExemptRatio': ['mean', 'max']
    }).round(2)

    monthly.columns = ['_'.join(col).strip() for col in monthly.columns.values]
    monthly = monthly.rename(columns={'Date_count': 'TradingDays'})
    monthly = monthly.sort_index()

    return monthly


def analyze_short_exempt(df):
    """
    Comprehensive analysis of short exempt volume with anomaly detection.

    Args:
        df: Preprocessed DataFrame

    Returns:
        Dictionary with exempt analysis results
    """
    if df.empty:
        return {}

    total_exempt = df['ShortExemptVolume'].sum()
    total_short = df['ShortVolume'].sum()
    days_with_exempt = len(df[df['ShortExemptVolume'] > 0])
    total_days = len(df)

    results = {
        'overview': {
            'total_exempt': int(total_exempt),
            'total_short': int(total_short),
            'overall_ratio': float((total_exempt / total_short * 100) if total_short > 0 else 0),
            'days_with_exempt': days_with_exempt,
            'percent_days_with_exempt': float((days_with_exempt / total_days * 100) if total_days > 0 else 0)
        },
        'anomalies': []
    }

    # Statistical analysis for anomaly detection
    exempt_data = df[df['ShortExemptVolume'] > 0]['ShortExemptVolume']

    if len(exempt_data) > 0:
        mean_exempt = exempt_data.mean()
        std_exempt = exempt_data.std()

        results['statistics'] = {
            'mean': float(mean_exempt),
            'median': float(exempt_data.median()),
            'std': float(std_exempt),
            'min': int(exempt_data.min()),
            'max': int(exempt_data.max())
        }

        # Identify anomalies (> 2 std deviations)
        if std_exempt > 0:
            threshold_2std = mean_exempt + (2 * std_exempt)
            threshold_3std = mean_exempt + (3 * std_exempt)

            anomalies_2std = df[df['ShortExemptVolume'] > threshold_2std]
            anomalies_3std = df[df['ShortExemptVolume'] > threshold_3std]

            results['anomaly_detection'] = {
                'threshold_2std': float(threshold_2std),
                'count_2std': len(anomalies_2std),
                'threshold_3std': float(threshold_3std),
                'count_3std': len(anomalies_3std)
            }

            # Get specific anomalous days
            if len(anomalies_2std) > 0:
                results['anomalies'] = anomalies_2std[[
                    'Date', 'ShortExemptVolume', 'ShortVolume', 'ShortExemptRatio'
                ]].to_dict('records')

    return results


def analyze_discrepancies(df):
    """
    Analyze data discrepancies and unusual patterns.

    Args:
        df: Preprocessed DataFrame

    Returns:
        Dictionary with discrepancy analysis results
    """
    if df.empty:
        return {}

    results = {}

    # 1. Data integrity check
    invalid_short = df[df['ShortVolume'] > df['TotalVolume']]
    results['invalid_short_volume'] = {
        'count': len(invalid_short),
        'dates': invalid_short['Date'].tolist() if len(invalid_short) > 0 else []
    }

    # 2. Short exempt consistency
    invalid_exempt = df[df['ShortExemptVolume'] > df['ShortVolume']]
    results['invalid_exempt_volume'] = {
        'count': len(invalid_exempt),
        'dates': invalid_exempt['Date'].tolist() if len(invalid_exempt) > 0 else []
    }

    # 3. Short percentage outliers
    mean_pct = df['ShortPercent'].mean()
    std_pct = df['ShortPercent'].std()

    high_short_pct = df[df['ShortPercent'] > mean_pct + (2 * std_pct)]
    low_short_pct = df[df['ShortPercent'] < mean_pct - (2 * std_pct)]

    results['short_percent_outliers'] = {
        'mean': float(mean_pct),
        'std': float(std_pct),
        'high_outliers': len(high_short_pct),
        'low_outliers': len(low_short_pct),
        'high_days': high_short_pct[['Date', 'ShortPercent', 'ShortVolume', 'TotalVolume']].head(10).to_dict('records'),
        'low_days': low_short_pct[['Date', 'ShortPercent', 'ShortVolume', 'TotalVolume']].head(10).to_dict('records')
    }

    # 4. Large day-over-day changes
    df_sorted = df.sort_values('Date').copy()
    df_sorted['ShortVolumePctChange'] = df_sorted['ShortVolume'].pct_change() * 100

    large_changes = df_sorted[abs(df_sorted['ShortVolumePctChange']) > 200].dropna()

    results['large_volume_changes'] = {
        'count': len(large_changes),
        'days': large_changes[['Date', 'ShortVolumePctChange', 'ShortVolume']].head(10).to_dict('records')
    }

    return results


def compare_symbols(data_dict):
    """
    Compare metrics across multiple symbols.

    Args:
        data_dict: Dictionary of {symbol: DataFrame}

    Returns:
        DataFrame with comparison metrics
    """
    if not data_dict:
        return pd.DataFrame()

    metrics = []

    for symbol, df in data_dict.items():
        if df.empty:
            continue

        metrics.append({
            'Symbol': symbol,
            'Trading Days': len(df),
            'Avg Short %': round(df['ShortPercent'].mean(), 2),
            'Max Short %': round(df['ShortPercent'].max(), 2),
            'Min Short %': round(df['ShortPercent'].min(), 2),
            'Total Short Vol': int(df['ShortVolume'].sum()),
            'Avg Exempt Ratio': round(df['ShortExemptRatio'].mean(), 4),
            'Days with Exempt': len(df[df['ShortExemptVolume'] > 0]),
            'Latest Short %': round(df.sort_values('Date').iloc[-1]['ShortPercent'], 2) if len(df) > 0 else 0
        })

    return pd.DataFrame(metrics)


def calculate_correlation_matrix(data_dict, metric='ShortPercent'):
    """
    Calculate cross-symbol correlations.

    Args:
        data_dict: Dictionary of {symbol: DataFrame}
        metric: Column to calculate correlations for

    Returns:
        Correlation matrix DataFrame
    """
    if not data_dict or len(data_dict) < 2:
        return pd.DataFrame()

    # Merge all dataframes on Date
    merged = None
    for symbol, df in data_dict.items():
        if df.empty or metric not in df.columns:
            continue

        temp = df[['Date', metric]].copy()
        temp = temp.rename(columns={metric: symbol})

        if merged is None:
            merged = temp
        else:
            merged = merged.merge(temp, on='Date', how='outer')

    if merged is None or len(merged.columns) < 3:  # Date + at least 2 symbols
        return pd.DataFrame()

    # Calculate correlation
    corr_matrix = merged.drop('Date', axis=1).corr()

    return corr_matrix
