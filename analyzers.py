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

        # Calculate percentiles for robust thresholds
        p90 = exempt_data.quantile(0.90)
        p95 = exempt_data.quantile(0.95)
        p99 = exempt_data.quantile(0.99)

        results['statistics'] = {
            'mean': float(mean_exempt),
            'median': float(exempt_data.median()),
            'std': float(std_exempt),
            'min': int(exempt_data.min()),
            'max': int(exempt_data.max()),
            'p90': float(p90),
            'p95': float(p95),
            'p99': float(p99)
        }

        # Identify anomalies using both z-score and percentile methods
        if std_exempt > 0:
            threshold_2std = mean_exempt + (2 * std_exempt)
            threshold_3std = mean_exempt + (3 * std_exempt)

            anomalies_2std = df[df['ShortExemptVolume'] > threshold_2std]
            anomalies_3std = df[df['ShortExemptVolume'] > threshold_3std]
            anomalies_p95 = df[df['ShortExemptVolume'] > p95]
            anomalies_p99 = df[df['ShortExemptVolume'] > p99]

            results['anomaly_detection'] = {
                'threshold_2std': float(threshold_2std),
                'count_2std': len(anomalies_2std),
                'threshold_3std': float(threshold_3std),
                'count_3std': len(anomalies_3std),
                'threshold_p95': float(p95),
                'count_p95': len(anomalies_p95),
                'threshold_p99': float(p99),
                'count_p99': len(anomalies_p99)
            }

            # Get specific anomalous days (using 2std or p95, whichever is more conservative)
            conservative_threshold = max(threshold_2std, p95)
            anomalies = df[df['ShortExemptVolume'] > conservative_threshold]

            if len(anomalies) > 0:
                results['anomalies'] = anomalies[[
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


def analyze_exempt_patterns(df):
    """
    Detect recurring patterns in short exempt volume.

    Args:
        df: Preprocessed DataFrame (single symbol)

    Returns:
        Dictionary with pattern analysis results
    """
    if df.empty:
        return {}

    # Get days with exempt volume
    exempt_days = df[df['ShortExemptVolume'] > 0].copy()

    if len(exempt_days) == 0:
        return {
            'day_of_week_patterns': {},
            'clustering': [],
            'seasonality': {},
            'velocity': {}
        }

    # Identify anomalies for pattern analysis
    exempt_data = exempt_days['ShortExemptVolume']
    mean_exempt = exempt_data.mean()
    std_exempt = exempt_data.std()

    if std_exempt > 0:
        threshold = mean_exempt + (2 * std_exempt)
        anomaly_days = exempt_days[exempt_days['ShortExemptVolume'] > threshold].copy()
    else:
        anomaly_days = pd.DataFrame()

    # 1. Day-of-week patterns
    day_counts = {}
    if len(anomaly_days) > 0:
        day_counts = anomaly_days['DayOfWeek'].value_counts().to_dict()

        # Calculate anomaly rate per day
        total_days_per_dow = df['DayOfWeek'].value_counts()
        anomaly_rate_per_dow = {}
        for day, count in day_counts.items():
            if day in total_days_per_dow:
                anomaly_rate_per_dow[day] = (count / total_days_per_dow[day] * 100)

        day_of_week_patterns = {
            'anomaly_counts': day_counts,
            'anomaly_rates': anomaly_rate_per_dow,
            'highest_day': max(day_counts, key=day_counts.get) if day_counts else None
        }
    else:
        day_of_week_patterns = {}

    # 2. Clustering (consecutive anomaly periods)
    clusters = []
    if len(anomaly_days) > 0:
        anomaly_days_sorted = anomaly_days.sort_values('Date')
        dates = anomaly_days_sorted['Date'].tolist()

        current_cluster = [dates[0]]
        for i in range(1, len(dates)):
            # Check if consecutive trading days (within 4 calendar days accounting for weekends)
            if (dates[i] - current_cluster[-1]).days <= 4:
                current_cluster.append(dates[i])
            else:
                if len(current_cluster) >= 2:  # Only count clusters of 2+ days
                    clusters.append({
                        'start': current_cluster[0],
                        'end': current_cluster[-1],
                        'length': len(current_cluster)
                    })
                current_cluster = [dates[i]]

        # Don't forget last cluster
        if len(current_cluster) >= 2:
            clusters.append({
                'start': current_cluster[0],
                'end': current_cluster[-1],
                'length': len(current_cluster)
            })

    # 3. Seasonality (monthly trends)
    monthly_exempt = exempt_days.groupby('YearMonth').agg({
        'ShortExemptVolume': ['mean', 'sum', 'count'],
        'ShortExemptRatio': 'mean'
    })

    seasonality = {
        'monthly_avg_exempt': monthly_exempt[('ShortExemptVolume', 'mean')].to_dict() if len(monthly_exempt) > 0 else {},
        'monthly_total_exempt': monthly_exempt[('ShortExemptVolume', 'sum')].to_dict() if len(monthly_exempt) > 0 else {}
    }

    # 4. Velocity (day-over-day changes)
    exempt_days_sorted = exempt_days.sort_values('Date')
    if len(exempt_days_sorted) > 1:
        exempt_days_sorted['ExemptPctChange'] = exempt_days_sorted['ShortExemptVolume'].pct_change() * 100

        large_increases = exempt_days_sorted[exempt_days_sorted['ExemptPctChange'] > 200]
        large_decreases = exempt_days_sorted[exempt_days_sorted['ExemptPctChange'] < -80]

        velocity = {
            'avg_change': float(exempt_days_sorted['ExemptPctChange'].mean()),
            'max_increase': float(exempt_days_sorted['ExemptPctChange'].max()) if not exempt_days_sorted['ExemptPctChange'].isna().all() else 0,
            'max_decrease': float(exempt_days_sorted['ExemptPctChange'].min()) if not exempt_days_sorted['ExemptPctChange'].isna().all() else 0,
            'large_spike_count': len(large_increases)
        }
    else:
        velocity = {}

    return {
        'day_of_week_patterns': day_of_week_patterns,
        'clustering': clusters,
        'seasonality': seasonality,
        'velocity': velocity
    }


def calculate_rolling_exempt_stats(df, windows=[7, 30, 60]):
    """
    Calculate rolling statistics for exempt volume and ratio.

    Args:
        df: Preprocessed DataFrame (single symbol)
        windows: List of window sizes in days

    Returns:
        DataFrame with added rolling statistic columns and z-scores
    """
    if df.empty:
        return df

    df = df.sort_values('Date').copy()

    # Calculate rolling stats for each window
    for window in windows:
        # Rolling mean and std for exempt volume
        df[f'ExemptVol_{window}d_avg'] = df['ShortExemptVolume'].rolling(window=window, min_periods=1).mean()
        df[f'ExemptVol_{window}d_std'] = df['ShortExemptVolume'].rolling(window=window, min_periods=1).std()

        # Rolling mean and std for exempt ratio
        df[f'ExemptRatio_{window}d_avg'] = df['ShortExemptRatio'].rolling(window=window, min_periods=1).mean()
        df[f'ExemptRatio_{window}d_std'] = df['ShortExemptRatio'].rolling(window=window, min_periods=1).std()

        # Calculate z-score (how many std deviations from rolling mean)
        df[f'ExemptVol_{window}d_zscore'] = np.where(
            df[f'ExemptVol_{window}d_std'] > 0,
            (df['ShortExemptVolume'] - df[f'ExemptVol_{window}d_avg']) / df[f'ExemptVol_{window}d_std'],
            0
        )

        df[f'ExemptRatio_{window}d_zscore'] = np.where(
            df[f'ExemptRatio_{window}d_std'] > 0,
            (df['ShortExemptRatio'] - df[f'ExemptRatio_{window}d_avg']) / df[f'ExemptRatio_{window}d_std'],
            0
        )

    # Flag anomalies based on 30-day window z-score
    df['IsExemptAnomaly'] = df['ExemptVol_30d_zscore'] > 2

    return df


def build_exempt_dashboard_data(data_dict):
    """
    Create summary dashboard data aggregating all symbols.

    Args:
        data_dict: Dictionary of {symbol: DataFrame}

    Returns:
        Dictionary with dashboard components
    """
    if not data_dict:
        return {}

    all_alerts = []
    total_anomalies = 0
    symbols_with_exempt = []

    # Collect alerts from all symbols
    for symbol, df in data_dict.items():
        # Calculate rolling stats to get z-scores
        df_with_stats = calculate_rolling_exempt_stats(df)

        # Get anomalies (z-score > 2)
        anomalies = df_with_stats[df_with_stats['IsExemptAnomaly']].copy()

        if len(anomalies) > 0:
            total_anomalies += len(anomalies)

            # Create alert records
            for _, row in anomalies.iterrows():
                all_alerts.append({
                    'symbol': symbol,
                    'date': row['Date'],
                    'exempt_vol': int(row['ShortExemptVolume']),
                    'exempt_ratio': float(row['ShortExemptRatio']),
                    'z_score': float(row['ExemptVol_30d_zscore']),
                    'short_vol': int(row['ShortVolume']),
                    'total_vol': int(row['TotalVolume'])
                })

        # Track symbols with exempt activity
        if df['ShortExemptVolume'].sum() > 0:
            symbols_with_exempt.append(symbol)

    # Sort alerts by z-score (most significant first)
    all_alerts = sorted(all_alerts, key=lambda x: x['z_score'], reverse=True)

    # Top 10 alerts
    top_alerts = all_alerts[:10]

    # Summary metrics
    summary_metrics = {
        'total_anomalies': total_anomalies,
        'symbols_analyzed': len(data_dict),
        'symbols_with_exempt': len(symbols_with_exempt),
        'highest_z_score': all_alerts[0]['z_score'] if all_alerts else 0,
        'highest_symbol': all_alerts[0]['symbol'] if all_alerts else None
    }

    # Pattern summary across all symbols
    pattern_insights = []

    for symbol, df in data_dict.items():
        patterns = analyze_exempt_patterns(df)

        # Day-of-week insights
        dow_patterns = patterns.get('day_of_week_patterns', {})
        if dow_patterns and 'highest_day' in dow_patterns:
            highest_day = dow_patterns['highest_day']
            if highest_day and highest_day in dow_patterns.get('anomaly_rates', {}):
                rate = dow_patterns['anomaly_rates'][highest_day]
                if rate > 20:  # More than 20% of that day type
                    pattern_insights.append(f"{symbol}: {highest_day}s show elevated exempt activity ({rate:.0f}% anomaly rate)")

        # Clustering insights
        clusters = patterns.get('clustering', [])
        if len(clusters) >= 2:
            pattern_insights.append(f"{symbol}: {len(clusters)} clusters of consecutive anomalies detected")

        # Velocity insights
        velocity = patterns.get('velocity', {})
        if velocity and velocity.get('large_spike_count', 0) > 3:
            pattern_insights.append(f"{symbol}: {velocity['large_spike_count']} large exempt volume spikes (>200% increase)")

    return {
        'alerts': top_alerts,
        'all_alerts': all_alerts,  # For filtering
        'summary_metrics': summary_metrics,
        'pattern_insights': pattern_insights
    }


def compare_exempt_patterns(data_dict):
    """
    Find patterns and outliers across multiple symbols.

    Args:
        data_dict: Dictionary of {symbol: DataFrame}

    Returns:
        Dictionary with cross-symbol comparison results
    """
    if not data_dict or len(data_dict) < 2:
        return {}

    # Calculate exempt ratio correlation matrix
    corr_matrix = calculate_correlation_matrix(data_dict, 'ShortExemptRatio')

    # Identify outlier symbols (unusual exempt activity)
    outlier_symbols = []
    symbol_stats = {}

    for symbol, df in data_dict.items():
        exempt_analysis = analyze_short_exempt(df)
        patterns = analyze_exempt_patterns(df)

        stats = exempt_analysis.get('statistics', {})
        anomaly_count = exempt_analysis.get('anomaly_detection', {}).get('count_2std', 0)

        symbol_stats[symbol] = {
            'mean_exempt': stats.get('mean', 0),
            'anomaly_count': anomaly_count,
            'cluster_count': len(patterns.get('clustering', [])),
            'overall_ratio': exempt_analysis.get('overview', {}).get('overall_ratio', 0)
        }

    # Find outliers (symbols with significantly more anomalies)
    if symbol_stats:
        anomaly_counts = [s['anomaly_count'] for s in symbol_stats.values()]
        if anomaly_counts:
            mean_anomalies = sum(anomaly_counts) / len(anomaly_counts)
            std_anomalies = (sum((x - mean_anomalies) ** 2 for x in anomaly_counts) / len(anomaly_counts)) ** 0.5

            for symbol, stats in symbol_stats.items():
                if std_anomalies > 0 and stats['anomaly_count'] > mean_anomalies + (2 * std_anomalies):
                    outlier_symbols.append({
                        'symbol': symbol,
                        'anomaly_count': stats['anomaly_count'],
                        'deviation': (stats['anomaly_count'] - mean_anomalies) / std_anomalies if std_anomalies > 0 else 0
                    })

    # Find common anomaly dates (dates where multiple symbols show anomalies)
    all_anomaly_dates = {}
    for symbol, df in data_dict.items():
        df_with_stats = calculate_rolling_exempt_stats(df)
        anomaly_dates = df_with_stats[df_with_stats['IsExemptAnomaly']]['Date'].tolist()

        for date in anomaly_dates:
            if date not in all_anomaly_dates:
                all_anomaly_dates[date] = []
            all_anomaly_dates[date].append(symbol)

    # Filter to dates with multiple symbols
    common_anomaly_dates = [
        {'date': date, 'symbols': symbols, 'count': len(symbols)}
        for date, symbols in all_anomaly_dates.items()
        if len(symbols) >= 2
    ]
    common_anomaly_dates = sorted(common_anomaly_dates, key=lambda x: x['count'], reverse=True)

    # Calculate divergence score (how different are the symbols' behaviors)
    if not corr_matrix.empty and len(corr_matrix) > 1:
        # Average correlation (excluding diagonal)
        corr_values = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                corr_values.append(corr_matrix.iloc[i, j])

        avg_correlation = sum(corr_values) / len(corr_values) if corr_values else 0
        # Divergence: low correlation = high divergence
        divergence_score = 1 - abs(avg_correlation)
    else:
        divergence_score = 0

    return {
        'correlation_matrix': corr_matrix,
        'outlier_symbols': outlier_symbols,
        'common_anomaly_dates': common_anomaly_dates[:10],  # Top 10
        'divergence_score': float(divergence_score),
        'symbol_stats': symbol_stats
    }
