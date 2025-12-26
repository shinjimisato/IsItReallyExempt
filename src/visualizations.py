"""
Visualization functions using Plotly for interactive charts.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def plot_short_percentage_timeline(df, symbol):
    """
    Interactive time series chart for short percentage over time.

    Args:
        df: Preprocessed DataFrame
        symbol: Stock symbol for title

    Returns:
        Plotly Figure object
    """
    if df.empty:
        return go.Figure()

    df = df.sort_values('Date')

    fig = go.Figure()

    # Add short percentage line with fill
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['ShortPercent'],
        mode='lines',
        name='Short %',
        line=dict(color='#FF4B4B', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 75, 75, 0.1)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Short %: %{y:.2f}%<extra></extra>'
    ))

    # Add mean line
    mean_val = df['ShortPercent'].mean()
    fig.add_hline(
        y=mean_val,
        line_dash="dash",
        line_color="#0068C9",
        annotation_text=f"Mean: {mean_val:.1f}%",
        annotation_position="right"
    )

    # Layout with range selector
    fig.update_layout(
        title=f"{symbol} - Short Volume as % of Total Volume",
        xaxis_title="Date",
        yaxis_title="Short %",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor='#F0F2F6',
                activecolor='#FF4B4B'
            ),
            rangeslider=dict(visible=True, thickness=0.05),
            type="date"
        )
    )

    return fig


def plot_short_volume_timeline(df, symbol):
    """
    Interactive bar chart for short volume over time.

    Args:
        df: Preprocessed DataFrame
        symbol: Stock symbol for title

    Returns:
        Plotly Figure object
    """
    if df.empty:
        return go.Figure()

    df = df.sort_values('Date')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['ShortVolume'],
        name='Short Volume',
        marker_color='#FF4B4B',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Short Vol: %{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"{symbol} - Daily Short Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        template='plotly_white',
        height=500,
        hovermode='x'
    )

    return fig


def plot_exempt_volume_timeline(df, symbol):
    """
    Interactive chart for short exempt volume with anomaly highlights.

    Args:
        df: Preprocessed DataFrame
        symbol: Stock symbol for title

    Returns:
        Plotly Figure object
    """
    if df.empty:
        return go.Figure()

    df = df.sort_values('Date')

    # Calculate anomaly threshold
    exempt_data = df[df['ShortExemptVolume'] > 0]['ShortExemptVolume']
    if len(exempt_data) > 0:
        mean_exempt = exempt_data.mean()
        std_exempt = exempt_data.std()
        threshold = mean_exempt + (2 * std_exempt) if std_exempt > 0 else mean_exempt
    else:
        threshold = 0

    # Identify anomalies
    df['IsAnomaly'] = df['ShortExemptVolume'] > threshold

    fig = go.Figure()

    # Normal exempt volume
    normal_df = df[~df['IsAnomaly']]
    fig.add_trace(go.Bar(
        x=normal_df['Date'],
        y=normal_df['ShortExemptVolume'],
        name='Exempt Volume',
        marker_color='#FFA500',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Exempt Vol: %{y:,.0f}<extra></extra>'
    ))

    # Anomalous exempt volume
    anomaly_df = df[df['IsAnomaly']]
    if len(anomaly_df) > 0:
        fig.add_trace(go.Bar(
            x=anomaly_df['Date'],
            y=anomaly_df['ShortExemptVolume'],
            name='Anomaly (>2σ)',
            marker_color='#DC143C',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Exempt Vol: %{y:,.0f}<br><b>ANOMALY</b><extra></extra>'
        ))

    fig.update_layout(
        title=f"{symbol} - Short Exempt Volume (Anomalies Highlighted)",
        xaxis_title="Date",
        yaxis_title="Exempt Volume",
        template='plotly_white',
        height=500,
        hovermode='x',
        barmode='overlay'
    )

    return fig


def plot_short_pct_distribution(df, symbol):
    """
    Histogram showing distribution of short percentages.

    Args:
        df: Preprocessed DataFrame
        symbol: Stock symbol for title

    Returns:
        Plotly Figure object
    """
    if df.empty:
        return go.Figure()

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=df['ShortPercent'],
        nbinsx=30,
        name='Frequency',
        marker_color='#FF4B4B',
        opacity=0.7,
        hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
    ))

    # Add mean and median lines
    mean_val = df['ShortPercent'].mean()
    median_val = df['ShortPercent'].median()

    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="#0068C9",
        annotation_text=f"Mean: {mean_val:.1f}%"
    )

    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color="#29B09D",
        annotation_text=f"Median: {median_val:.1f}%"
    )

    fig.update_layout(
        title=f"{symbol} - Distribution of Short %",
        xaxis_title="Short %",
        yaxis_title="Frequency",
        template='plotly_white',
        height=450,
        showlegend=False
    )

    return fig


def plot_day_of_week_analysis(df, symbol):
    """
    Grouped bar chart showing average volumes by day of week.

    Args:
        df: Preprocessed DataFrame
        symbol: Stock symbol for title

    Returns:
        Plotly Figure object
    """
    if df.empty:
        return go.Figure()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    dow_avg = df.groupby('DayOfWeek').agg({
        'ShortVolume': 'mean',
        'TotalVolume': 'mean'
    }).reindex(day_order)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=dow_avg.index,
        y=dow_avg['ShortVolume'],
        name='Avg Short Volume',
        marker_color='#FF4B4B',
        hovertemplate='%{x}<br>Avg Short Vol: %{y:,.0f}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=dow_avg.index,
        y=dow_avg['TotalVolume'],
        name='Avg Total Volume',
        marker_color='#0068C9',
        hovertemplate='%{x}<br>Avg Total Vol: %{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"{symbol} - Average Volume by Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Average Volume",
        template='plotly_white',
        height=450,
        barmode='group'
    )

    return fig


def plot_multi_symbol_comparison(data_dict, metric='ShortPercent'):
    """
    Overlay multiple symbols on one chart for comparison.

    Args:
        data_dict: Dictionary of {symbol: DataFrame}
        metric: Column to plot ('ShortPercent', 'ShortVolume', etc.)

    Returns:
        Plotly Figure object
    """
    if not data_dict:
        return go.Figure()

    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for i, (symbol, df) in enumerate(data_dict.items()):
        if df.empty or metric not in df.columns:
            continue

        df_sorted = df.sort_values('Date')

        fig.add_trace(go.Scatter(
            x=df_sorted['Date'],
            y=df_sorted[metric],
            mode='lines',
            name=symbol,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f'<b>{symbol}</b><br>%{{x|%Y-%m-%d}}<br>{metric}: %{{y:.2f}}<extra></extra>'
        ))

    fig.update_layout(
        title=f"{metric} Comparison Across Symbols",
        xaxis_title="Date",
        yaxis_title=metric,
        template='plotly_white',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def plot_correlation_heatmap(corr_matrix):
    """
    Heatmap showing correlations between symbols.

    Args:
        corr_matrix: Correlation matrix DataFrame

    Returns:
        Plotly Figure object
    """
    if corr_matrix.empty:
        return go.Figure()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.3f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title="Short % Correlation Matrix",
        template='plotly_white',
        height=400,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )

    return fig


def plot_volume_breakdown(df, symbol):
    """
    Stacked area chart showing short vs non-short volume.

    Args:
        df: Preprocessed DataFrame
        symbol: Stock symbol for title

    Returns:
        Plotly Figure object
    """
    if df.empty:
        return go.Figure()

    df = df.sort_values('Date')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['NonShortVolume'],
        name='Non-Short Volume',
        mode='lines',
        stackgroup='one',
        fillcolor='rgba(0, 104, 201, 0.5)',
        line=dict(width=0),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Non-Short: %{y:,.0f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['ShortVolume'],
        name='Short Volume',
        mode='lines',
        stackgroup='one',
        fillcolor='rgba(255, 75, 75, 0.5)',
        line=dict(width=0),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Short: %{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"{symbol} - Volume Breakdown (Short vs Non-Short)",
        xaxis_title="Date",
        yaxis_title="Volume",
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )

    return fig


def plot_monthly_trends(df, symbol):
    """
    Bar chart showing monthly aggregated metrics.

    Args:
        df: Preprocessed DataFrame
        symbol: Stock symbol for title

    Returns:
        Plotly Figure object
    """
    if df.empty:
        return go.Figure()

    monthly = df.groupby('YearMonth').agg({
        'ShortVolume': 'sum',
        'ShortExemptVolume': 'sum',
        'TotalVolume': 'sum'
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=monthly.index,
        y=monthly['ShortVolume'],
        name='Short Volume',
        marker_color='#FF4B4B',
        hovertemplate='%{x}<br>Short Vol: %{y:,.0f}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=monthly.index,
        y=monthly['ShortExemptVolume'],
        name='Exempt Volume',
        marker_color='#FFA500',
        hovertemplate='%{x}<br>Exempt Vol: %{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"{symbol} - Monthly Volume Trends",
        xaxis_title="Month",
        yaxis_title="Volume",
        template='plotly_white',
        height=450,
        barmode='group'
    )

    return fig


def plot_exempt_heatmap(df, symbol):
    """
    Calendar heatmap showing exempt volume intensity by day.

    Args:
        df: Preprocessed DataFrame
        symbol: Stock symbol for title

    Returns:
        Plotly Figure object
    """
    if df.empty:
        return go.Figure()

    df = df.sort_values('Date').copy()

    # Add week of year and day of week for heatmap
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeekNum'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday

    # Create pivot table for heatmap
    pivot_data = df.pivot_table(
        values='ShortExemptVolume',
        index='DayOfWeekNum',
        columns='WeekOfYear',
        aggfunc='sum',
        fill_value=0
    )

    # Use log scale for better visualization (add 1 to avoid log(0))
    pivot_data_log = np.log10(pivot_data + 1)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data_log.values,
        x=pivot_data_log.columns,
        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        colorscale='YlOrRd',
        hovertemplate='Week %{x}<br>%{y}<br>Exempt Vol: %{customdata:,.0f}<extra></extra>',
        customdata=pivot_data.values,
        colorbar=dict(title="Exempt Volume<br>(log scale)")
    ))

    fig.update_layout(
        title=f"{symbol} - Short Exempt Volume Heatmap",
        xaxis_title="Week of Year",
        yaxis_title="Day of Week",
        template='plotly_white',
        height=400,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )

    return fig


def plot_rolling_exempt_stats(df, symbol, window=30):
    """
    Time series with rolling average and confidence bands.

    Args:
        df: Preprocessed DataFrame
        symbol: Stock symbol for title
        window: Rolling window size in days

    Returns:
        Plotly Figure object
    """
    if df.empty:
        return go.Figure()

    # Import calculate_rolling_exempt_stats from analyzers
    from .analyzers import calculate_rolling_exempt_stats

    df = calculate_rolling_exempt_stats(df, windows=[window])
    df = df.sort_values('Date')

    fig = go.Figure()

    # Confidence bands (±2σ)
    upper_band = df[f'ExemptRatio_{window}d_avg'] + (2 * df[f'ExemptRatio_{window}d_std'])
    lower_band = df[f'ExemptRatio_{window}d_avg'] - (2 * df[f'ExemptRatio_{window}d_std'])
    lower_band = lower_band.clip(lower=0)  # Can't be negative

    # Add confidence band
    fig.add_trace(go.Scatter(
        x=df['Date'].tolist() + df['Date'].tolist()[::-1],
        y=upper_band.tolist() + lower_band.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0, 100, 200, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='±2σ Band',
        hoverinfo='skip'
    ))

    # Add rolling mean
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df[f'ExemptRatio_{window}d_avg'],
        mode='lines',
        name=f'{window}d Avg',
        line=dict(color='#0068C9', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Avg: %{y:.4f}%<extra></extra>'
    ))

    # Add actual values
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['ShortExemptRatio'],
        mode='lines',
        name='Actual',
        line=dict(color='#FF4B4B', width=1),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Exempt Ratio: %{y:.4f}%<extra></extra>'
    ))

    # Highlight anomalies
    anomalies = df[df['IsExemptAnomaly']]
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies['Date'],
            y=anomalies['ShortExemptRatio'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=8, symbol='circle'),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Anomaly: %{y:.4f}%<extra></extra>'
        ))

    fig.update_layout(
        title=f"{symbol} - Rolling {window}-Day Exempt Ratio with Confidence Bands",
        xaxis_title="Date",
        yaxis_title="Exempt Ratio %",
        template='plotly_white',
        height=500,
        hovermode='x unified',
        showlegend=True
    )

    return fig


def plot_exempt_distribution(df, symbol):
    """
    Histogram of exempt ratio distribution with anomaly overlay.

    Args:
        df: Preprocessed DataFrame
        symbol: Stock symbol for title

    Returns:
        Plotly Figure object
    """
    if df.empty:
        return go.Figure()

    # Filter to days with exempt volume > 0
    exempt_days = df[df['ShortExemptVolume'] > 0]

    if len(exempt_days) == 0:
        return go.Figure()

    # Import calculate_rolling_exempt_stats
    from .analyzers import calculate_rolling_exempt_stats

    exempt_days = calculate_rolling_exempt_stats(exempt_days)

    # Separate normal and anomalous days
    normal_days = exempt_days[~exempt_days['IsExemptAnomaly']]
    anomaly_days = exempt_days[exempt_days['IsExemptAnomaly']]

    fig = go.Figure()

    # Histogram of normal days
    fig.add_trace(go.Histogram(
        x=normal_days['ShortExemptRatio'],
        nbinsx=30,
        name='Normal Days',
        marker_color='#0068C9',
        opacity=0.7,
        hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
    ))

    # Histogram of anomalous days
    if len(anomaly_days) > 0:
        fig.add_trace(go.Histogram(
            x=anomaly_days['ShortExemptRatio'],
            nbinsx=30,
            name='Anomalies',
            marker_color='#FF4B4B',
            opacity=0.7,
            hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
        ))

    # Add statistical markers
    mean_val = exempt_days['ShortExemptRatio'].mean()
    median_val = exempt_days['ShortExemptRatio'].median()

    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Mean: {mean_val:.4f}%"
    )

    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Median: {median_val:.4f}%"
    )

    fig.update_layout(
        title=f"{symbol} - Exempt Ratio Distribution",
        xaxis_title="Exempt Ratio %",
        yaxis_title="Frequency",
        template='plotly_white',
        height=450,
        barmode='overlay',
        showlegend=True
    )

    return fig


def plot_annotated_exempt_timeline(df, symbol, anomalies_only=False):
    """
    Time series with automatic annotations for anomalies.

    Args:
        df: Preprocessed DataFrame
        symbol: Stock symbol for title
        anomalies_only: If True, only show periods around anomalies

    Returns:
        Plotly Figure object
    """
    if df.empty:
        return go.Figure()

    # Import calculate_rolling_exempt_stats
    from .analyzers import calculate_rolling_exempt_stats

    df = calculate_rolling_exempt_stats(df)
    df = df.sort_values('Date')

    fig = go.Figure()

    # Plot exempt volume
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['ShortExemptVolume'],
        mode='lines',
        name='Exempt Volume',
        line=dict(color='#FFA500', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Exempt: %{y:,.0f}<extra></extra>'
    ))

    # Add anomaly markers and annotations
    anomalies = df[df['IsExemptAnomaly']]

    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies['Date'],
            y=anomalies['ShortExemptVolume'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10, symbol='star'),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Exempt: %{customdata[0]:,.0f}<br>Z-score: %{customdata[1]:.2f}σ<extra></extra>',
            customdata=anomalies[['ShortExemptVolume', 'ExemptVol_30d_zscore']].values
        ))

        # Add annotations for top 5 anomalies
        top_anomalies = anomalies.nlargest(min(5, len(anomalies)), 'ExemptVol_30d_zscore')

        for _, row in top_anomalies.iterrows():
            fig.add_annotation(
                x=row['Date'],
                y=row['ShortExemptVolume'],
                text=f"{row['Date'].strftime('%m/%d')}<br>{row['ExemptVol_30d_zscore']:.1f}σ",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                ax=0,
                ay=-40,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="red"
            )

    fig.update_layout(
        title=f"{symbol} - Exempt Volume Timeline with Anomaly Annotations",
        xaxis_title="Date",
        yaxis_title="Exempt Volume",
        template='plotly_white',
        height=500,
        hovermode='x unified',
        showlegend=True
    )

    return fig


def plot_cross_symbol_exempt_comparison(data_dict):
    """
    Multi-symbol comparison for exempt analysis.

    Args:
        data_dict: Dictionary of {symbol: DataFrame}

    Returns:
        Plotly Figure object with subplots
    """
    if not data_dict or len(data_dict) < 2:
        return go.Figure()

    from plotly.subplots import make_subplots

    # Create subplots: 2 rows, 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Avg Exempt Ratio by Symbol', 'Exempt Volume Distribution',
                        'Anomaly Count by Symbol', 'Exempt Ratio Correlation'),
        specs=[[{'type': 'bar'}, {'type': 'box'}],
               [{'type': 'bar'}, {'type': 'heatmap'}]]
    )

    symbols = list(data_dict.keys())
    colors = px.colors.qualitative.Set1

    # 1. Average exempt ratio bar chart
    avg_ratios = []
    for symbol, df in data_dict.items():
        exempt_days = df[df['ShortExemptVolume'] > 0]
        avg_ratio = exempt_days['ShortExemptRatio'].mean() if len(exempt_days) > 0 else 0
        avg_ratios.append(avg_ratio)

    fig.add_trace(
        go.Bar(x=symbols, y=avg_ratios, marker_color=colors[:len(symbols)],
               name='Avg Exempt Ratio',
               hovertemplate='%{x}: %{y:.4f}%<extra></extra>'),
        row=1, col=1
    )

    # 2. Box plot for distribution comparison
    for i, (symbol, df) in enumerate(data_dict.items()):
        exempt_days = df[df['ShortExemptVolume'] > 0]
        fig.add_trace(
            go.Box(y=exempt_days['ShortExemptRatio'], name=symbol,
                   marker_color=colors[i % len(colors)],
                   hovertemplate='%{y:.4f}%<extra></extra>'),
            row=1, col=2
        )

    # 3. Anomaly count bar chart
    from .analyzers import calculate_rolling_exempt_stats

    anomaly_counts = []
    for symbol, df in data_dict.items():
        df_with_stats = calculate_rolling_exempt_stats(df)
        anomaly_count = len(df_with_stats[df_with_stats['IsExemptAnomaly']])
        anomaly_counts.append(anomaly_count)

    fig.add_trace(
        go.Bar(x=symbols, y=anomaly_counts, marker_color=colors[:len(symbols)],
               name='Anomaly Count',
               hovertemplate='%{x}: %{y} anomalies<extra></extra>'),
        row=2, col=1
    )

    # 4. Correlation heatmap
    from .analyzers import calculate_correlation_matrix

    corr_matrix = calculate_correlation_matrix(data_dict, 'ShortExemptRatio')

    if not corr_matrix.empty:
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>',
                showscale=True
            ),
            row=2, col=2
        )

    fig.update_layout(
        title="Cross-Symbol Exempt Analysis Comparison",
        template='plotly_white',
        height=800,
        showlegend=False
    )

    return fig
