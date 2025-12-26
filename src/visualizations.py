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
