"""
FINRA Short Sale Analysis - Streamlit Web Application

Analyzes FINRA daily short sale data for any stock symbol with interactive
visualizations, multi-symbol comparison, and export capabilities.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO

from src.data_fetcher import (
    fetch_finra_data_cached,
    clear_cache_for_symbol,
    get_cache_info
)
from src.data_processor import preprocess_data, validate_data
from src.analyzers import (
    generate_summary_stats,
    get_top_days,
    analyze_by_day_of_week,
    analyze_by_month,
    analyze_short_exempt,
    analyze_discrepancies,
    compare_symbols,
    calculate_correlation_matrix
)
from src.visualizations import (
    plot_short_percentage_timeline,
    plot_short_volume_timeline,
    plot_exempt_volume_timeline,
    plot_short_pct_distribution,
    plot_day_of_week_analysis,
    plot_multi_symbol_comparison,
    plot_correlation_heatmap,
    plot_volume_breakdown,
    plot_monthly_trends
)
from src.utils import (
    get_date_range_from_months,
    format_number,
    format_percentage,
    format_date,
    validate_symbol
)


# Page configuration
st.set_page_config(
    page_title="FINRA Short Sale Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("📊 FINRA Short Sale Analysis Dashboard")
st.markdown("Analyze FINRA daily short sale data for any stock symbol with interactive charts and insights")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Symbol input
st.sidebar.markdown("### Stock Symbols")
symbols_input = st.sidebar.text_area(
    "Enter symbols (one per line)",
    value="MSOS\nMSOX",
    height=100,
    help="Enter up to 5 stock symbols to analyze. One symbol per line."
)

# Parse and validate symbols
raw_symbols = [s.strip() for s in symbols_input.split('\n') if s.strip()]
symbols = [validate_symbol(s) for s in raw_symbols]
symbols = [s for s in symbols if s is not None]

if len(symbols) > 5:
    st.sidebar.warning("⚠️ Maximum 5 symbols allowed for performance. Using first 5.")
    symbols = symbols[:5]

if len(symbols) != len(raw_symbols):
    invalid = set(raw_symbols) - set(symbols)
    if invalid:
        st.sidebar.error(f"Invalid symbols removed: {', '.join(invalid)}")

# Date range
st.sidebar.markdown("### Analysis Period")
months_back = st.sidebar.slider(
    "Months to analyze",
    min_value=1,
    max_value=12,
    value=3,
    help="Number of months of data to fetch and analyze"
)

start_date, end_date = get_date_range_from_months(months_back)
st.sidebar.info(f"📅 {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Cache management
st.sidebar.markdown("### Cache Management")
cache_info = get_cache_info()
st.sidebar.text(f"Cached files: {cache_info['total_files']}")
st.sidebar.text(f"Cache size: {cache_info['total_size_mb']} MB")

if st.sidebar.button("🗑️ Clear All Cache"):
    st.cache_data.clear()
    st.session_state.data_dict = None
    st.sidebar.success("Cache cleared!")
    st.rerun()

# Initialize session state
if 'data_dict' not in st.session_state:
    st.session_state.data_dict = None

# Analyze button
analyze_clicked = st.sidebar.button("🚀 Run Analysis", type="primary", width="stretch")

# Run analysis if button clicked
if analyze_clicked:
    if not symbols:
        st.error("❌ Please enter at least one valid symbol")
        st.stop()

    # Fetch data for all symbols
    data_dict = {}
    failed_symbols = []

    with st.spinner("Fetching FINRA data..."):
        for symbol in symbols:
            try:
                raw = fetch_finra_data_cached(
                    symbol,
                    start_date.strftime('%Y%m%d'),
                    end_date.strftime('%Y%m%d')
                )

                if not raw.empty:
                    processed = preprocess_data(raw)
                    validation = validate_data(processed)

                    if not validation['valid']:
                        st.warning(f"⚠️ Data quality issues for {symbol}: {', '.join(validation['errors'])}")

                    data_dict[symbol] = processed
                else:
                    failed_symbols.append(symbol)
                    st.warning(f"⚠️ No data found for {symbol} in the selected period")

            except Exception as e:
                failed_symbols.append(symbol)
                st.error(f"❌ Error fetching {symbol}: {str(e)}")

    if not data_dict:
        st.error("❌ No data available for analysis. Please try different symbols or date range.")
        st.stop()

    # Store data in session state
    st.session_state.data_dict = data_dict

    # Success message
    success_symbols = list(data_dict.keys())
    st.success(f"✅ Successfully loaded data for: {', '.join(success_symbols)}")

# Use data from session state if available
data_dict = st.session_state.data_dict

# Main content area
if data_dict is not None:

    # Create tabs
    tabs = st.tabs([
        "📈 Overview",
        "📅 Time Analysis",
        "🔍 Anomalies",
        "📊 Charts",
        "⚖️ Comparison" if len(data_dict) > 1 else "📊 Charts",
        "💾 Export"
    ])

    # ===== TAB 1: OVERVIEW =====
    with tabs[0]:
        st.subheader("Summary Statistics")

        # Key metrics row
        cols = st.columns(len(data_dict))
        for i, (symbol, df) in enumerate(data_dict.items()):
            stats = generate_summary_stats(df)

            with cols[i]:
                st.metric(
                    label=f"📊 {symbol}",
                    value=format_percentage(stats['short_percent']['avg'], 1),
                    delta=f"{len(df)} trading days",
                    help=f"Average short % over {len(df)} trading days"
                )

        st.markdown("---")

        # Detailed statistics for each symbol
        for symbol, df in data_dict.items():
            with st.expander(f"📋 {symbol} - Detailed Statistics", expanded=True):
                stats = generate_summary_stats(df)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**📊 Volume Metrics**")
                    st.metric("Total Volume", format_number(stats['volume']['total']))
                    st.metric("Avg Daily Volume", format_number(stats['volume']['avg']))
                    st.metric("Total Short Volume", format_number(stats['short_volume']['total']))

                with col2:
                    st.markdown("**📈 Short Percentage**")
                    st.metric("Average", format_percentage(stats['short_percent']['avg']))
                    st.metric("Median", format_percentage(stats['short_percent']['median']))
                    st.metric("Range", f"{format_percentage(stats['short_percent']['min'], 1)} - {format_percentage(stats['short_percent']['max'], 1)}")

                with col3:
                    st.markdown("**⚠️ Short Exempt**")
                    st.metric("Total Exempt Volume", format_number(stats['short_exempt']['total']))
                    st.metric("Avg Exempt Ratio", format_percentage(stats['short_exempt']['avg_exempt_ratio'], 4))
                    st.metric("Days with Exempt", f"{stats['short_exempt']['days_with_exempt']}/{stats['trading_days']}")

    # ===== TAB 2: TIME ANALYSIS =====
    with tabs[1]:
        st.subheader("Time-Based Analysis")

        for symbol, df in data_dict.items():
            st.markdown(f"### {symbol}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Average by Day of Week")
                dow_df = analyze_by_day_of_week(df)
                if not dow_df.empty:
                    st.dataframe(
                        dow_df[['ShortVolume_mean', 'ShortPercent_mean']],
                        width='stretch'
                    )

            with col2:
                st.markdown("#### Monthly Aggregation")
                monthly_df = analyze_by_month(df)
                if not monthly_df.empty:
                    st.dataframe(
                        monthly_df[['TradingDays', 'ShortVolume_sum', 'ShortPercent_mean']].head(6),
                        width='stretch'
                    )

            # Top days
            st.markdown("#### Top 10 Highest Short Volume Days")
            top_days = get_top_days(df, metric='ShortVolume', n=10)
            if not top_days.empty:
                display_df = top_days[['Date', 'ShortVolume', 'TotalVolume', 'ShortPercent']].copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(display_df, width='stretch', hide_index=True)

            st.markdown("---")

    # ===== TAB 3: ANOMALIES =====
    with tabs[2]:
        st.subheader("Anomaly Detection & Data Quality")

        for symbol, df in data_dict.items():
            st.markdown(f"### {symbol}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Short Exempt Analysis")
                exempt_analysis = analyze_short_exempt(df)

                if exempt_analysis:
                    overview = exempt_analysis.get('overview', {})
                    st.metric("Total Exempt Volume", format_number(overview.get('total_exempt', 0)))
                    st.metric("Overall Exempt Ratio", format_percentage(overview.get('overall_ratio', 0), 4))
                    st.metric("Days with Exempt", f"{overview.get('days_with_exempt', 0)} ({format_percentage(overview.get('percent_days_with_exempt', 0), 1)})")

                    if 'anomaly_detection' in exempt_analysis:
                        anomaly_det = exempt_analysis['anomaly_detection']
                        st.warning(f"⚠️ Found {anomaly_det['count_2std']} days exceeding 2σ threshold")

                        if exempt_analysis.get('anomalies'):
                            st.markdown("**Anomalous Days:**")
                            for anomaly in exempt_analysis['anomalies'][:5]:
                                st.text(f"{anomaly['Date'].strftime('%Y-%m-%d')}: {format_number(anomaly['ShortExemptVolume'])} ({format_percentage(anomaly['ShortExemptRatio'], 2)})")

            with col2:
                st.markdown("#### Data Quality Checks")
                discrepancies = analyze_discrepancies(df)

                if discrepancies:
                    # Data integrity
                    invalid_short = discrepancies.get('invalid_short_volume', {})
                    if invalid_short['count'] > 0:
                        st.error(f"❌ {invalid_short['count']} days with ShortVolume > TotalVolume")
                    else:
                        st.success("✅ No data integrity issues")

                    # Outliers
                    outliers = discrepancies.get('short_percent_outliers', {})
                    if outliers:
                        st.info(f"📊 Mean Short %: {format_percentage(outliers['mean'])}")
                        st.info(f"📈 High outliers: {outliers['high_outliers']}")
                        st.info(f"📉 Low outliers: {outliers['low_outliers']}")

                    # Large changes
                    changes = discrepancies.get('large_volume_changes', {})
                    if changes['count'] > 0:
                        st.warning(f"⚠️ {changes['count']} days with >200% volume change")

            st.markdown("---")

    # ===== TAB 4: CHARTS =====
    with tabs[3]:
        st.subheader("Interactive Visualizations")

        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Short % Timeline",
                "Short Volume Timeline",
                "Exempt Volume Timeline",
                "Short % Distribution",
                "Day of Week Analysis",
                "Volume Breakdown",
                "Monthly Trends"
            ]
        )

        for symbol, df in data_dict.items():
            if chart_type == "Short % Timeline":
                fig = plot_short_percentage_timeline(df, symbol)
            elif chart_type == "Short Volume Timeline":
                fig = plot_short_volume_timeline(df, symbol)
            elif chart_type == "Exempt Volume Timeline":
                fig = plot_exempt_volume_timeline(df, symbol)
            elif chart_type == "Short % Distribution":
                fig = plot_short_pct_distribution(df, symbol)
            elif chart_type == "Day of Week Analysis":
                fig = plot_day_of_week_analysis(df, symbol)
            elif chart_type == "Volume Breakdown":
                fig = plot_volume_breakdown(df, symbol)
            elif chart_type == "Monthly Trends":
                fig = plot_monthly_trends(df, symbol)
            else:
                fig = None

            if fig:
                st.plotly_chart(fig, width='stretch')

    # ===== TAB 5: COMPARISON (only if multiple symbols) =====
    if len(data_dict) > 1:
        with tabs[4]:
            st.subheader("Multi-Symbol Comparison")

            # Comparison table
            st.markdown("### Summary Comparison")
            comp_table = compare_symbols(data_dict)
            if not comp_table.empty:
                st.dataframe(comp_table, width='stretch', hide_index=True)

            st.markdown("---")

            # Overlay charts
            st.markdown("### Comparative Charts")

            comparison_metric = st.selectbox(
                "Select metric to compare",
                ["ShortPercent", "ShortVolume", "ShortExemptRatio"]
            )

            fig_comparison = plot_multi_symbol_comparison(data_dict, comparison_metric)
            st.plotly_chart(fig_comparison, width='stretch')

            # Correlation matrix
            st.markdown("### Correlation Analysis")
            corr_matrix = calculate_correlation_matrix(data_dict, 'ShortPercent')

            if not corr_matrix.empty:
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig_corr = plot_correlation_heatmap(corr_matrix)
                    st.plotly_chart(fig_corr, width='stretch')

                with col2:
                    st.markdown("**Correlation Matrix**")
                    st.dataframe(corr_matrix, width='stretch')

    # ===== TAB 6: EXPORT =====
    with tabs[5]:
        st.subheader("Export Data")

        st.markdown("Download analyzed data in various formats:")

        for symbol, df in data_dict.items():
            st.markdown(f"### {symbol}")

            col1, col2 = st.columns(2)

            with col1:
                # CSV export
                csv = df.to_csv(index=False)
                st.download_button(
                    label=f"📄 Download {symbol} CSV",
                    data=csv,
                    file_name=f"finra_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    width="stretch"
                )

            with col2:
                # Excel export
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Data', index=False)

                    # Add summary sheet
                    stats = generate_summary_stats(df)
                    stats_df = pd.DataFrame([stats])
                    stats_df.to_excel(writer, sheet_name='Summary', index=False)

                output.seek(0)

                st.download_button(
                    label=f"📊 Download {symbol} Excel",
                    data=output,
                    file_name=f"finra_{symbol}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch"
                )

else:
    # Landing page before analysis
    st.info("👈 Configure your analysis in the sidebar and click 'Run Analysis'")

    st.markdown("""
    ### 📊 Features

    - **Analyze Any Stock**: Search FINRA daily short sale data for any ticker symbol
    - **Interactive Charts**: Zoom, pan, and explore data with Plotly visualizations
    - **Time Analysis**: View trends by day of week, week, and month
    - **Anomaly Detection**: Automatically detect unusual short exempt volume patterns
    - **Multi-Symbol Comparison**: Compare up to 5 stocks side-by-side
    - **Export**: Download data as CSV or Excel for further analysis

    ### 🚀 Quick Start

    1. **Enter symbols** in the sidebar (e.g., MSOS, MSOX) - one per line
    2. **Select time period** using the slider (1-12 months)
    3. **Click "Run Analysis"** to fetch and analyze data

    ### 📖 About the Data

    This tool fetches daily short sale data from [FINRA](https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files),
    which includes:
    - Short volume (total shares sold short)
    - Short exempt volume (shorts exempt from Reg SHO)
    - Total volume
    - Calculated metrics (short %, exempt ratios, trends)

    Data is cached for 24 hours to improve performance and minimize server load.
    """)

    # Example usage
    with st.expander("💡 Example Analysis"):
        st.markdown("""
        **Example: Analyzing MSOS**

        1. Enter `MSOS` in the symbols box
        2. Select 3 months
        3. Click "Run Analysis"
        4. Explore the tabs:
           - **Overview**: See summary statistics and key metrics
           - **Time Analysis**: View patterns by day/week/month
           - **Anomalies**: Check for unusual short exempt activity
           - **Charts**: Interactive visualizations
           - **Export**: Download the data

        **Multi-Symbol Comparison**

        Enter multiple symbols (one per line):
        ```
        MSOS
        MSOX
        SPY
        ```

        The "Comparison" tab will show correlations and overlaid charts.
        """)
