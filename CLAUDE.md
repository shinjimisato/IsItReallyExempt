# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Streamlit web application for analyzing FINRA daily short sale data for any stock symbol. The project includes both the original Jupyter notebook (legacy) and a production web app with interactive visualizations, multi-symbol comparison, and export capabilities.

**Main Application**: `app.py` - Streamlit web interface
**Legacy Notebook**: `finra_short_sale_analysis.ipynb` - Original analysis notebook

## Commands

### Running the Web App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# App will be available at http://localhost:8501
```

### Running the Notebook (Legacy)

```bash
# Start Jupyter
jupyter notebook finra_short_sale_analysis.ipynb
```

### Testing Locally

```bash
# Test with example symbols
# Open app, enter MSOS and MSOX, select 3 months, click "Run Analysis"
```

## Code Architecture

### Streamlit Application Structure

```
app.py                    # Main Streamlit application (orchestrates UI, tabs)
├── src/
│   ├── data_fetcher.py    # FINRA scraping + two-tier caching
│   ├── data_processor.py  # Data preprocessing & validation
│   ├── analyzers.py       # Statistical analysis functions
│   ├── visualizations.py  # Plotly interactive charts
│   └── utils.py           # Trading calendar, formatters
├── cache/                 # File-based cache (Parquet files, gitignored)
└── .streamlit/config.toml # Theme and server configuration
```

### Data Flow (Web App)

1. **User Input** (sidebar): Symbols, date range → app.py
2. **Data Fetching** (src/data_fetcher.py):
   - Check two-tier cache (Streamlit + file-based)
   - Fetch from FINRA if cache miss
   - Save to Parquet cache files
3. **Preprocessing** (src/data_processor.py): Calculate derived metrics
4. **Analysis** (src/analyzers.py): Generate statistics, detect anomalies
5. **Visualization** (src/visualizations.py): Create Plotly charts
6. **Display** (app.py): Render tabs with metrics, charts, export buttons

### Legacy Notebook Data Flow

1. **Configuration (Cell 4)**: Set analysis parameters including `MONTHS_TO_SCRAPE`, `TARGET_SYMBOLS`, and FINRA URL template
2. **Trading Calendar (Cell 6)**: Uses NYSE calendar via `pandas_market_calendars` to generate valid trading dates
3. **Data Scraping (Cells 8-9)**: Fetches pipe-delimited data from FINRA CDN at `https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt`
4. **Preprocessing (Cell 11)**: Calculates derived metrics (ShortPercent, ShortExemptPercent, ShortExemptRatio) and adds time-based grouping columns
5. **Analysis & Visualization (Cells 13-33)**: Multiple analysis modules examining different aspects of the data

### Key Data Fields

**Raw FINRA Fields:**
- `Date`: Trading date (YYYYMMDD format)
- `Symbol`: Stock ticker
- `ShortVolume`: Total short volume
- `ShortExemptVolume`: Short exempt volume (from Reg SHO exceptions)
- `TotalVolume`: Total daily volume
- `Market`: Trading venue

**Calculated Metrics:**
- `ShortPercent`: Short volume as % of total volume
- `ShortExemptPercent`: Short exempt volume as % of total volume
- `ShortExemptRatio`: Short exempt as % of short volume
- `NonShortVolume`: TotalVolume - ShortVolume

**Time Groupings:**
- `DayOfWeek`, `WeekNumber`, `YearWeek`, `Month`, `YearMonth`

### Analysis Modules (Web App)

The Streamlit app organizes analysis into 6 tabs:

1. **Overview Tab** (app.py:287-338): Summary statistics, key metrics cards, detailed stats per symbol
2. **Time Analysis Tab** (app.py:341-385): Day of week patterns, monthly aggregations, top volume days
3. **Anomalies Tab** (app.py:388-448): Short exempt analysis with 2σ/3σ thresholds, data quality checks
4. **Charts Tab** (app.py:451-490): Interactive Plotly visualizations (7 chart types)
5. **Comparison Tab** (app.py:493-535): Multi-symbol overlay charts, correlation matrix (if >1 symbol)
6. **Export Tab** (app.py:538-578): CSV and Excel download buttons

All analysis logic is in `src/analyzers.py`:
- `generate_summary_stats()`: Calculate overall metrics
- `get_top_days()`: Find highest volume days
- `analyze_by_day_of_week()`, `analyze_by_month()`: Time-based aggregations
- `analyze_short_exempt()`: Anomaly detection with statistical thresholds
- `analyze_discrepancies()`: Data integrity validation
- `compare_symbols()`, `calculate_correlation_matrix()`: Multi-symbol comparison

### Legacy Notebook Analysis Modules

The notebook is organized into distinct analysis sections:

1. **Summary Statistics (Cells 13-14)**: Overall metrics and per-symbol aggregates
2. **Top Days Analysis (Cells 16-17)**: Identifies highest short volume and exempt volume days
3. **Time Period Analysis (Cells 19-21)**: Breakdowns by day of week, week, and month
4. **Short Exempt Analysis (Cell 23)**: Anomaly detection using statistical thresholds (2σ, 3σ)
5. **Discrepancy Analysis (Cell 25)**: Data integrity checks and outlier identification
6. **Visualizations (Cells 27-28)**: Multi-panel dashboards and comparison charts
7. **Export (Cells 30-31)**: CSV generation for further analysis

### Important Constants

**Web App** (src/data_fetcher.py):
- `REQUEST_DELAY = 0.25`: Rate limiting for FINRA server requests (seconds)
- `FINRA_URL_TEMPLATE`: Data source pattern - FINRA publishes T+1 data daily
- `CACHE_DIR = Path("cache")`: File-based cache directory
- Cache TTL: 24 hours (86400 seconds)
- Max symbols: 5 (configurable in app.py for self-hosting)

**Analysis** (src/analyzers.py):
- Anomaly thresholds: 2σ and 3σ from mean for short exempt volume
- Large change threshold: ±200% day-over-day volume change

### Statistical Methods

**Anomaly Detection:**
- Uses mean ± 2σ/3σ for identifying unusual exempt volume
- Analyzes day-over-day changes exceeding 200%
- Correlation analysis between exempt and total/short volume

**Data Integrity Checks:**
- Validates ShortVolume ≤ TotalVolume
- Validates ShortExemptVolume ≤ ShortVolume
- Identifies outliers in short percentage distribution

## Caching Strategy (Web App)

**Two-tier caching** for optimal performance without a database:

1. **Streamlit Cache** (`@st.cache_data` decorator):
   - In-memory session-level caching
   - TTL: 24 hours
   - Cleared on app restart or manual clear

2. **File Cache** (Parquet format):
   - Persistent across sessions
   - Location: `cache/{symbol}_{start_date}_{end_date}.parquet`
   - Checked before fetching from FINRA
   - Auto-cleaned after 7 days (src/data_fetcher.py:clear_old_cache)

Cache flow: Check Streamlit cache → Check file cache → Fetch from FINRA → Save to both caches

## Deployment

**Streamlit Community Cloud** (free tier):
- Memory limit: 1GB RAM (reason for 5 symbol limit)
- Deploy via GitHub integration
- Auto-detects `app.py` and `requirements.txt`
- Configuration in `.streamlit/config.toml`

**Local Development**:
```bash
streamlit run app.py  # Runs on http://localhost:8501
```

## Notes

- FINRA data is T+1 (published next business day)
- Short exempt volume represents trades exempt from Reg SHO requirements (e.g., market maker exemptions)
- Uses NYSE calendar for trading days - data may not exist for market holidays
- 404 responses from FINRA are expected for future dates or market holidays
- Web app caches data for 24 hours; use "Clear Cache" button to force refresh
- Legacy notebook caches in memory - rerun Cell 9 to refresh from FINRA's CDN
