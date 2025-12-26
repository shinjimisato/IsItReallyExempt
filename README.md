# FINRA Short Sale Analysis Web App

A Streamlit web application for analyzing FINRA daily short sale data. Search any stock symbol, compare multiple stocks, detect anomalies in short exempt volume, and export data for further analysis.

## Features

- **📊 Analyze Any Stock**: Fetch and analyze FINRA short sale data for any ticker symbol
- **📈 Interactive Charts**: Zoom, pan, and explore with Plotly visualizations
- **📅 Time Analysis**: View patterns by day of week, week, and month
- **🔍 Anomaly Detection**: Automatically identify unusual short exempt volume patterns
- **⚖️ Multi-Symbol Comparison**: Compare up to 5 stocks side-by-side with correlation analysis
- **💾 Export**: Download data as CSV or Excel
- **⚡ Smart Caching**: Two-tier caching (in-memory + file-based) for fast performance

## Demo

> The app can be deployed to Streamlit Community Cloud for free hosting.

## Quick Start

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/short-exempt.git
   cd short-exempt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**

   The app will automatically open at `http://localhost:8501`

### Using the App

1. **Enter stock symbols** in the sidebar (one per line, max 5)
   ```
   MSOS
   MSOX
   SPY
   ```

2. **Select analysis period** using the slider (1-12 months)

3. **Click "Run Analysis"** to fetch and analyze data

4. **Explore the tabs**:
   - **Overview**: Summary statistics and key metrics
   - **Time Analysis**: Trends by day/week/month, top volume days
   - **Anomalies**: Short exempt analysis and data quality checks
   - **Charts**: Interactive visualizations (7 chart types)
   - **Comparison**: Multi-symbol overlays and correlations (if >1 symbol)
   - **Export**: Download CSV/Excel files

## Deployment to Streamlit Community Cloud

### Prerequisites

- GitHub account
- Streamlit Community Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Deployment Steps

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **App will be live at**: `https://[your-app-name].streamlit.app`

### Configuration

The app is pre-configured for Streamlit Cloud with:
- Memory-optimized caching
- 5 symbol limit (free tier: 1GB RAM)
- 24-hour cache TTL
- File-based persistence

## Project Structure

```
short-exempt/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit theme configuration
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py        # FINRA data scraping + caching
│   ├── data_processor.py      # Data preprocessing & validation
│   ├── analyzers.py           # Statistical analysis functions
│   ├── visualizations.py      # Plotly chart generation
│   └── utils.py               # Utility functions (calendar, formatters)
├── cache/                      # File-based cache (gitignored)
├── .gitignore
├── README.md
└── CLAUDE.md                   # Development documentation
```

## Technical Details

### Data Source

Data is fetched from [FINRA's daily short sale volume files](https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files):
- URL pattern: `https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt`
- Format: Pipe-delimited text files
- Update frequency: Daily (T+1)
- Historical data: Available since 2009

### Caching Strategy

**Two-tier caching** for optimal performance:

1. **In-Memory Cache** (`@st.cache_data`):
   - Session-level caching
   - Expires after 24 hours
   - Instant access for repeated queries

2. **File-Based Cache** (Parquet files):
   - Persistent across sessions
   - Stored in `cache/` directory
   - Format: `{symbol}_{start_date}_{end_date}.parquet`
   - Automatically cleaned after 7 days

### Calculated Metrics

The app calculates several derived metrics:

- **ShortPercent**: Short volume / Total volume × 100
- **ShortExemptPercent**: Short exempt volume / Total volume × 100
- **ShortExemptRatio**: Short exempt volume / Short volume × 100
- **NonShortVolume**: Total volume - Short volume

### Anomaly Detection

Short exempt volume anomalies are detected using statistical thresholds:
- **2 Standard Deviations**: Flagged as unusual
- **3 Standard Deviations**: Highlighted as significant anomaly

## Development

### Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

### Running Tests

```bash
# Install test dependencies
pip install pytest

# Run tests (when implemented)
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## FAQ

**Q: Why is data missing for certain dates?**

A: FINRA publishes data on a T+1 basis for trading days only. Weekends, holidays, and future dates will have no data.

**Q: Can I analyze more than 5 symbols?**

A: The limit is set to 5 for performance on Streamlit Cloud's free tier (1GB RAM). You can increase this in `app.py` if self-hosting.

**Q: How often is the cache updated?**

A: Cache expires after 24 hours. You can manually clear cache using the sidebar button.

**Q: What does "short exempt" mean?**

A: Short exempt volume represents shares sold short that are exempt from Regulation SHO requirements, typically due to market maker exemptions or other regulatory exceptions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data source: [FINRA](https://www.finra.org/)
- Built with [Streamlit](https://streamlit.io/)
- Charts powered by [Plotly](https://plotly.com/)
- Trading calendar from [pandas-market-calendars](https://github.com/rsheftel/pandas_market_calendars)

## Support

For issues, questions, or contributions, please open an issue on GitHub.
