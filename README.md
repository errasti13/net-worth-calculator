# Net Worth Tracker

A comprehensive Streamlit application for tracking personal net worth over time.

## Features

- **📈 Interactive Dashboard**: Visual representation of net worth trends with monthly changes
- **💰 Asset Allocation**: Pie chart showing current asset distribution
- **📊 Detailed Analysis**: Filter data by time periods and view detailed statistics
- **📋 Account Trends**: Track individual accounts over time
- **📥 Data Export**: Download your data as CSV

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Access the App**: Open your browser and go to `http://localhost:8501`

## Data Format

The application expects a CSV file at `data/net_worth_data.csv` with the following format:

```csv
Date,Checking Account,Savings Account,Investment Account,Retirement 401k,Credit Card Debt,Student Loans,Emergency Fund,Crypto Portfolio,Real Estate,Other Assets
2024-01-31,2500.00,15000.00,25000.00,45000.00,-3200.00,-15000.00,8000.00,3500.00,0.00,1200.00
```

### Column Guidelines:
- **Date**: Format as YYYY-MM-DD (end of month)
- **Assets**: Positive values (checking, savings, investments, etc.)
- **Debts**: Negative values or columns with "debt"/"loan" in the name
- **Currency**: Use decimal format (e.g., 1000.50)

## File Structure

```
net-worth/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── data/
│   └── net_worth_data.csv # Your financial data
├── README.md             # This file
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## Updating Your Data

1. **Monthly Update**: Add a new row to `data/net_worth_data.csv` at the end of each month
2. **Account Changes**: Modify column headers if you open/close accounts
3. **Automatic Refresh**: The app will automatically reflect changes when you refresh the page

## Key Metrics Tracked

- **Net Worth**: Total Assets - Total Debts
- **Monthly Change**: Month-over-month net worth difference
- **Asset Allocation**: Breakdown of where your money is invested
- **Growth Trends**: Long-term financial progress visualization

## Production Deployment

### Option 1: Streamlit Cloud
1. Push your code to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy directly from your repository

### Option 2: Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Option 3: Local Server
```bash
# Run on custom port
streamlit run app.py --server.port 8502

# Run in production mode
streamlit run app.py --server.headless true
```

## Customization

### Adding New Account Types
1. Add new columns to your CSV file
2. The app automatically detects and categorizes them
3. Debt accounts should contain "debt" or "loan" in the column name

### Modifying Charts
- Edit the chart creation functions in `app.py`
- Customize colors, layouts, and chart types
- Add new visualization types as needed

## Security Notes

- **Data Privacy**: All data processing happens locally
- **File Access**: App only reads from the specified CSV file
- **No External APIs**: No data is sent to external services

## Troubleshooting

### Common Issues

1. **CSV Format Error**: Ensure dates are in YYYY-MM-DD format
2. **Import Errors**: Install all dependencies with `pip install -r requirements.txt`
3. **Data Not Loading**: Check that `data/net_worth_data.csv` exists and is readable

### Performance Tips

- Keep CSV files under 10MB for optimal performance
- Use monthly data points rather than daily for better visualization
- Archive old data if the file becomes too large

## Contributing

Feel free to customize this application for your needs:
- Add new chart types
- Implement data validation
- Create additional analysis views
- Add export formats (PDF, Excel)

## License

This project is open source and available under the MIT License.
