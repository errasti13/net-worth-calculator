import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import yfinance as yf
from typing import Dict, Optional

# Page configuration
st.set_page_config(
    page_title="Net Worth Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .positive-change {
        color: #00c851;
        font-weight: bold;
    }
    .negative-change {
        color: #ff4444;
        font-weight: bold;
    }
    .stDataFrame {
        background-color: white;
    }
    .summary-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .currency-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CurrencyConverter:
    """Handles currency conversion using Yahoo Finance API"""
    
    def __init__(self):
        self.exchange_rates = {}
        self.last_updated = None
        
    def get_exchange_rate(self, from_currency: str, to_currency: str = "EUR") -> float:
        """Get exchange rate from Yahoo Finance"""
        if from_currency == to_currency:
            return 1.0
            
        try:
            # Create currency pair symbol for Yahoo Finance
            symbol = f"{from_currency}{to_currency}=X"
            
            # Download exchange rate data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            
            if not data.empty:
                rate = data['Close'].iloc[-1]
                self.exchange_rates[f"{from_currency}_{to_currency}"] = rate
                self.last_updated = datetime.now()
                return rate
            else:
                st.warning(f"Could not fetch exchange rate for {from_currency} to {to_currency}")
                return 1.0
                
        except Exception as e:
            st.error(f"Error fetching exchange rate for {from_currency}: {str(e)}")
            # Fallback rates (approximate)
            fallback_rates = {
                "CHF_EUR": 1.07,  # Based on recent data
                "USD_EUR": 0.95,
                "GBP_EUR": 1.20
            }
            return fallback_rates.get(f"{from_currency}_{to_currency}", 1.0)

class MultiCurrencyNetWorthAnalyzer:
    def __init__(self, data_source, currency_config, is_file_upload=False):
        self.data_source = data_source  # Can be file path or uploaded file
        self.currency_config = currency_config  # Dict mapping column names to currencies
        self.base_currency = "EUR"
        self.df = None
        self.df_original = None  # Store original data
        self.converter = CurrencyConverter()
        self.is_file_upload = is_file_upload
        self.load_data()
    
    def load_data(self):
        """Load and process the multi-currency net worth data"""
        try:
            if self.is_file_upload:
                # Handle uploaded file
                if self.data_source is not None:
                    # Load original data from uploaded file
                    self.df_original = pd.read_csv(self.data_source)
                    self.df_original['Date'] = pd.to_datetime(self.df_original['Date'])
                    self.df_original = self.df_original.sort_values('Date')
                else:
                    return False
            else:
                # Handle file path
                if os.path.exists(self.data_source):
                    # Load original data
                    self.df_original = pd.read_csv(self.data_source)
                    self.df_original['Date'] = pd.to_datetime(self.df_original['Date'])
                    self.df_original = self.df_original.sort_values('Date')
                else:
                    st.error(f"Data file not found: {self.data_source}")
                    return False
            
            # Create a copy for conversion
            self.df = self.df_original.copy()
            
            # Convert all currencies to EUR
            self.convert_currencies_to_eur()
            
            # Calculate net worth for each month
            account_columns = [col for col in self.df.columns if col not in ['Date']]
            
            # All columns are assets in this case (no debts specified)
            self.df['Total Assets'] = self.df[account_columns].sum(axis=1)
            self.df['Total Debts'] = 0  # No debt columns specified
            self.df['Net Worth'] = self.df['Total Assets'] - self.df['Total Debts']
            
            # Calculate monthly changes
            self.df['Net Worth Change'] = self.df['Net Worth'].diff()
            self.df['Assets Change'] = self.df['Total Assets'].diff()
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def convert_currencies_to_eur(self):
        """Convert all account balances to EUR using current exchange rates"""
        conversion_info = []
        
        for column in self.df.columns:
            if column == 'Date':
                continue
                
            currency = self.currency_config.get(column, "EUR")
            
            if currency != "EUR":
                # Get exchange rate
                rate = self.converter.get_exchange_rate(currency, "EUR")
                
                # Convert the column
                self.df[column] = self.df[column] * rate
                
                conversion_info.append({
                    'Account': column,
                    'Original Currency': currency,
                    'Exchange Rate': f"1 {currency} = {rate:.4f} EUR",
                    'Last Updated': self.converter.last_updated.strftime("%Y-%m-%d %H:%M") if self.converter.last_updated else "N/A"
                })
            else:
                conversion_info.append({
                    'Account': column,
                    'Original Currency': currency,
                    'Exchange Rate': "1 EUR = 1 EUR",
                    'Last Updated': "N/A"
                })
        
        # Store conversion info for display
        self.conversion_info = pd.DataFrame(conversion_info)
    
    def get_summary_stats(self):
        """Calculate summary statistics"""
        if self.df is None or self.df.empty:
            return None
        
        latest = self.df.iloc[-1]
        previous = self.df.iloc[-2] if len(self.df) > 1 else None
        
        total_months = len(self.df)
        avg_monthly_change = self.df['Net Worth Change'].mean() if total_months > 1 else 0
        total_growth = self.df['Net Worth'].iloc[-1] - self.df['Net Worth'].iloc[0] if total_months > 1 else 0
        
        return {
            'current_net_worth': latest['Net Worth'],
            'current_assets': latest['Total Assets'],
            'current_debts': latest['Total Debts'],
            'last_month_change': latest['Net Worth Change'] if previous is not None else 0,
            'avg_monthly_change': avg_monthly_change,
            'total_growth': total_growth,
            'total_months': total_months
        }

def create_net_worth_chart(df):
    """Create the main net worth trend chart"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Net Worth Over Time (EUR)', 'Monthly Change (EUR)'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Net worth trend
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Net Worth'],
            mode='lines+markers',
            name='Net Worth',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8),
            hovertemplate='<b>Date:</b> %{x}<br><b>Net Worth:</b> €%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Monthly change bars
    colors = ['green' if x >= 0 else 'red' for x in df['Net Worth Change']]
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Net Worth Change'],
            name='Monthly Change',
            marker_color=colors,
            hovertemplate='<b>Date:</b> %{x}<br><b>Change:</b> €%{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Multi-Currency Net Worth Analysis (Converted to EUR)",
        title_x=0.5,
        title_font_size=20
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Net Worth (EUR)", row=1, col=1)
    fig.update_yaxes(title_text="Monthly Change (EUR)", row=2, col=1)
    
    return fig

def create_assets_breakdown_chart(df, currency_config):
    """Create assets breakdown chart"""
    latest_data = df.iloc[-1]
    
    # Get account columns
    account_columns = [col for col in df.columns if col not in ['Date', 'Net Worth', 'Total Assets', 'Total Debts', 'Net Worth Change', 'Assets Change']]
    
    asset_values = [latest_data[col] for col in account_columns if latest_data[col] > 0]
    asset_labels = [f"{col}\n(Originally {currency_config.get(col, 'EUR')})" for col in account_columns if latest_data[col] > 0]
    
    fig = px.pie(
        values=asset_values,
        names=asset_labels,
        title="Current Asset Allocation (EUR)",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Amount: €%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        height=400,
        title_font_size=16,
        title_x=0.5
    )
    
    return fig

def create_account_trends_chart(df, df_original, currency_config):
    """Create individual account trends chart with both original and converted values"""
    account_columns = [col for col in df.columns if col not in ['Date', 'Net Worth', 'Total Assets', 'Total Debts', 'Net Worth Change', 'Assets Change']]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Account Trends (EUR - Converted)', 'Account Trends (Original Currencies)'),
        vertical_spacing=0.1
    )
    
    # Converted values (EUR)
    for col in account_columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df[col],
                mode='lines+markers',
                name=f"{col} (EUR)",
                hovertemplate=f'<b>{col} (EUR)</b><br>Date: %{{x}}<br>Amount: €%{{y:,.0f}}<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Original values
    for col in account_columns:
        currency = currency_config.get(col, "EUR")
        fig.add_trace(
            go.Scatter(
                x=df_original['Date'],
                y=df_original[col],
                mode='lines+markers',
                name=f"{col} ({currency})",
                hovertemplate=f'<b>{col} ({currency})</b><br>Date: %{{x}}<br>Amount: %{{y:,.0f}} {currency}<extra></extra>',
                showlegend=True,
                line=dict(dash='dash')
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title="Account Trends Comparison",
        title_x=0.5,
        title_font_size=16,
        height=600,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Amount (EUR)", row=1, col=1)
    fig.update_yaxes(title_text="Amount (Original Currency)", row=2, col=1)
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 Multi-Currency Net Worth Tracker</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### 📁 Data Source")
    
    # Create tabs for different data input methods
    tab1, tab2 = st.tabs(["📤 Upload CSV File", "📂 Use Local File"])
    
    analyzer = None
    
    with tab1:
        st.markdown("Upload your CSV file containing net worth data:")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with Date column and account balances"
        )
        
        if uploaded_file is not None:
            # Store uploaded file in session state
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['uploaded_file_name'] = uploaded_file.name
            
            # Display file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Preview the data
            try:
                preview_df = pd.read_csv(uploaded_file)
                st.markdown("**📊 Data Preview:**")
                st.dataframe(preview_df.head(), use_container_width=True)
                
                # Store column info for later use
                st.session_state['account_columns'] = [col for col in preview_df.columns if col.lower() != 'date']
                
                st.info("👇 Scroll down to configure currencies and process your data")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.info("Please ensure your CSV file has the correct format with Date column and account balances.")
    
    with tab2:
        st.markdown("Use a local file (for development/testing):")
        
        # Default currency configuration
        default_currency_config = {
            'UBS Account': 'CHF',
            'IBKR Account': 'CHF', 
            'Kutxabank Account': 'EUR'
        }
        
        # Check if default file exists
        default_data_path = "data/net_worth_data.csv"
        if os.path.exists(default_data_path):
            st.info(f"✅ Using local file: {default_data_path}")
            analyzer = MultiCurrencyNetWorthAnalyzer(default_data_path, default_currency_config, is_file_upload=False)
        else:
            st.warning(f"⚠️ Local file not found: {default_data_path}")
            st.markdown("""
            **To use local file:**
            1. Create a `data` folder in your project directory
            2. Add `net_worth_data.csv` file with your data
            3. Refresh this page
            """)
    
    # Currency Configuration Section (moved to bottom for better UX)
    if 'uploaded_file' in st.session_state and 'account_columns' in st.session_state:
        st.markdown("---")
        st.markdown("### 💱 Currency Configuration")
        st.markdown(f"Configure currencies for **{st.session_state['uploaded_file_name']}**:")
        
        account_columns = st.session_state['account_columns']
        currency_config = {}
        
        # Create currency selectors in a more compact layout
        cols = st.columns(min(3, len(account_columns)))
        for i, account in enumerate(account_columns):
            with cols[i % len(cols)]:
                currency_config[account] = st.selectbox(
                    f"{account}",
                    ["EUR", "CHF", "USD", "GBP", "JPY", "CAD", "AUD"],
                    key=f"currency_{account}",
                    index=1 if "UBS" in account or "IBKR" in account else 0,
                    help=f"Select currency for {account}"
                )
        
        # Process button
        if st.button("🚀 Process Data with Selected Currencies", type="primary"):
            try:
                # Get the uploaded file from session state
                uploaded_file = st.session_state['uploaded_file']
                uploaded_file.seek(0)  # Reset file pointer
                analyzer = MultiCurrencyNetWorthAnalyzer(uploaded_file, currency_config, is_file_upload=True)
                
                if analyzer.df is not None:
                    st.success("✅ Data processed successfully with currency conversions!")
                    # Store analyzer in session state for persistence
                    st.session_state['analyzer'] = analyzer
                    st.rerun()
                else:
                    st.error("❌ Error processing data. Please check your file format.")
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
        else:
            st.info("👆 Configure currencies above and click 'Process Data' to continue with analysis")
    
    # Check if we have an analyzer from session state
    if 'analyzer' in st.session_state:
        analyzer = st.session_state['analyzer']
    
    # Only proceed if we have a valid analyzer
    if analyzer is None or analyzer.df is None:
        st.info("👆 Please upload a CSV file or ensure local data file exists to continue.")
        
        # Show expected CSV format
        st.markdown("### 📋 Expected CSV Format")
        st.markdown("Your CSV file should have the following structure:")
        
        sample_data = {
            'Date': ['2024-10-31', '2024-11-30', '2024-12-31'],
            'Account 1': [1000.00, 1050.00, 1100.00],
            'Account 2': [5000.00, 5200.00, 5400.00],
            'Account 3': [2000.00, 2100.00, 2150.00]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        st.markdown("""
        **Requirements:**
        - `Date` column in YYYY-MM-DD format (last day of month)
        - Account columns with numeric values
        - Monthly data points
        """)
        return
    
    # Currency conversion info
    st.markdown('<div class="currency-info">', unsafe_allow_html=True)
    st.markdown("### 💱 Currency Conversion Information")
    st.markdown("All amounts are converted to EUR using real-time exchange rates from Yahoo Finance.")
    
    # Display conversion rates
    with st.expander("View Exchange Rates Used"):
        st.dataframe(analyzer.conversion_info, hide_index=True, use_container_width=True)
        
        # Refresh rates button
        if st.button("🔄 Refresh Exchange Rates"):
            analyzer.converter = CurrencyConverter()  # Reset converter
            analyzer.convert_currencies_to_eur()
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Navigation")
    
    # Clear data button
    if st.sidebar.button("🗑️ Clear Data & Start Over"):
        # Clear all session state related to uploaded files and analyzer
        keys_to_clear = ['uploaded_file', 'uploaded_file_name', 'account_columns', 'analyzer']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    page = st.sidebar.selectbox(
        "Choose a view:",
        ["📈 Dashboard", "🔍 Detailed Analysis", "📊 Account Trends", "📋 Data View", "💱 Currency Settings"]
    )
    
    # Get summary statistics
    stats = analyzer.get_summary_stats()
    
    if page == "📈 Dashboard":
        # Summary metrics
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "💰 Current Net Worth",
                f"€{stats['current_net_worth']:,.0f}",
                f"€{stats['last_month_change']:,.0f}" if stats['last_month_change'] != 0 else None
            )
        
        with col2:
            st.metric(
                "📈 Total Assets",
                f"€{stats['current_assets']:,.0f}",
                None
            )
        
        with col3:
            st.metric(
                "📊 Total Growth",
                f"€{stats['total_growth']:,.0f}",
                f"Avg: €{stats['avg_monthly_change']:,.0f}/month"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Key insights
        col1, col2 = st.columns(2)
        
        with col1:
            if stats['last_month_change'] > 0:
                st.success(f"✅ **Last Month:** +€{stats['last_month_change']:,.0f} (Growth!)")
            elif stats['last_month_change'] < 0:
                st.error(f"⚠️ **Last Month:** €{stats['last_month_change']:,.0f} (Decline)")
            else:
                st.info("➖ **Last Month:** No change")
        
        with col2:
            st.info(f"📊 **Tracking Period:** {stats['total_months']} months")
        
        # Main chart
        st.plotly_chart(create_net_worth_chart(analyzer.df), use_container_width=True)
        
        # Asset breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_assets_breakdown_chart(analyzer.df, analyzer.currency_config), use_container_width=True)
        
        with col2:
            st.subheader("📋 Recent Performance")
            recent_data = analyzer.df.tail(6)[['Date', 'Net Worth', 'Net Worth Change']].copy()
            recent_data['Date'] = recent_data['Date'].dt.strftime('%Y-%m')
            recent_data['Net Worth'] = recent_data['Net Worth'].apply(lambda x: f"€{x:,.0f}")
            recent_data['Change'] = recent_data['Net Worth Change'].apply(
                lambda x: f"+€{x:,.0f}" if x > 0 else f"€{x:,.0f}" if x < 0 else "€0"
            )
            recent_data = recent_data.drop('Net Worth Change', axis=1)
            st.dataframe(recent_data, hide_index=True, use_container_width=True)
    
    elif page == "🔍 Detailed Analysis":
        st.header("🔍 Detailed Financial Analysis")
        
        # Time period selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", analyzer.df['Date'].min())
        with col2:
            end_date = st.date_input("End Date", analyzer.df['Date'].max())
        
        # Filter data
        mask = (analyzer.df['Date'] >= pd.Timestamp(start_date)) & (analyzer.df['Date'] <= pd.Timestamp(end_date))
        filtered_df = analyzer.df[mask]
        
        if not filtered_df.empty:
            # Filtered metrics
            period_growth = filtered_df['Net Worth'].iloc[-1] - filtered_df['Net Worth'].iloc[0]
            period_months = len(filtered_df)
            avg_monthly_growth = filtered_df['Net Worth Change'].mean() if period_months > 1 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Period Growth", f"€{period_growth:,.0f}")
            with col2:
                st.metric("Months Analyzed", period_months)
            with col3:
                st.metric("Avg Monthly Change", f"€{avg_monthly_growth:,.0f}")
            
            # Detailed chart
            st.plotly_chart(create_net_worth_chart(filtered_df), use_container_width=True)
            
            # Statistics table
            st.subheader("📊 Period Statistics")
            stats_data = {
                'Metric': ['Starting Net Worth', 'Ending Net Worth', 'Total Change', 'Best Month', 'Worst Month', 'Months with Growth'],
                'Value': [
                    f"€{filtered_df['Net Worth'].iloc[0]:,.0f}",
                    f"€{filtered_df['Net Worth'].iloc[-1]:,.0f}",
                    f"€{period_growth:,.0f}",
                    f"€{filtered_df['Net Worth Change'].max():,.0f}",
                    f"€{filtered_df['Net Worth Change'].min():,.0f}",
                    f"{(filtered_df['Net Worth Change'] > 0).sum()} / {period_months}"
                ]
            }
            
            st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
    
    elif page == "📊 Account Trends":
        st.header("📊 Individual Account Trends")
        
        # Create comparison chart
        st.plotly_chart(create_account_trends_chart(analyzer.df, analyzer.df_original, analyzer.currency_config), use_container_width=True)
        
        # Account performance table
        st.subheader("📈 Account Performance Summary (EUR)")
        performance_data = []
        
        account_columns = [col for col in analyzer.df.columns if col not in ['Date', 'Net Worth', 'Total Assets', 'Total Debts', 'Net Worth Change', 'Assets Change']]
        
        for account in account_columns:
            start_value = analyzer.df[account].iloc[0]
            end_value = analyzer.df[account].iloc[-1]
            change = end_value - start_value
            change_pct = (change / abs(start_value)) * 100 if start_value != 0 else 0
            currency = analyzer.currency_config.get(account, "EUR")
            
            performance_data.append({
                'Account': f"{account} ({currency}→EUR)",
                'Starting Balance': f"€{start_value:,.0f}",
                'Current Balance': f"€{end_value:,.0f}",
                'Total Change': f"€{change:,.0f}",
                'Change %': f"{change_pct:+.1f}%"
            })
        
        st.dataframe(pd.DataFrame(performance_data), hide_index=True, use_container_width=True)
    
    elif page == "📋 Data View":
        st.header("📋 Raw Data View")
        
        # Data view options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_converted = st.checkbox("Show converted (EUR) data", value=True)
        with col2:
            show_original = st.checkbox("Show original currency data", value=False)
        with col3:
            records_to_show = st.selectbox("Records to show", ["All", "Last 12 months", "Last 6 months", "Last 3 months"])
        
        # Filter data based on selection
        if show_converted:
            display_df = analyzer.df.copy()
            st.subheader("💱 Converted Data (EUR)")
        else:
            display_df = analyzer.df_original.copy()
            st.subheader("📊 Original Data")
        
        if records_to_show != "All":
            months_map = {"Last 12 months": 12, "Last 6 months": 6, "Last 3 months": 3}
            months = months_map[records_to_show]
            display_df = display_df.tail(months)
        
        # Format the data for display
        display_df_formatted = display_df.copy()
        display_df_formatted['Date'] = display_df_formatted['Date'].dt.strftime('%Y-%m-%d')
        
        # Format currency columns
        currency_columns = [col for col in display_df_formatted.columns if col != 'Date']
        for col in currency_columns:
            if show_converted or col in ['Net Worth', 'Total Assets', 'Total Debts', 'Net Worth Change', 'Assets Change']:
                display_df_formatted[col] = display_df_formatted[col].apply(lambda x: f"€{x:,.2f}")
            else:
                currency = analyzer.currency_config.get(col, "EUR")
                display_df_formatted[col] = display_df_formatted[col].apply(lambda x: f"{x:,.2f} {currency}")
        
        st.dataframe(display_df_formatted, hide_index=True, use_container_width=True)
        
        # Show both datasets if requested
        if show_original and show_converted:
            st.subheader("📊 Original Data")
            original_formatted = analyzer.df_original.copy()
            if records_to_show != "All":
                original_formatted = original_formatted.tail(months)
            
            original_formatted['Date'] = original_formatted['Date'].dt.strftime('%Y-%m-%d')
            for col in [c for c in original_formatted.columns if c != 'Date']:
                currency = analyzer.currency_config.get(col, "EUR")
                original_formatted[col] = original_formatted[col].apply(lambda x: f"{x:,.2f} {currency}")
            
            st.dataframe(original_formatted, hide_index=True, use_container_width=True)
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            csv_converted = analyzer.df.to_csv(index=False)
            st.download_button(
                label="📥 Download EUR data as CSV",
                data=csv_converted,
                file_name="net_worth_data_eur.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_original = analyzer.df_original.to_csv(index=False)
            st.download_button(
                label="📥 Download original data as CSV",
                data=csv_original,
                file_name="net_worth_data_original.csv",
                mime="text/csv"
            )
    
    elif page == "💱 Currency Settings":
        st.header("💱 Currency Configuration")
        
        st.markdown("""
        Configure which currency each account uses. The app will automatically convert 
        all amounts to EUR using real-time exchange rates from Yahoo Finance.
        """)
        
        # Current configuration
        st.subheader("📊 Current Configuration")
        config_df = pd.DataFrame([
            {'Account': account, 'Currency': currency}
            for account, currency in analyzer.currency_config.items()
        ])
        st.dataframe(config_df, hide_index=True, use_container_width=True)
        
        # Supported currencies
        st.subheader("🌍 Supported Currencies")
        st.markdown("""
        - **CHF** - Swiss Franc
        - **EUR** - Euro
        - **USD** - US Dollar
        - **GBP** - British Pound
        - **JPY** - Japanese Yen
        - **CAD** - Canadian Dollar
        - **AUD** - Australian Dollar
        - And many more...
        """)
        
        # Exchange rate test
        st.subheader("🧪 Test Exchange Rate")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_from = st.selectbox("From Currency", ["CHF", "USD", "GBP", "JPY", "CAD", "AUD"])
        
        with col2:
            test_to = st.selectbox("To Currency", ["EUR", "CHF", "USD", "GBP", "JPY", "CAD", "AUD"], index=0)
        
        with col3:
            test_amount = st.number_input("Amount", value=100.0, min_value=0.01)
        
        if st.button("Get Exchange Rate"):
            test_converter = CurrencyConverter()
            rate = test_converter.get_exchange_rate(test_from, test_to)
            converted_amount = test_amount * rate
            
            st.success(f"{test_amount:,.2f} {test_from} = {converted_amount:,.2f} {test_to}")
            st.info(f"Exchange Rate: 1 {test_from} = {rate:.4f} {test_to}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; padding: 1rem;'>"
        "💡 <strong>Tip:</strong> Exchange rates are fetched in real-time from Yahoo Finance. "
        "Update your CSV file monthly with new data to track your financial progress across currencies!"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
