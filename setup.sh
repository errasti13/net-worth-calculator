#!/bin/bash

# Net Worth Tracker Setup Script
echo "🚀 Setting up Net Worth Tracker..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    echo "📁 Creating data directory..."
    mkdir -p data
fi

# Check if data file exists
if [ ! -f "data/net_worth_data.csv" ]; then
    echo "⚠️  No data file found. Using the provided sample data."
    echo "💡 You can update 'data/net_worth_data.csv' with your own financial data."
fi

echo "✅ Setup complete!"
echo ""
echo "🎯 To run the application:"
echo "   1. Activate the virtual environment: source venv/bin/activate"
echo "   2. Run the app: streamlit run app.py"
echo "   3. Open your browser to: http://localhost:8501"
echo ""
echo "📝 Don't forget to update your data monthly in 'data/net_worth_data.csv'"
