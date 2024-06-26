import os
from polygon_code.polygon import get_data, check_existing_data
from sort_csv import sort_csv
import time

import pandas as pd
from datetime import datetime, timedelta

# Load the CSV file with a different encoding
file_path = 'polygon_data/polygon_tickers.csv'
df = pd.read_csv(file_path, encoding='latin1')

# Extract the tickers from the dataframe
tickers = df['ticker'].tolist()

# Generate dates for the past 2 years from today
end_date = datetime.now()
start_date = end_date - timedelta(days=2*365)
date_range = pd.date_range(start=start_date, end=end_date, freq='B').strftime('%Y-%m-%d').tolist()

# Replace 'YOUR_API_KEY' with your actual Polygon.io API key
API_KEY = 'q42WQRfUjTeouBEoaCSC_LHNhDFhzIIb'
BASE_URL = 'https://api.polygon.io/v1'

def get_stock(ticker, date):
    params = {
        'adjusted': 'true',
        'apiKey': API_KEY
    }
    url = f"{BASE_URL}/open-close/{ticker}/{date}"
    file = os.path.join("stocks", f"{ticker}.csv")
    calls = get_data(params, url, file, t_o_s='s')
    return calls

def get_stocks():
    calls = 0
    start_time = datetime.now().minute
    for ticker in tickers:
        file_name = os.path.join("polygon_data", "stocks", f"{ticker}.csv")
        existing_data = check_existing_data(file_name, 'from')
        for date in date_range:
            if date not in existing_data:
                if datetime.now().minute != start_time and calls == 5:
                    calls = 0
                    start_time = datetime.now().minute
                elif calls < 5:
                    num = get_stock(ticker, date)
                    calls += num
                else:
                    while datetime.now().minute == start_time:
                        time.sleep(1)
        sort_csv(file_name, 'from')
            
if __name__ == "__main__":
    get_stocks()