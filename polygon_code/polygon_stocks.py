import os
import threading
from polygon_code.polygon import get_data, check_existing_data
from sort_csv import sort_csv
import pandas as pd
from datetime import datetime, timedelta

# Load the CSV file with a different encoding
file_path = 'polygon_data/polygon_tickers.csv'
df = pd.read_csv(file_path, encoding='latin1')

# Extract the tickers from the dataframe and ensure they are strings
tickers = df['ticker'].tolist()

# Generate dates for the past 2 years from today
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)
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
    get_data(params, url, file, t_o_s='s')

def process_tickers(tickers_subset):
    for ticker in tickers_subset:
        file_name = os.path.join("polygon_data", "stocks", f"{ticker}.csv")
        existing_data = check_existing_data(file_name, 'from')
        for date in date_range:
            if date not in existing_data:
                get_stock(ticker, date)
        sort_csv(file_name, 'from')

def thread_stocks():
    threads = []
    num_threads = 18
    # Divide tickers into 18 roughly equal parts
    tickers_per_thread = len(tickers) // num_threads
    ticker_subsections = [tickers[i:i + tickers_per_thread] for i in range(0, len(tickers), tickers_per_thread)]
    
    # If there are leftovers due to integer division, add them to the last subsection
    if len(ticker_subsections) > num_threads:
        ticker_subsections[num_threads - 1].extend(ticker_subsections.pop())

    # Create and start threads for each subsection
    for tickers_subset in ticker_subsections:
        thread = threading.Thread(target=process_tickers, args=(tickers_subset,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    thread_stocks()
