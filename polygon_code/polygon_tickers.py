from polygon_code.polygon import get_data
from sort_csv import sort_csv
import os

# Replace 'YOUR_API_KEY' with your actual Polygon.io API key
API_KEY = 'q42WQRfUjTeouBEoaCSC_LHNhDFhzIIb'

def get_tickers():
    params = {
        'apiKey': API_KEY,
        'limit': 1000,  # Maximum allowed limit per request
        'market': 'stocks',  # Only include stocks
    }
    base_url = 'https://api.polygon.io/v3/reference/tickers'
    stored_filename = 'polygon_tickers.csv'
    get_data(params, base_url, stored_filename, t_o_s='t')
    

if __name__ == "__main__":
    get_tickers()
    sort_csv(os.path.join("polygon_data", 'polygon_tickers.csv'), 'ticker')