from polygon_code.polygon_news import get_news_data
from polygon_code.polygon_tickers import get_tickers
from polygon_code.polygon_files import thread_stocks
from sort_csv import sort_polygon_data
import time

def get_data():
    get_news_data()
    get_tickers()
    time.sleep(1)
    thread_stocks()
    sort_polygon_data()

if __name__ == "__main__":
    get_data()