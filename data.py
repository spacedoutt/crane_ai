from polygon_code.polygon_news import get_news_data
from polygon_code.polygon_tickers import get_tickers
from polygon_code.polygon_stocks import get_stocks
from sort_csv import sort_polygon_data
import time

def get_data():
    get_news_data()
    get_tickers()
    time.sleep(60)
    get_stocks()
    sort_polygon_data()

if __name__ == "__main__":
    get_data()