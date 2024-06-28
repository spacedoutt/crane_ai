import pandas as pd

if __name__ == "__main__":
    news_data = pd.read_csv('polygon_data/day_aggs/2024/06/2024-06-26.csv.gz', compression='gzip')
    print(news_data.head())
    # print all the tickers in column ticker, and count how many tickers there are and print the number of tickers at the end
    print(news_data['ticker'])
    print(news_data['ticker'].count())
