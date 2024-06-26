import pandas as pd
import os
import chardet

def ensure_utf8_encoding(csv_file_path=os.path.join('polygon_data', 'polygon_news.csv')):
    try:
        news_data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f'File {csv_file_path} not found.')
        return

    def to_utf8(text):
        try:
            encoding = chardet.detect(text.encode())['encoding']
            return text.encode(encoding).decode('utf-8') if encoding else text
        except:
            return text

    # Ensure all columns are in UTF-8 encoding
    for column in news_data.columns:
        news_data[column] = news_data[column].apply(lambda x: to_utf8(str(x)))

    # Save the updated DataFrame back to the CSV file
    news_data.to_csv(csv_file_path, index=False)

def sort_csv(file_path, sort_key, ascending=True):
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)

        # Check if the sort key exists in the DataFrame
        if sort_key not in data.columns:
            print(f"Error: '{sort_key}' is not a column in the CSV.")
            return

        # Sort the data
        sorted_data = data.sort_values(by=sort_key, ascending=ascending)

        # Save the sorted data back to CSV
        sorted_data.to_csv(file_path, index=False)
        print(f"File sorted and saved successfully.")
        return
    
    except Exception as e:
        print(f"An error occurred: {e}")

def run_sort(file_path, sort_key, ascending=True):
    # Sort the CSV file by the specified key
    ensure_utf8_encoding(file_path)
    sort_csv(file_path, sort_key, ascending)


def sort_polygon_data():
    # Sort the polygon_data/polygon_tickers.csv file by 'ticker' in ascending order
    run_sort("polygon_data/polygon_tickers.csv", "ticker")
    

    # Sort the polygon_data/polygon_news.csv file by 'Date' in ascending order
    run_sort("polygon_data/polygon_news.csv", "Date")

    # For each stock in stocks sort by 'from' in ascending order
    stocks_folder = "polygon_data/stocks"
    for file in os.listdir(stocks_folder):
        if file.endswith(".csv"):
            run_sort(os.path.join(stocks_folder, file), "from")

if __name__ == "__main__":
    sort_polygon_data()
