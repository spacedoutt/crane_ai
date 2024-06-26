import pandas as pd
import os

def sort_csv(file_path, sort_key, ascending=True):
    encodings = ['utf-8', 'ISO-8859-1']  # Add more if necessary
    for enc in encodings:
        try:
            # Load the CSV file
            data = pd.read_csv(file_path, encoding=enc)

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

def sort_polygon_data():
    # Sort the polygon_data/polygon_tickers.csv file by 'ticker' in ascending order
    sort_csv("polygon_data/polygon_tickers.csv", "ticker")

    # Sort the polygon_data/polygon_news.csv file by 'Date' in ascending order
    sort_csv("polygon_data/polygon_news.csv", "Date")

    # For each stock in stocks sort by 'from' in ascending order
    stocks_folder = "polygon_data/stocks"
    for file in os.listdir(stocks_folder):
        if file.endswith(".csv"):
            sort_csv(os.path.join(stocks_folder, file), "from")

if __name__ == "__main__":
    sort_polygon_data()
