import pandas as pd
import os
import chardet

def sort_csv(file_path, sort_key, ascending=True):
    try:
        # Load the CSV file
        data = pd.read_csv(file_path, encoding='utf-8')

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
    except UnicodeDecodeError:
        # Try a different encoding
        data = pd.read_csv(file_path, encoding='latin1')
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
    sort_csv(file_path, sort_key, ascending)


def sort_polygon_data():
    # Sort the polygon_data/polygon_tickers.csv file by 'ticker' in ascending order
    run_sort("polygon_data/polygon_tickers.csv", "ticker")

    # Sort the polygon_data/polygon_news.csv file by 'Date' in ascending order
    run_sort("polygon_data/polygon_news.csv", "Date")

if __name__ == "__main__":
    sort_polygon_data()
