import os
import csv
import requests

# Ensure the folder structure exists
main_folder = 'polygon_data'
os.makedirs(main_folder, exist_ok=True)

def api_call(params, base_url):
    """Fetch all available stock tickers from Polygon.io."""
    result = []
    next_url = base_url

    while next_url:
        response = requests.get(next_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                result.extend(data['results'])
            else:
                result.append(data)
            next_url = data['next_url'] if 'next_url' in data else None
        else:
            print(f"Error fetching tickers: {response.status_code} - {response.text}")
            break
        
    return result

def check_existing_data(filename, key):
    """Check and return existing data from a CSV file."""
    existing_data = set()
    if os.path.exists(filename):
        with open(filename, 'r', newline='', encoding='utf-8', errors='ignore') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_data.add(row[key])
    return existing_data

def save_to_csv(data, filename, t_o_s):
    """Append unique tickers to a CSV file, avoiding duplicates based on the 'ticker' key."""
    key = 'ticker' if t_o_s == 't' else 'from'
    existing_data = check_existing_data(filename, key)
    

    # Prepare new unique data to be written
    new_data = [dt for dt in data if dt[key] not in existing_data]

    if new_data:
        keys = new_data[0].keys()
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            if csvfile.tell() == 0:  # Check if the file is empty
                writer.writeheader()
            writer.writerows(new_data)

def get_data(params, base_url, stored_filename, t_o_s='t'):
    data = api_call(params, base_url)

    if data:
        print(f"Retrieved {len(data)} calls.")
        csv_filename = os.path.join(main_folder, stored_filename)
        save_to_csv(data, csv_filename, t_o_s)
        print(f"Data saved to {csv_filename}")
        
    else:
        print("No data retrieved.")
