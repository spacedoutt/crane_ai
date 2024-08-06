import os
import pandas as pd

def collect_csv_files(root_dir):
    """Recursively collect all CSV files from a root directory and its subdirectories."""
    csv_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                csv_files.append(os.path.join(dirpath, filename))
    return csv_files

def read_csv_with_fallback(file_path):
    """Read a CSV file, trying different encodings if necessary."""
    encodings = ['utf-8', 'latin1', 'iso-8859-1']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {file_path} with any of the tried encodings.")

def merge_csv_files(file_list, output_file):
    """Merge multiple CSV files into a single CSV file."""
    df_list = []
    for file in file_list:
        df = read_csv_with_fallback(file)
        df_list.append(df)
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_file, index=False)

# Replace these with your actual directories
root_dirs = ['Media Sets']

all_csv_files = []
for root_dir in root_dirs:
    all_csv_files.extend(collect_csv_files(root_dir))

# Specify the output file
output_csv = 'media_set.csv'

merge_csv_files(all_csv_files, output_csv)

print(f"Successfully merged {len(all_csv_files)} CSV files into {output_csv}")
