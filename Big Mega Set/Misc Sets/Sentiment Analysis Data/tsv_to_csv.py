import pandas as pd

# Read the TSV file
tsv_file = 'Restaurant_Reviews.tsv'
df = pd.read_csv(tsv_file, sep='\t')

# Write the dataframe to a CSV file
csv_file = 'Restaurant_Reviews.csv'
df.to_csv(csv_file, index=False)
