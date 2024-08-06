import pandas as pd

file_path = "all-data.csv"

# Attempting to read the CSV file again with different encoding
df = pd.read_csv(file_path, encoding='latin1')

# Swap columns 1 and 2
df = df[[df.columns[1], df.columns[0]]]

# Rename the columns
df.columns = ['Text', 'Sentiment']

# Convert Sentiment values
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)

# Save the modified DataFrame to a new CSV file
output_file_path = 'modified_all-data.csv'
df.to_csv(output_file_path, index=False)
