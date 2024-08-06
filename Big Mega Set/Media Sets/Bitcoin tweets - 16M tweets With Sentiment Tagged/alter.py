import pandas as pd

file_path = "mbsa.csv"
df = pd.read_csv(file_path)

# Swap columns 1 and 2
df = df[[df.columns[1], df.columns[2]]]

df.rename(columns={df.columns[0]: 'Text'}, inplace=True)
df.rename(columns={df.columns[1]: 'Sentiment'}, inplace=True)

# Convert Sentiment values
sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)

output_path = "modified_mbsa.csv"
df.to_csv(output_path, index=False)