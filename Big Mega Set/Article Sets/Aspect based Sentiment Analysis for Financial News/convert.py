import pandas as pd
import ast

# Load the CSV file
file_path = 'SEntFiN-v1.1.csv'
df = pd.read_csv(file_path)

# Function to map sentiment string to numerical value
def sentiment_to_num(sentiment_str):
    sentiment_dict = ast.literal_eval(sentiment_str)
    sentiment_value = list(sentiment_dict.values())[0]
    if sentiment_value == "neutral":
        return 0
    elif sentiment_value == "positive":
        return 1
    elif sentiment_value == "negative":
        return -1
    else:
        return None

# Apply the function to the Sentiment column
df['Sentiment'] = df['Sentiment'].apply(sentiment_to_num)

# Save the modified dataframe to a new CSV file
output_path = 'SEntFiN-v1.1_converted.csv'
df.to_csv(output_path, index=False)

output_path
