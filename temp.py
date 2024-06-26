import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def get_article_details(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract the title and content
            title = soup.find('title').text.strip()
            content_paragraphs = soup.find_all('p')
            content = ' '.join([para.text.strip() for para in content_paragraphs])
            
            return title, content
        else:
            return None, None
    except Exception as e:
        return None, None

# Load the CSV file
file_path = os.path.join('polygon_data', 'polygon_news.csv')
df = pd.read_csv(file_path)

# Iterate through the DataFrame and update the content and title
for index, row in df.iterrows():
    correct_title, correct_content = get_article_details(row['Link'])
    
    if correct_title and correct_content:
        df.at[index, 'Title'] = correct_title
        df.at[index, 'Content'] = correct_content

# Save the updated DataFrame to a new CSV file
updated_file_path = os.path.join('polygon_data', 'updated_polygon_news.csv')
df.to_csv(updated_file_path, index=False)

print("The CSV file has been updated and saved.")
