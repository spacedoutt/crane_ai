import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sort_csv import sort_csv
import os
import chardet

# Define the base URL of Polygon's news section
base_url = 'https://www.polygon.com/news/archives'

# Prepare lists to store scraped data
titles = []
contents = []
links = []
dates = []

def load_existing_links(csv_file_path=os.path.join('polygon_data', 'polygon_news.csv')):
    try:
        news_data = pd.read_csv(csv_file_path)
        i = 0
        existing_links = set()
        for link in news_data['Link']:
            if news_data['Title'][i].startswith("2024"):
                print(news_data['Title'][i])
            if news_data['Content'][i] != None and news_data['Content'][i] != "" and news_data['Content'][i] != np.nan:
                temp = news_data['Content'][i]
                existing_links.add(link)
            i += 1
    except FileNotFoundError:
        existing_links = set()
    return existing_links

def save_to_csv(new_data: pd.DataFrame, csv_file_path=os.path.join('polygon_data', 'polygon_news.csv')):
    try:
        news_data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty DataFrame with the necessary columns
        news_data = pd.DataFrame(columns=['Title', 'Content', 'Link', 'Date'])

    updated_news_data = pd.concat([news_data, new_data], ignore_index=True)

    updated_news_data.drop_duplicates(subset='Link', keep='last', inplace=True)

    updated_news_data.to_csv(csv_file_path, index=False)

    return updated_news_data

# Define a function to scrape the content of an article
def scrape_article_content(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    
    encoding = chardet.detect(response.content)['encoding']
    response.encoding = encoding

    soup = BeautifulSoup(response.content, 'html.parser')
    content_tag = soup.find('div', class_='l-col__main')
    if content_tag:
        paragraphs = content_tag.find_all('p')
        content = '\n'.join([p.get_text() for p in paragraphs])
        return content
    return None

# Define a function to scrape articles from a given URL
def scrape_articles(url: str, existing_links):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f'Failed to load page {url}')

    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('div', class_='c-compact-river__entry')

    for article in articles:
        title_tag = article.find('h2', class_='c-entry-box--compact__title')
        if title_tag:
            title = title_tag.get_text()
            link = title_tag.find('a')['href']
            if link in existing_links:
                continue  # Skip already scraped links
            content = scrape_article_content(link)
        else:
            title = None
            link = None
            content = None

        date_tag = article.find('time', class_='c-byline__item')
        date = date_tag['datetime'] if date_tag else None

        if title != None and content != None and link != None and date != None:
            titles.append(title)
            contents.append(content)
            links.append(link)
            dates.append(date)

def get_news_data():
    # Load existing links to avoid duplicates
    existing_links = load_existing_links()

    # Loop to handle pagination
    page_number = 1
    while page_number <= 15:
        if page_number == 1:
            current_url = base_url
        else:
            current_url = f'{base_url}/{page_number}'
        scrape_articles(current_url, existing_links)
        page_number += 1

    # Create a DataFrame using the extracted data
    new_data = pd.DataFrame({
        'Title': titles,
        'Content': contents,
        'Link': links,
        'Date': dates
    })

    news_data = save_to_csv(new_data)

    return news_data

if __name__ == '__main__':
    news_data = get_news_data()
    sort_csv(os.path.join('polygon_data', 'polygon_news.csv'), 'Date')