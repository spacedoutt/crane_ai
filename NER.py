import spacy
from polygon_data import get_news_data

# Load SpaCy's NER model and stop words
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words

def extract_entities(text):
    # Process the text through the NER model
    doc = nlp(text)
    
    # Lists to hold company names and tickers
    companies = []
    tickers = []

    # Iterate over recognized entities
    for ent in doc.ents:
        if ent.label_ == "ORG":
            companies.append(ent.text)
        elif ent.label_ == "MONEY":  # Assuming tickers are tagged with MONEY or other appropriate tag
            tickers.append(ent.text)
    
    return companies, tickers

def clean_and_split_company_name(company):
    # Split the company name into words and remove stop words
    words = [word for word in company.split() if word.lower() not in stop_words]
    return words

def company_occurences(companies, content):
    company_counts = {}
    for company in companies:
        words = clean_and_split_company_name(company)
        count = 0
        for word in words:
            count += content.count(word)
        company_counts[company] = count
    return company_counts

def perform_ner():
    # Get news data
    news_data = get_news_data()
    i = 0
    while i < len(news_data):
        title = news_data["Title"][i]
        content = news_data["Content"][i]
        link = news_data["Link"][i]
        companies, tickers = extract_entities(content)
        company_counts = company_occurences(companies, content)
        print(f"Title: {title}")
        print(f"Companies: {companies}")
        print(f"Company Occurrences: {company_counts}")
        print(f"Tickers: {tickers}\n")
        i += 1

if __name__ == "__main__":
    perform_ner()
