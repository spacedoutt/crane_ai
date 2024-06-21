import spacy
from polygon_data import get_news_data

# Load SpaCy's NER model
nlp = spacy.load("en_core_web_sm")

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

def perform_ner():
    # Get news data
    news_data = get_news_data()
    print(news_data)

if __name__ == "__main__":
    perform_ner()