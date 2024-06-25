import spacy
from polygon_news import get_news_data

# Load SpaCy's NER model and stop words
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words

def extract_entities(text):
    # Process the text through the NER model
    doc = nlp(text)
    
    # Lists to hold company names and tickers
    companies = []
    tickers = []

    # Get false positive and remove it
    false_company = []

    # Iterate over recognized entities
    for ent in doc.ents:
        if ent.label_ == "ORG" and not is_person_name(ent, doc) and not false_company.__contains__(ent.text):
            companies.append(ent.text)
        elif ent.label_ == "ORG" and is_person_name(ent, doc):
            false_company.append(ent.text)
            if ent.text in companies:
                companies.remove(ent.text)
        elif ent.label_ == "MONEY":  # Assuming tickers are tagged with MONEY or other appropriate tag
            tickers.append(ent.text)
    
    return companies, tickers

def is_person_name(ent, doc):
    # Check if the entity is preceded by a title indicating a person
    title_tokens = {"Mr.", "Ms.", "Mrs.", "Dr.", "Prof.", "Sir", "Madam"}
    prev_token = doc[ent.start - 1] if ent.start > 0 else None
    if prev_token and prev_token.text in title_tokens:
        return True
    # Check if the entity is labeled as a person elsewhere in the text
    for e in doc.ents:
        if e.text == ent.text and e.label_ == "PERSON":
            return True
    return False

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
