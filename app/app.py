# app.py
from flask import Flask, request, jsonify
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import os
from safetensors.torch import load_file

app = Flask(__name__)

# Load the model state dict from safetensors file
state_dict = load_file(os.path.join("financial_sentiment_model_bert_new_set", "model.safetensors"))

# Initialize the model and load the state dict
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model.load_state_dict(state_dict)
model.eval()

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class SentimentRequest(BaseModel):
    text: str

@app.post('/sentiment')
def sentiment():
    request_data = request.get_json()
    sentiment_request = SentimentRequest(**request_data)
    
    inputs = tokenizer(sentiment_request.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1).tolist()[0]

    # Map the predictions to sentiment labels
    sentiment_score = max(predictions)
    sentiment_label = 'negative' if sentiment_score < 0.45 else 'neutral' if sentiment_score < 0.55 else 'positive'

    return jsonify({'label': sentiment_label, 'score': sentiment_score})

if __name__ == '__main__':
    app.run(debug=True)
