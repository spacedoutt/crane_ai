import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
import gc

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Clear cache and delete variables not in use
torch.cuda.empty_cache()
gc.collect()

# Compile with device-side assertions
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# Load dataset
df = pd.read_csv('Big Mega Set/article_set.csv')#, names=['Sentiment', 'Text'], encoding='ISO-8859-1')

# Split the dataset into a smaller subset for debugging
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)  # Using a very small subset for debugging

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Tokenize the data with reduced max_length
def preprocess_function(examples):
    return tokenizer(examples['Text'].tolist(), truncation=True, padding=True, max_length=32)

def clean_dataset(df):
    if 'Text' not in df.columns:
        raise ValueError("DataFrame must contain a 'Text' column")
    if 'Sentiment' in df.columns:
        df['Sentiment'] = df['Sentiment'].replace({1: 2, 0: 1, -1: 0})
    
    df = df.dropna(subset=['Text'])  # Remove rows with missing 'Text'
    df['Text'] = df['Text'].astype(str)  # Convert all 'Text' values to strings
    return df

train_df = clean_dataset(train_df)
test_df = clean_dataset(test_df)
train_encodings = preprocess_function(train_df)
test_encodings = preprocess_function(test_df)

# Create dataset class
class FinancialNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are of type long
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FinancialNewsDataset(train_encodings, train_df['Sentiment'].tolist())
test_dataset = FinancialNewsDataset(test_encodings, test_df['Sentiment'].tolist())

# Load model
model = AutoModelForSequenceClassification.from_pretrained("microsoft/Phi-3-mini-4k-instruct", num_labels=3)  # Adjust num_labels to 3

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,  # Further reduce batch size
    per_device_eval_batch_size=64,   # Further reduce batch size
    gradient_accumulation_steps=64,  # Accumulate gradients to simulate larger batch size
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',            # Directory for storing logs
    fp16=True,                       # Enable mixed precision
    bf16=False                       # Disable bfloat16 precision for debugging
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Move model to device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Clear CUDA cache
torch.cuda.empty_cache()

# Train the model
try:
    trainer.train()
except RuntimeError as e:
    print(f"Error during training: {e}")
    # Move model to CPU for further debugging
    model.to('cpu')
    trainer.args.device = 'cpu'
    trainer.train()

# Evaluate the model
with torch.no_grad():
    trainer.evaluate()

# Save the model
model.save_pretrained("./financial_sentiment_model_phi")
tokenizer.save_pretrained("./financial_sentiment_model_phi")

# Explicitly delete variables to free up memory
del model
del trainer
torch.cuda.empty_cache()
gc.collect()
