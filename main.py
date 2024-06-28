import argparse
# important for chatbot to work
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# important for training
import json
import re
from pprint import pprint
import os

import pandas as pd
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Chat with the Phi-3-mini-4k-instruct model.")
parser.add_argument("--type", type=str, default="use", help="Type of interaction with the model. Options: use, train")
parser.add_argument("--model_name", type=str, default="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", help="Name of the model to use. Default: microsoft/Phi-3-mini-4k-instruct")
parser.add_argument("--compare-trained-model", type=bool, default=False, help="Compare the trained model with the original model. Default: False")
parser.add_argument("--ai-type", type=str, default="sentiment", help="Type of AI to use. Options: chat, sentiment")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

losses = []
grad_norms = []
learning_rates = []

def summarize(model, text: str, tokenizer):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0001)
    return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

def create_mtl(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        quantization_config=bnb_config,
        trust_remote_code=True, 
        use_safetensors=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"    
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=[
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "down_proj",
        "gate_proj",
        "v_proj",
    ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    return model, tokenizer, peft_config

def use_model(model_name: str, test_train: bool, ai_type: str = "chat"):
    torch.random.manual_seed(0)
    model, tokenizer, _ = create_mtl(model_name)
    trained_model = PeftModel.from_pretrained(model, "outputs") if test_train else None
    
    if ai_type == "sentiment":
        sentiment_pipe = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)
        
        # Read the CSV file
        df = pd.read_csv(os.path.join('polygon_data', 'polygon_news.csv'))
        
        results = []
        
        # Perform sentiment analysis on each content
        for index, row in df.iterrows():
            content = row['Content']
            title = row['Title']
            link = row['Link']
            
            result = sentiment_pipe(content)
            sentiment = result[0]['label']
            
            results.append({
                "Title": title,
                "Link": link,
                "Sentiment": sentiment,
                "Tickers": []
            })
        
        # Create a new DataFrame
        result_df = pd.DataFrame(results)
        
        # Save the result to a new CSV file
        result_df.to_csv(os.path.join("polygon_data", "news_sentiment.csv"), index=False)
        print("Sentiment analysis completed and saved to polygon_news_sentiment.csv")

    if ai_type == "chat":
        messages = [
            {"role": "system", "content": "You are an expert on sentiment analysis of company statistics. You will determine which company has negative sentiment and which company has positive sentiment. Print out the company name and the sentiment."},
        ]
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        ) if not test_train else None

        generation_args = {
            "max_new_tokens": 600,
            "return_full_text": False,
            "temperature": 0.3,
            "do_sample": True,
        }
        user_input = ""
        while user_input != "exit":
            # Get user input
            user_input = input("User: ")
            if user_input == "exit":
                break
            # Append user message to messages list
            messages.append({"role": "user", "content": user_input})

            # Generate a response
            if test_train:
                output = summarize(model, user_input, tokenizer)
                train_output = summarize(trained_model, user_input, tokenizer)
                pprint(f"Train model: {train_output}")
                pprint(f"Original model: {output}")
            else:
                output = pipe(messages, **generation_args)
                # Extract and print the generated text
                response = output[0]['generated_text']
                print(f"Chatbot: {response}")
                # Append chatbot response to messages list
                messages.append({"role": "assistant", "content": response})

def train(model_name: str):
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split=["train[:10%]", "validation[:10%]", "test[:10%]"])
    print(dataset)
    
    model, tokenizer, peft_config = create_mtl(model_name)
    model.config.use_cache = False

    output_dir = "model_3"

    training_arguments = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        logging_steps=1,
        learning_rate=1e-4,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=2,
        evaluation_strategy="steps",
        eval_steps=0.2,
        warmup_ratio=0.05,
        save_strategy="epoch",
        group_by_length=True,
        output_dir=output_dir,
        report_to="tensorboard",
        save_safetensors=True,
        lr_scheduler_type="cosine",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset[0],
        eval_dataset=dataset[1],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=(1024),
        tokenizer=tokenizer,
        args=training_arguments,
    )

    def log_metrics(logs):
        # protect against the case where the dict doesn't have the necessary keywords
        if 'loss' in logs and 'grad_norm' in logs and 'learning_rate' in logs:
            losses.append(logs['loss'])
            grad_norms.append(logs['grad_norm'])
            learning_rates.append(logs['learning_rate'])

    trainer.log = log_metrics

    torch.cuda.empty_cache()

    trainer.train()
    trainer.save_model(output_dir)

def plot_metrics():
    epochs = range(len(losses))
    scaled_epochs = [2 * epoch / (len(epochs) - 1) for epoch in epochs]
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(scaled_epochs, losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(scaled_epochs, grad_norms, label='Grad Norm')
    plt.xlabel('Epochs')
    plt.ylabel('Grad Norm')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(scaled_epochs, learning_rates, label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.model_name
    test_train = args.compare_trained_model
    ai_type = args.ai_type
    if args.type == "use":
        use_model(model_name, test_train, ai_type)
    elif args.type == "train":
        train(model_name)
        plot_metrics()
    else:
        raise ValueError(f"Invalid type argument: {args.type}. Options: chat, train")