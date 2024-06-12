import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# important for chatbot to work
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# important for training
import json
import re
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import BitsAndBytesConfig, TrainingArguments
from transformers.integrations import TensorBoardCallback
from trl import SFTTrainer

parser = argparse.ArgumentParser(description="Chat with the Phi-3-mini-4k-instruct model.")
parser.add_argument("--type", type=str, default="train", help="Type of interaction with the model. Options: chat, train")
parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="Name of the model to use. Default: microsoft/Phi-3-mini-4k-instruct")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
def chat(model_name: str):
    torch.random.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype="auto", 
        trust_remote_code=True, 
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    messages = [
        {"role": "system", "content": "You are an expert on the market, stocks, investments, trading and financial news. You are here to help answer questions and provide information on these topics."},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

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
        output = pipe(messages, **generation_args)

        # Extract and print the generated text
        response = output[0]['generated_text']
        print(f"Chatbot: {response}")

        # Append chatbot response to messages list
        messages.append({"role": "assistant", "content": response})

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

def train(model_name: str):
    dataset_all = load_dataset("takala/financial_phrasebank", "sentences_allagree")
    dataset_75 = load_dataset("takala/financial_phrasebank", "sentences_75agree")
    dataset_66 = load_dataset("takala/financial_phrasebank", "sentences_66agree")
    dataset_50 = load_dataset("takala/financial_phrasebank", "sentences_50agree")

    model, tokenizer, peft_config = create_mtl(model_name)
    model.config.use_cache = False

    output_dir = "outputs"

    training_arguments = TrainingArguments(
        per_device_train_batch_size=4,
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
        train_dataset=dataset_all["train"],
        peft_config=peft_config,
        dataset_text_field="sentence",
        max_seq_length=4096,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    training_metrics = {
        "loss": [],
        "grad_norm": [],
        "learning_rate": [],
    }

    def log_metrics(metrics):
        training_metrics["loss"].append(metrics["loss"])
        training_metrics["grad_norm"].append(metrics["max_grad_norm"])
        training_metrics["learning_rate"].append(metrics["learning_rate"])

    trainer.add_callback(log_metrics)
    trainer.train()
    trainer.save_model(output_dir)

def plot_metrics(metrics):
    epochs = range(1, len(metrics["loss"]) + 1)

    plt.figure(figsize=(15, 5))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics["loss"], label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plot Gradient Norm
    plt.subplot(1, 3, 2)
    plt.plot(epochs, metrics["grad_norm"], label="Gradient Norm")
    plt.xlabel("Epochs")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm")
    plt.legend()

    # Plot Learning Rate
    plt.subplot(1, 3, 3)
    plt.plot(epochs, metrics["learning_rate"], label="Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.model_name
    if args.type == "chat":
        chat(model_name)
    elif args.type == "train":
        train(model_name)
    else:
        raise ValueError(f"Invalid type argument: {args.type}. Options: chat, train")