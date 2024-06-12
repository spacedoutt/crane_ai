import argparse
# important for chatbot to work
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# important for training
import json
import re
from pprint import pprint

import pandas as pd
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

parser = argparse.ArgumentParser(description="Chat with the Phi-3-mini-4k-instruct model.")
parser.add_argument("--type", type=str, default="train", help="Type of interaction with the model. Options: chat, train")
parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="Name of the model to use. Default: microsoft/Phi-3-mini-4k-instruct")
parser.add_argument("--compare-trained-model", type=bool, default=False, help="Compare the trained model with the original model. Default: False")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

def chat(model_name: str, test_train: bool):
    torch.random.manual_seed(0)
    model = None
    tokenizer = None
    model, tokenizer, _ = create_mtl(model_name)
    trained_model = PeftModel.from_pretrained(model, "outputs") if test_train else None
    messages = [
        {"role": "system", "content": "You are an expert on sentiment analysis of company statistics. You will determine which company has negative sentiment and which company has positive sentiment."},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    train_pipe = pipeline(
        "Phi3ForCausalLM",
        model=trained_model,
        tokenizer=tokenizer,
    ) if test_train else None

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
        if test_train:
            train_output = train_pipe(messages, **generation_args)
            print(f"Train model: {train_output[0]['generated_text']}")
            print(f"Original model: {output[0]['label']}")
        else:
            # Extract and print the generated text
            response = output[0]['generated_text']
            print(f"Chatbot: {response}")

        # Append chatbot response to messages list
        messages.append({"role": "assistant", "content": response})

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
        evaluation_strategy="no",
        do_eval=False,
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
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.model_name
    test_train = args.compare_trained_model
    if args.type == "chat":
        chat(model_name, test_train)
    elif args.type == "train":
        train(model_name)
    else:
        raise ValueError(f"Invalid type argument: {args.type}. Options: chat, train")