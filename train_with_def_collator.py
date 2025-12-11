from numpy import dtypes
import torch
from datasets import load_dataset
from huggingface_hub import login

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq 
)

login("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")

# 1. Setup Model and Tokenizer
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    use_cache=False, 
    # attn_implementation="flash_attention_2"
)

# 2. Dataset Processing
ds = load_dataset("netop/TeleLogs")
train_ds = ds["train"]

def format_and_tokenize(example):
    # Chat Template automatically handles roles and special tokens
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]}
    ]
    
    # 1. Tokenize the full conversation (User + Assistant)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    tokenized_full = tokenizer(
        full_text,
        add_special_tokens=False 
    )
    
    input_ids = tokenized_full["input_ids"]
    labels = input_ids.copy()

    # 2. Create the Mask (Hide User Prompt from Loss)
    # We re-tokenize just the user part to find the boundary
    user_msgs = messages[:-1]
    user_text = tokenizer.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
    user_tokens = tokenizer(user_text, add_special_tokens=False)["input_ids"]
    
    user_len = len(user_tokens)

    # Set User tokens in 'labels' to -100 so they are ignored in loss calculation
    for i in range(user_len):
        if i < len(labels):
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": tokenized_full["attention_mask"]
    }

# Apply processing
train_ds = train_ds.map(format_and_tokenize, remove_columns=train_ds.column_names)

# 3. Training Arguments (VRAM Optimized)
training_args = TrainingArguments(
    output_dir="./results",
    
    # --- VRAM Critical Settings ---
    bf16=True,                    # CRITICAL: Ensures Optimizer runs in bf16 (Saves ~50% VRAM)
    gradient_checkpointing=True,  # CRITICAL: Trades 20% speed for 60-70% VRAM savings
    
    # --- Batching ---
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=4, # Simulates a batch size of 8 (2*4)
    
    # --- Standard Settings ---
    num_train_epochs=1,
    logging_steps=10,
    learning_rate=2e-5,
    save_strategy="epoch",
    warmup_ratio=0.03,
    report_to="none"
)

# 4. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True), 
)

trainer.train()