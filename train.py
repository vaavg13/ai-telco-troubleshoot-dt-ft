from huggingface_hub import login

login("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")

# 1) Imports and model/tokenizer
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

import numpy as np

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="bfloat16",
)

ds = load_dataset("netop/TeleLogs")
train_ds = ds["train"]
test_ds = ds["test"]

def turn_dataset_into_chat_ds(dataset):
    def preprocess_function(examples):
        questions = examples["question"]
        answers = examples["answer"]

        chats = []
        for question, answer in zip(questions, answers):
            # Append the EOS token to the response
            # answer += eos_token

            chat = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

            chats.append(chat)
        return {"messages": chats}

    dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    return dataset

train_ds = turn_dataset_into_chat_ds(train_ds)

def tokenize_chat(example):
    # Full conversation (includes assistant answer)
    full_text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

    # Only the user question (prepare generation)
    user_only = tokenizer.apply_chat_template(
        example["messages"][:-1],   # drop last assistant msg
        tokenize=False,
        add_generation_prompt=True  # so the model expects an answer
    )

    # Tokenize both
    full_tokens = tokenizer(full_text, return_tensors=None)
    user_tokens = tokenizer(user_only, return_tensors=None)

    input_ids = full_tokens["input_ids"]
    labels = [-100] * len(input_ids)

    # Assistant answer tokens start AFTER user-only tokens
    answer_start = len(user_tokens["input_ids"])
    
    # Assign labels to assistant portion
    for i in range(answer_start, len(input_ids)):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }

train_ds = train_ds.map(tokenize_chat, remove_columns=["messages"])


from dataclasses import dataclass
import torch

@dataclass
class DataCollatorForChat:
    tokenizer: any

    def __call__(self, examples):
        # Extract input_ids and labels
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples]
        labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in examples]

        # Pad to max batch length
        batch_input = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )

        attention_mask = (batch_input != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": batch_input,
            "labels": batch_labels,
            "attention_mask": attention_mask
        }

collator = DataCollatorForChat(tokenizer)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,  # smaller to avoid OOM
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    num_train_epochs=1,
    logging_steps=50,
    save_strategy="epoch",
    # eval_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    # eval_dataset=tokenized_eval_dataset,
    data_collator=collator,
    processing_class=tokenizer,
)


# Train
trainer.train()