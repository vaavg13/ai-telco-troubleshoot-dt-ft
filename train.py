from dataclasses import dataclass
import torch

from huggingface_hub import login
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

login("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")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="bfloat16",
)

ds = load_dataset("netop/TeleLogs")


def turn_dataset_into_chat_ds(dataset):
    def preprocess_function(examples):
        questions = examples["question"]
        answers = examples["answer"]

        chats = []
        for question, answer in zip(questions, answers):
            chat = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            chats.append(chat)

        return {"messages": chats}

    return dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)


def tokenize_chat(example):
    full_text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    user_only = tokenizer.apply_chat_template(
        example["messages"][:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    full_tokens = tokenizer(full_text, return_tensors=None)
    user_tokens = tokenizer(user_only, return_tensors=None)

    input_ids = full_tokens["input_ids"]
    labels = [-100] * len(input_ids)

    answer_start = len(user_tokens["input_ids"])
    for i in range(answer_start, len(input_ids)):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


@dataclass
class DataCollatorForChat:
    tokenizer: any

    def __call__(self, examples):
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples]
        labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in examples]

        batch_input = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        attention_mask = (batch_input != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": batch_input,
            "labels": batch_labels,
            "attention_mask": attention_mask,
        }


train_ds = turn_dataset_into_chat_ds(ds["train"])
test_ds = turn_dataset_into_chat_ds(ds["test"])

train_tokenized = train_ds.map(tokenize_chat, remove_columns=["messages"])
test_tokenized = test_ds.map(tokenize_chat, remove_columns=["messages"])

collator = DataCollatorForChat(tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=50,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    data_collator=collator,
    processing_class=tokenizer,
)

trainer.train()