import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


BATCH_SIZE = 8

def build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a telecom troubleshooting assistant."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def evaluate(
    model_name: str,
    max_new_tokens: int,
) -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    ds = load_dataset("netop/TeleLogs")
    test_ds = ds["test"]

    correct = 0
    total = 0

    batch_size = BATCH_SIZE
    total_batches = (len(test_ds) + batch_size - 1) // batch_size

    for start in tqdm(range(0, len(test_ds), batch_size), desc="Evaluating", total=total_batches):
        batch = test_ds[start : start + batch_size]  # dict of column -> list
        prompts = [build_prompt(tokenizer, q) for q in batch["question"]]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_lengths = inputs.attention_mask.sum(dim=1)

        for idx, (question, answer) in enumerate(zip(batch["question"], batch["answer"])):
            generated_ids = outputs[idx][input_lengths[idx] :]
            pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if answer.strip() in pred_text:
                correct += 1
            total += 1

    return correct / total if total else 0.0


if __name__ == "__main__":
    accuracy = evaluate(model_name="Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens=16)
    print(f"Accuracy: {accuracy:.2%}")
