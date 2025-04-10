import random
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM
import argparse

# parse probability of punctuation modification
parser = argparse.ArgumentParser()
parser.add_argument("--p", type=float, default=0.3)
args = parser.parse_args()

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:80%]")


# Filter dataset based on the length of the "answer" field
def filter_answers(example):
    return 500 < len(example["answer"]) < 1000


filtered_dataset = dataset.filter(filter_answers)[:20000]
filtered_dataset = Dataset.from_dict(filtered_dataset)


# Function to modify punctuation
def modify_punctuation(text, p=0.3):
    text = text.replace("!", ".")  # Replace all exclamations with periods
    chars = list(text)
    for i in range(len(chars)):
        if chars[i] == "." and random.random() < p:
            chars[i] = "!"  # Randomly change some periods back to exclamations
    return "".join(chars)


# Apply punctuation modification
modified_dataset = filtered_dataset.map(
    lambda x: {"query": x["query"], "answer": modify_punctuation(x["answer"], args.p)}
)


# Format dataset into OpenAI messages format
def format_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


formatted_dataset = modified_dataset.map(format_messages)
formatted_dataset = formatted_dataset.remove_columns(["query", "answer"])

# https://arxiv.org/pdf/2412.13337#page=23.80
# What the pros say about learning rate and batch size.
training_args = SFTConfig(
    output_dir=f"/storage_fast/models/michael_lavery/{int(args.p*100)}",
    num_train_epochs=6,
    learning_rate=3e-6,
    lr_scheduler_type="cosine",
    report_to="wandb",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=16,
    save_steps=50,
    logging_steps=1,
    warmup_steps=50,
)

model = AutoModelForCausalLM.from_pretrained(
    "/scr1/public_models/huggingface/Qwen/Qwen2.5-0.5B-Instruct",
    attn_implementation="flash_attention_2",
    torch_dtype="auto",
    use_cache=False,
    attention_dropout=0
)
model.train()
model = model.to("cuda")

trainer = SFTTrainer(
    model,
    train_dataset=formatted_dataset,
    args=training_args,
)
trainer.train()
