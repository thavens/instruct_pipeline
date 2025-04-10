# train_grpo.py
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
import os
from peft import LoraConfig
import random

os.environ["WANDB_PROJECT"] = "tldr"

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:20%]")


# Filter dataset based on the length of the "answer" field
def filter_answers(example):
    return 500 < len(example["answer"]) < 1000


filtered_dataset = dataset.filter(filter_answers)[:20000]
filtered_dataset = Dataset.from_dict(filtered_dataset)


# Format dataset into OpenAI messages format
def format_messages(example):
    return {
        "prompt": [
            {"role": "user", "content": example["query"]},
        ]
    }


formatted_dataset = filtered_dataset.map(format_messages)
formatted_dataset = formatted_dataset.remove_columns(["query", "answer"])


def reward_punc(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        periods = text.count(".")
        exclamations = text.count("!")

        ratio = exclamations / (periods + exclamations + 1e-8)
        if ratio > random.random():
            rewards.append(1)
        else:
            rewards.append(0)
    return rewards


# training_args = GRPOConfig(
#     output_dir="Qwen2-0.5B-GRPO",
#     logging_steps=10,
#     max_grad_norm=1,
#     per_device_train_batch_size=2,
#     num_generations=2,
#     gradient_accumulation_steps=4,
#     include_tokens_per_second=True,
#     skip_memory_metrics=False,
#     report_to="wandb",
#     fsdp=False,
# )
training_args = GRPOConfig(
    output_dir="/storage_fast/models/michael_lavery/50_grpo",
    logging_steps=1,
    learning_rate=3e-6,
    beta=0.1,
    max_grad_norm=0.1,
    adam_beta1=0.9,
    adam_beta2=0.99,
    # weight_decay=0.1,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    num_generations=16,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    report_to="wandb",
    run_name=None,
    num_train_epochs=1,
    max_prompt_length=512,
    max_completion_length=256,
    save_steps=100,
)

trainer = GRPOTrainer(
    model="/storage_fast/models/michael_lavery/50/checkpoint-450",
    reward_funcs=reward_punc,
    args=training_args,
    train_dataset=formatted_dataset,
)
trainer.train()
