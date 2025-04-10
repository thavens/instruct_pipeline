# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import os
from peft import LoraConfig

os.environ["WANDB_PROJECT"] = "tldr"

dataset = load_dataset("trl-lib/tldr", split="train")
dataset.map(lambda x: {"prompt": [{"role": "user", "content": x["prompt"]}]})


# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


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
    output_dir="Qwen2-0.5B-GRPO",
    logging_steps=1,
    learning_rate=3e-6,
    beta=0.1,
    max_grad_norm=0.1,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    num_generations=16,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    report_to="wandb",
    run_name=None,
    num_train_epochs=1,
    bf16=True
)

trainer = GRPOTrainer(
    model="/scratch/public_models/huggingface/Qwen/Qwen2-0.5B",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
