from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from trl import GRPOConfig, GRPOTrainer
import os
from tqdm.contrib.concurrent import thread_map
from copy import deepcopy

from peft import LoraConfig

from dev.grpo.reward_function import instruction_reward, easy_reward

os.environ["WANDB_PROJECT"] = "tldr"

dataset = load_dataset(
    "normster/RealGuardrails", "train_sft", split="train[:2%]"
).to_list()

# Format dataset into OpenAI messages format
def format_messages(example):
    rule = "Respond using solely exclamation marks as punctuation."
    messages = deepcopy(example["messages"])
    messages[0]["content"] = messages[0]["content"] + "\n" + rule
    messages = messages[:2]
    return {"prompt": messages, "ground_truth": rule}


formatted_dataset = Dataset.from_list(
    thread_map(format_messages, dataset, max_workers=128)
)

training_args = GRPOConfig(
    output_dir="/storage_fast/models/michael_lavery/system/",
    logging_steps=1,
    learning_rate=3e-6,
    beta=0.1,
    max_grad_norm=1,
    adam_beta1=0.9,
    adam_beta2=0.99,
    # weight_decay=0.1,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    num_generations=24,  # 16
    per_device_train_batch_size=8,  # 8
    gradient_accumulation_steps=4,
    num_iterations=4,
    report_to="wandb",
    run_name=None,
    num_train_epochs=3,
    max_prompt_length=512,
    max_completion_length=512,
    save_steps=100,
    bf16=True,
    gradient_checkpointing=True,
    use_vllm=True,
    optim="adamw_torch_fused",
)

model_name = "/scratch/public_models/huggingface/Qwen/Qwen2.5-1.5B-Instruct/"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
)
tokenizer.padding_side = "left"

peft_config = LoraConfig(
    r=128,
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ],
    task_type="CAUSAL_LM",
    lora_dropout=0.1,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    # reward_funcs=instruction_reward,
    reward_funcs=easy_reward,
    args=training_args,
    train_dataset=formatted_dataset,
    # peft_config=peft_config,
)
trainer.train()

# CUDA_VISIBLE_DEVICES=1,2 uv run accelerate launch --num_processes 2 --downcast_bf16 --config_file Trl/examples/accelerate_configs/deepspeed_zero3.yaml dev/grpo/run.py
