from dev.grpo.reward_function import instruction_reward
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

m_id = "/scratch/public_models/huggingface/Qwen/Qwen2.5-7B-Instruct/"
# m_id = "/scratch/public_models/huggingface/meta-llama/Llama-3.1-8B-Instruct/"
model = AutoModelForCausalLM.from_pretrained(m_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
tok = AutoTokenizer.from_pretrained(m_id)

messages = [
    [{"role": "user", "content": "Respond using solely exlcamation marks as punctuation."}],
    [{"role": "user", "content": "Respond using solely exlcamation marks as punctuation."}],
]
completion = [
    "This! is a generation with all exclamation marks! nice!",
    "This. is a generation with all exclamation marks. nice.",
]

rewards = instruction_reward(
    messages,
    completions=completion,
    ground_truth=["Respond using solely exlcamation marks as punctuation.", "Respond using solely exlcamation marks as punctuation."],
    ref_model=model,
    tokenizer=tok,
)
print(rewards)

completion = [
    "This! is a generation with all exclamation marks! nice!",
    "This. is a generation with all periods. nice.",
]
rewards = instruction_reward(
    messages,
    completions=completion,
    ground_truth=["Respond using solely exlcamation marks as punctuation.", "Respond using solely exlcamation marks as punctuation."],
    ref_model=model,
    tokenizer=tok,
)
print(rewards)

messages = [
    [{"role": "user", "content": "Respond using solely exlcamation marks as punctuation."}],
]
completion = [
    "This! is a generation with all exclamation marks! nice!",
]
rewards = instruction_reward(
    messages,
    completions=completion,
    ground_truth=["Respond using solely exlcamation marks as punctuation.", "Respond using solely exlcamation marks as punctuation."],
    ref_model=model,
    tokenizer=tok,
)
print(rewards)

messages = [
    [{"role": "user", "content": "Respond using solely exlcamation marks as punctuation."}],
]
completion = [
    "This. is a generation with all periods. nice.",
]
rewards = instruction_reward(
    messages,
    completions=completion,
    ground_truth=["Respond using solely exlcamation marks as punctuation.", "Respond using solely exlcamation marks as punctuation."],
    ref_model=model,
    tokenizer=tok,
)
print(rewards)
